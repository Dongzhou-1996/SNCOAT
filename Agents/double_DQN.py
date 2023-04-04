import argparse
import os
import numpy as np
import glob
import threading
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter
from Envs.SNCOAT_Env_v1 import DISCRETE_ACTIONS
from Models.critic_models import ConvQNet, ResQNet, BasicBlock, Bottleneck
from Utils.replay_memory import Transition, Memory

SCENES_DIR = '/home/group1/dzhou/SNCOAT/Scenes/'



class DoubleDQN(object):
    """Double DQN algorithm"""

    def __init__(self, name='double_dqn', log_dir='log', observe_type='Color', backbone='ConvNet',
                 action_type='Discrete', actuator_type='Position', action_dim=11,
                 replay_buffer_size=50000, batch_size=32, lr=1e-4, gpu_idx=0,
                 gamma=0.99, start_epsilon=0.9, end_epsilon=0.1,update_interval=1000,
                 restore=True, is_training=True):
        super(DoubleDQN, self).__init__()
        self.name = name
        assert action_type in ['Discrete', 'Continuous']
        self.action_type = action_type
        self.action_dim = action_dim
        assert actuator_type in ['Position', 'Velocity', 'Force']
        self.actuator_type = actuator_type
        assert backbone in ['ConvNet', 'ResNet18', 'ResNet34', 'ResNet50']
        self.backbone = backbone
        self.device = torch.device('cuda', gpu_idx) if gpu_idx < torch.cuda.device_count() \
            else torch.device('cpu', gpu_idx)

        self.lr = lr
        self.gamma = gamma
        self.batch_size = batch_size
        self.start_epsilon = start_epsilon
        self.end_epsilon = end_epsilon
        self.update_interval = update_interval
        self.replay_buffer_size = replay_buffer_size
        self.observe_type = observe_type
        self.log_dir = log_dir

        self.episode_counter = 0
        self.global_step = 0
        self.memory_counter = 0

        if self.backbone == 'ConvNet':
            self.eval_net = ConvQNet(observe_type=observe_type, action_dim=self.action_dim).to(self.device)
            self.target_net = ConvQNet(observe_type=observe_type, action_dim=self.action_dim, ).to(self.device)
        elif self.backbone == 'ResNet18':
            self.eval_net = ResQNet(observe_type=observe_type, action_dim=self.action_dim, block=BasicBlock,
                                    num_blocks=[2, 2, 2, 2]).to(self.device)
            self.target_net = ResQNet(observe_type=observe_type, action_dim=self.action_dim, block=BasicBlock,
                                      num_blocks=[2, 2, 2, 2]).to(self.device)
        elif self.backbone == 'ResNet34':
            self.eval_net = ResQNet(observe_type=observe_type, action_dim=self.action_dim, block=BasicBlock,
                                    num_blocks=[3, 4, 6, 3]).to(self.device)
            self.target_net = ResQNet(observe_type=observe_type, action_dim=self.action_dim, block=BasicBlock,
                                      num_blocks=[3, 4, 6, 3]).to(self.device)
        elif self.backbone == 'ResNet50':
            self.eval_net = ResQNet(observe_type=observe_type, action_dim=self.action_dim, block=Bottleneck,
                                    num_blocks=[3, 4, 6, 3]).to(self.device)
            self.target_net = ResQNet(observe_type=observe_type, action_dim=self.action_dim, block=Bottleneck,
                                      num_blocks=[3, 4, 6, 3]).to(self.device)
        else:
            raise ValueError('Unsupported backbone type!')

        self.optimizer = optim.Adam(self.eval_net.parameters(), lr=self.lr)
        self.lr_scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9)
        self.loss_func = nn.MSELoss()

        if restore:
            model_path = self.get_latest_model_path()
            if os.path.exists(model_path):
                print('=> restoring model from {}'.format(model_path))
                checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage.cuda())
                # checkpoint = torch.load(model_path)
                self.eval_net.load_state_dict(checkpoint['eval_net'])
                self.target_net.load_state_dict(checkpoint['target_net'])
                self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
                self.episode_counter = checkpoint['episodes']
                self.global_step = checkpoint['steps'] + 1
            else:
                raise ValueError('No model file is found in {}'.format(model_path))

        # When we store the memory, we put the state, action, reward and next_state in the memory
        # here reward and action is a number, state is a ndarray
        self.memory = Memory(self.replay_buffer_size)

        self.is_training = is_training
        if not self.is_training:
            self.eval_net.eval()
            self.target_net.eval()
        else:
            self.summary_writer = SummaryWriter(log_dir=self.log_dir)

    def choose_action(self, state, env):
        state = torch.from_numpy(np.transpose(state, (2, 0, 1)))
        state = torch.unsqueeze(torch.Tensor(state), 0)
        if self.is_training:
            epsilon = self.start_epsilon * np.exp(- 0.1 * self.episode_counter)
            epsilon = max(self.end_epsilon, epsilon)
            self.summary_writer.add_scalar('epsilon_factor', epsilon, self.episode_counter)

            if np.random.rand() >= epsilon:
                # greedy policy
                action_value = self.eval_net.forward(state.to(self.device))
                indx = torch.argmax(action_value, 1)[0].item()
            else:
                # random policy
                indx = np.random.randint(0, len(DISCRETE_ACTIONS))
        else:
            # greedy policy
            action_value = self.eval_net.forward(state.to(self.device))
            indx = torch.argmax(action_value, 1)[0].item()

        action = DISCRETE_ACTIONS[indx]

        return indx, action

    def store_transition(self, state, action_idx, reward, next_state, done):
        transition = Transition(state, action_idx, reward, next_state, done)
        self.memory.push(transition)
        self.memory_counter += 1

    def learn(self):
        # update the parameters
        self.global_step += 1

        # sample batch from memory
        batch_states, batch_action_idxs, batch_rewards, batch_next_states, batch_dones = self.memory.sample(self.batch_size)

        batch_states = torch.FloatTensor(np.transpose(batch_states, (0, 3, 1, 2))).to(self.device)
        batch_next_states = torch.FloatTensor(np.transpose(batch_next_states, (0, 3, 1, 2))).to(self.device)
        batch_action_idxs = torch.LongTensor(batch_action_idxs).unsqueeze(1).to(self.device)
        batch_rewards = torch.FloatTensor(batch_rewards).to(self.device)
        batch_dones = torch.FloatTensor(1 - batch_dones).to(self.device)

        # q_eval
        q_eval = self.eval_net(batch_states).gather(1, batch_action_idxs).squeeze(1)
        q_next_eval = self.eval_net(batch_next_states)
        q_next_target = self.target_net(batch_next_states)
        q_next = q_next_target.gather(1, torch.max(q_next_eval, 1)[1].unsqueeze(1)).squeeze(1)

        q_target = batch_rewards + self.gamma * q_next
        loss = self.loss_func(q_eval, q_target.to(self.device)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # if self.global_step % 1e4 == 0:
        #     self.lr_scheduler.step()

        self.summary_writer.add_scalar('loss/value_loss', loss, self.global_step)
        self.summary_writer.add_scalar('Q/action_value', q_eval.sum() / self.batch_size, self.global_step)

        return loss.item()

    def get_latest_model_path(self):
        model_paths = glob.glob(os.path.join(self.log_dir, 'model_*.pth'))
        if len(model_paths) > 0:
            print('=> found {} models in {}'.format(len(model_paths), self.log_dir))
            created_times = [os.path.getmtime(path) for path in model_paths]
            latest_path = model_paths[np.argmax(created_times)]
            print('=> the latest model path: {}'.format(latest_path))
            return latest_path
        else:
            raise ValueError('No pre-trained model found!')

    def reset(self, is_record='', record_path=''):
        return


    def state_transform(self, state, env):
        """
        Normalize depth channel of state
        :param state: the state of observation
        :param env: the simulation env
        :return: norm_image
        """
        if self.observe_type == 'Color':
            norm_image = state
        elif self.observe_type == 'Depth':
            norm_image = state / env.cam_far_distance
        elif self.observe_type == 'RGBD':
            image = state[:, :, :3]
            depth = state[:, :, -1] / env.cam_far_distance
            norm_image = np.append(image, np.expand_dims(depth, 2), axis=2)
        else:
            raise ValueError('Unsupported observation type!')
        return norm_image

    @staticmethod
    def action_pos_2_force(pos_action, env):
        return 2 * env.chaser_mass * pos_action / (env.delta_time ** 2)

    @staticmethod
    def action_vel_2_force(vel_action, env):
        return vel_action / env.delta_time