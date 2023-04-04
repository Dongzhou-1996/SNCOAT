import os

import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import Utils.data_augs as rad
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from collections import namedtuple
from Models.actor_models import Conv_LSTM_Actor, Conv_LSTM_MHA_Actor
from Models.critic_models import Conv_LSTM_QNet, Conv_LSTM_MHA_QNet, ConvQNet
from Utils.structured_memory import Memory, Transition
from Config.env_setting import IMAGE_CHANNELS
from Envs.SNCOAT_Env_v2 import SNCOAT_Env_v2
from evaluation import AVTEval
from typing import Tuple, Union

aug_to_func = {
    'Crop': rad.random_crop,
    'Cutout': rad.random_cutout,
    'CutoutColor': rad.random_cutout_color,
    'Flip': rad.random_flip,
    'Rotate': rad.random_rotation,
    'SensorNoise': rad.random_sensor_noise,
    'No': rad.no_aug,
}


class RDDPG(object):
    def __init__(
            self, name: str, log_dir='./log',
            state_type='Image', image_type='Color',
            action_type='Continuous', actuator_type='Velocity',
            attention_type='No', data_augs=None,
            pretrained_model=None,
            action_dim=3, gpu_idx=0, seed=1,
            replay_buffer_size=50000,
            init_buffer_size=10000,
            episode_nums=500,
            min_history=4, state_to_update=8,
            max_ep_len=1000, max_action=2,
            start_epsilon=0.9, end_epsilon=0.1,
            epsilon_decay_steps=100000,
            lr=1e-4, tau=1e-4, gamma=0.99,
            lstm_dim=512, lstm_layers=2,
            batch_size=32, cam_far_distance=20,
            is_train=True, restore=False,
            with_SE=True, headless=True, vis=False,
    ):
        self.name = name
        assert state_type in ['Image', 'PosImage', 'OrientImage', 'PoseImage', 'Pose', 'Position'], print(
            'Unsupported state type!')
        self.state_type = state_type
        assert image_type in ['Color', 'Depth', 'RGBD'], print('Unsupported image type!')
        self.image_type = image_type
        assert action_type in ['Discrete', 'Continuous'], \
            print('Unsupported action type!')
        self.action_type = action_type
        self.action_dim = action_dim
        assert actuator_type in ['Force', 'Velocity', 'Position'], \
            print('Unsupported actuator type!')
        self.actuator_type = actuator_type
        assert attention_type in ['No', 'Add', 'DotProd', 'MHA'], print('Unsupported attention mechanism!')
        self.attention_type = attention_type

        self.device = torch.device('cuda', gpu_idx) if gpu_idx < torch.cuda.device_count() \
            else torch.device('cpu', gpu_idx)
        self.seed(seed)

        self.data_augs = {}
        for aug_name in data_augs:
            assert aug_name in aug_to_func, 'invalid data augmentation method'
            self.data_augs[aug_name] = aug_to_func[aug_name]

        self.lr = lr
        self.tau = tau
        self.vis = vis
        self.gamma = gamma
        self.replay_buffer_size = replay_buffer_size
        self.init_buffer_size = init_buffer_size
        self.max_action = max_action
        self.max_ep_len = max_ep_len
        self.episode_nums = episode_nums
        self.history_len = 1
        self.min_history = min_history
        self.state_to_update = state_to_update
        self.lstm_dim = lstm_dim
        self.lstm_layers = lstm_layers
        self.batch_size = batch_size
        self.start_epsilon = start_epsilon
        self.end_epsilon = end_epsilon
        self.epsilon_decay_steps = epsilon_decay_steps
        self.is_train = is_train
        self.headless = headless
        self.restore = restore
        self.pretrained_model = pretrained_model
        self.log_dir = log_dir
        if not os.path.exists(self.log_dir):
            print('=> log directory is not existed! it will be created soon ...')
            os.makedirs(self.log_dir)

        self.episode_counter = 0
        self.global_step = 0


        self.cam_far_distance = cam_far_distance
        self.image_channels = IMAGE_CHANNELS[self.image_type] * self.history_len

        ## create memory pool
        self.memory = Memory(max_replay_experiences=self.replay_buffer_size,
                             episode_len=self.max_ep_len, history_len=self.min_history+self.state_to_update,
                             aug_funcs=self.data_augs)

        if self.state_type == 'Image':
            if self.attention_type == 'No':
                self.actor = Conv_LSTM_Actor(
                    input_channels=self.image_channels, output_dim=self.action_dim, max_action=self.max_action,
                    lstm_dim=self.lstm_dim, lstm_layers=self.lstm_layers, with_SE=with_SE, vis=self.vis,
                    activation='Tanh'
                ).to(self.device)
                self.target_actor = Conv_LSTM_Actor(
                    input_channels=self.image_channels, output_dim=self.action_dim, max_action=self.max_action,
                    lstm_dim=self.lstm_dim, lstm_layers=self.lstm_layers, with_SE=with_SE, vis=self.vis,
                    activation='Tanh'
                ).to(self.device)
                self.critic = ConvQNet(
                    image_channels=self.image_channels, action_dim=self.action_dim, with_SE=with_SE, vis=self.vis
                ).to(self.device)
                self.target_critic = ConvQNet(
                    image_channels=self.image_channels, action_dim=self.action_dim, with_SE=with_SE, vis=self.vis
                ).to(self.device)
            else:
                self.actor = Conv_LSTM_MHA_Actor(
                    input_channels=self.image_channels, output_dim=self.action_dim, max_action=self.max_action,
                    lstm_dim=self.lstm_dim, lstm_layers=self.lstm_layers, with_SE=with_SE, vis=self.vis,
                    activation='Tanh', attention_type=self.attention_type, head_num=8,
                ).to(self.device)
                self.target_actor = Conv_LSTM_MHA_Actor(
                    input_channels=self.image_channels, output_dim=self.action_dim, max_action=self.max_action,
                    lstm_dim=self.lstm_dim, lstm_layers=self.lstm_layers, with_SE=with_SE, vis=self.vis,
                    activation='Tanh', attention_type=self.attention_type, head_num=8,
                ).to(self.device)
                self.critic = ConvQNet(
                    image_channels=self.image_channels, action_dim=self.action_dim, with_SE=with_SE, vis=self.vis
                ).to(self.device)
                self.target_critic = ConvQNet(
                    image_channels=self.image_channels, action_dim=self.action_dim, with_SE=with_SE, vis=self.vis,
                ).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=10 * self.lr)
        self.eval_lstm_tuple = (Variable(torch.zeros(self.lstm_layers, 1, self.lstm_dim).float().to(self.device)),
                                Variable(torch.zeros(self.lstm_layers, 1, self.lstm_dim).float().to(self.device)))


        self.actor_lr_scheduler = optim.lr_scheduler.ExponentialLR(
            self.actor_optimizer, gamma=0.9
        )
        self.critic_lr_scheduler = optim.lr_scheduler.ExponentialLR(
            self.critic_optimizer, gamma=0.9
        )

        if self.restore:
            if pretrained_model is not None:
                model_path = pretrained_model
                if os.path.exists(model_path):
                    print('=> found pretrained mode: {}'.format(model_path))
                else:
                    raise ValueError('pretrained model path is not found!')
            else:
                model_path = self.get_latest_model_path()
            self.reload_model(model_path, is_train=is_train)

        if self.is_train:
            self.actor.train()
            self.target_actor.train()
            self.critic.train()
            self.target_critic.train()
            self.summary_writer = SummaryWriter(log_dir=self.log_dir)
        else:
            self.actor.eval()
            self.critic.eval()
            self.target_actor.eval()
            self.target_critic.eval()

    def learn(self, evaluator: AVTEval, scene_paths: list,
              init_buffer_size=10000, episode_nums=500,
              dis_penalty_f=0.1, outview_penalty=5,
              multi_scene=True, eval_interval=50, record_interval=10,):
        if init_buffer_size is not None:
            self.init_buffer_size = init_buffer_size
        if episode_nums is not None:
            self.episode_nums = episode_nums

        self.memory_initialization(
            self.init_buffer_size, scene_paths, multi_scene, dis_penalty_f, outview_penalty)

        print('\n=> Start training ...')

        if multi_scene:
            scene_path = scene_paths[np.random.randint(0, len(scene_paths))]
        else:
            scene_path = scene_paths[0]
        print('=> reloading scene in {} ...'.format(scene_path))
        env = SNCOAT_Env_v2(
            name='SNCOAT_Env', scene_path=scene_path, log_dir=self.log_dir,
            action_type=self.action_type, state_type=self.state_type, image_type=self.image_type,
            actuator_type=self.actuator_type, headless=self.headless, clear_record=False,
            dis_penalty_f=dis_penalty_f, outview_penalty=outview_penalty,
            wait_done_steps=10, history_len=self.history_len,
            actuator_noise=False, time_delay=False, image_blur=False)

        for i in range(self.episode_counter, self.episode_nums):
            self.episode_counter += 1
            print('=> training on {}th episode ...'.format(self.episode_counter))

            if i % record_interval == 0:
                record = True
            else:
                record = False

            ep_reward, actor_loss, critic_loss = self.rollout_v2(env, is_record=record)
            print('=> [Train]: episode num: {}, episode length: {}, episode reward: {:0.3f}, actor loss: {:0.3f}, critic loss: {:0.3f}'.format(
                i+1, env.step_count, ep_reward, actor_loss, critic_loss))

            if (i + 1) % eval_interval == 0:
                env.stop()
                env.shutdown()

                model_path = os.path.join(self.log_dir, 'model_ep_{:03d}.pth'.format(self.episode_counter + 1))
                self.save_model(model_path)

                self.mode_transfer(is_training=False)
                evaluator.eval([self], action_type=self.action_type, actuator_type=self.actuator_type,
                               eval_episode=self.episode_counter, max_episode_len=self.max_ep_len,
                               headless=True, image_blur=False, actuator_noise=False,
                               time_delay=False, overwrite=True)
                self.mode_transfer(is_training=True)

                if multi_scene:
                    scene_path = scene_paths[np.random.randint(0, len(scene_paths))]
                else:
                    scene_path = scene_paths[0]
                print('=> reloading scene in {} ...'.format(scene_path))
                env = SNCOAT_Env_v2(
                    name='SNCOAT_Env', scene_path=scene_path, log_dir=self.log_dir,
                    dis_penalty_f=dis_penalty_f, outview_penalty=outview_penalty,
                    action_type=self.action_type, state_type=self.state_type, image_type=self.image_type,
                    actuator_type=self.actuator_type, headless=self.headless, clear_record=False,
                    wait_done_steps=10, history_len=self.history_len,
                    actuator_noise=False, time_delay=False, image_blur=False)

        evaluator.train_report([self.name], actuator_type=self.actuator_type, plot=True)

        return

    def memory_initialization(self, init_buffer_num: int, scene_paths: list,
                              multi_scene: bool, dis_penalty_f=0.1, outview_penalty=5,):
        if multi_scene:
            scene_path = scene_paths[np.random.randint(0, len(scene_paths))]
        else:
            scene_path = scene_paths[0]
        env = SNCOAT_Env_v2(
            name='SNCOAT_Env', scene_path=scene_path, log_dir=self.log_dir,
            dis_penalty_f=dis_penalty_f, outview_penalty=outview_penalty,
            action_type=self.action_type, state_type=self.state_type, image_type=self.image_type,
            actuator_type=self.actuator_type, headless=self.headless, history_len=self.history_len,
            wait_done_steps=10, clear_record=True, actuator_noise=False, time_delay=False, image_blur=False)
        state_dict = env.reset()
        print("=> Collecting Experience....")
        while True:
            if self.memory.total_count < init_buffer_num:
                print('\r=> experience {}/{}'.format(self.memory.total_count, init_buffer_num), end="")
            else:
                env.stop()
                env.shutdown()
                break
            action, action_prob = self.choose_action(state_dict)
            next_state_dict, reward, done, _ = env.env_step(action)

            self.store_transition(state_dict, action, reward, next_state_dict, done)
            state_dict = next_state_dict

            if done:
                state_dict = env.reset()
                self.reset()

            if self.memory.total_count % 1000 == 0 and multi_scene:
                env.stop()
                env.shutdown()
                scene_path = scene_paths[np.random.randint(0, len(scene_paths))]
                print('=> reloading scene in {} ...'.format(scene_path))
                env = SNCOAT_Env_v2(
                    name='SNCOAT_Env', scene_path=scene_path, log_dir=self.log_dir,
                    dis_penalty_f=dis_penalty_f, outview_penalty=outview_penalty,
                    action_type=self.action_type, state_type=self.state_type, image_type=self.image_type,
                    actuator_type=self.actuator_type, headless=self.headless, history_len=self.history_len,
                    wait_done_steps=10, clear_record=True, actuator_noise=False, time_delay=False, image_blur=False)
                state_dict = env.reset()
                self.reset()

    def rollout_v1(self, env: SNCOAT_Env_v2, is_record=False):
        # init hidden variable in LSTM modules
        actor_lstm_tuple = (
            Variable(torch.zeros(self.lstm_layers, 1, self.lstm_dim).float().to(self.device)),
            Variable(torch.zeros(self.lstm_layers, 1, self.lstm_dim).float().to(self.device))
        )
        target_actor_lstm_tuple = (
            Variable(torch.zeros(self.lstm_layers, 1, self.lstm_dim).float().to(self.device)),
            Variable(torch.zeros(self.lstm_layers, 1, self.lstm_dim).float().to(self.device))
        )
        # critic_lstm_tuple = (
        #     Variable(torch.zeros(self.lstm_layers, 1, self.lstm_dim).float().to(self.device)),
        #     Variable(torch.zeros(self.lstm_layers, 1, self.lstm_dim).float().to(self.device))
        # )
        # target_critic_lstm_tuple = (
        #     Variable(torch.zeros(self.lstm_layers, 1, self.lstm_dim).float().to(self.device)),
        #     Variable(torch.zeros(self.lstm_layers, 1, self.lstm_dim).float().to(self.device))
        # )

        state_dict = env.reset(is_record)
        ep_reward = 0
        for i in range(self.min_history):
            with torch.no_grad():
                state = self.state_dict_transfer(state_dict)
                action, actor_lstm_tuple = self.actor(
                    state[0].to(self.device), state[1].to(self.device),
                    hidden=actor_lstm_tuple)
                q_value = self.critic(
                    state[0].to(self.device), state[1].to(self.device),
                    action=action
                )
                n_state_dict, reward, done, _ = env.env_step(action.detach().cpu().numpy())
                ep_reward += reward
                n_state = self.state_dict_transfer(n_state_dict)
                n_action, target_actor_lstm_tuple = self.target_actor(
                    n_state[0].to(self.device), n_state[1].to(self.device),
                    hidden=target_actor_lstm_tuple)
                n_q_value = self.target_critic(
                    n_state[0].to(self.device), n_state[1].to(self.device),
                    action=n_action
                )
                state_dict = n_state_dict
        actor_losses = []
        critic_losses = []
        while True:
            state = self.state_dict_transfer(state_dict)
            action, actor_lstm_tuple = self.actor(
                state[0].to(self.device), state[1].to(self.device),
                hidden=(actor_lstm_tuple[0].detach(), actor_lstm_tuple[1].detach()))
            q_value = self.critic(
                state[0].to(self.device), state[1].to(self.device),
                action=action.detach())
            n_state_dict, r, terminal, _ = env.env_step(action.detach().cpu().numpy())
            ep_reward += r
            n_state = self.state_dict_transfer(n_state_dict)
            n_action, target_actor_lstm_tuple = self.target_actor(
                n_state[0].to(self.device), n_state[1].to(self.device),
                hidden=(target_actor_lstm_tuple[0].detach(), target_actor_lstm_tuple[1].detach()))
            n_q_value = self.target_critic(
                n_state[0].to(self.device), n_state[1].to(self.device),
                action=n_action)
            r = torch.tensor([r], dtype=torch.float).view(-1, 1).to(self.device)
            done = torch.tensor([terminal], dtype=torch.float).view(-1, 1).to(self.device)
            q_target = r + self.gamma * n_q_value * (1 - done)
            critic_loss = F.mse_loss(q_target.detach(), q_value)
            actor_loss = - torch.mean(self.critic(
                state[0].to(self.device), state[1].to(self.device),
                action=action))
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            actor_loss.backward()
            self.critic_optimizer.step()
            self.actor_optimizer.step()
            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())
            state_dict = n_state_dict
            if terminal or env.step_count > self.max_ep_len:
                break
        env.save_records()
        self.soft_update(target_network=self.target_critic, source_network=self.critic)
        self.soft_update(target_network=self.target_actor, source_network=self.actor)
        return ep_reward, np.mean(actor_losses), np.mean(critic_losses)

    def rollout_v2(self, env: SNCOAT_Env_v2, is_record=False):
        state_dict = env.reset(is_record)
        ep_reward = 0
        while True:
            action, action_prob = self.choose_action(state_dict)
            n_state_dict, r, terminal, _ = env.env_step(action)
            self.store_transition(state_dict, action, r, n_state_dict, terminal)
            state_dict = n_state_dict
            ep_reward += r
            if terminal:
                break
        actor_loss, critic_loss = self.update()
        self.reset()
        return ep_reward, actor_loss, critic_loss

    def update(self):
        batch_images, batch_vectors, batch_actions, batch_rewards, \
            batch_n_images, batch_n_vectors, batch_terminals = self.memory.sample(self.batch_size)  # BxMxCxHxW

        batch_images = torch.FloatTensor(np.transpose(batch_images, (1, 0, 2, 3, 4))).to(self.device)  # MxBxCxHxW
        batch_vectors = torch.FloatTensor(np.transpose(batch_vectors, (1, 0, 2, 3))).to(self.device)  # MxBx1xN
        batch_n_images = torch.FloatTensor(np.transpose(
            batch_n_images, (1, 0, 2, 3, 4))).to(self.device)  # MxBxCxHxW
        batch_n_vectors = torch.FloatTensor(np.transpose(batch_n_vectors, (1, 0, 2, 3))).to(
            self.device)  # MxBx1xN

        batch_actions = torch.FloatTensor(np.transpose(batch_actions, (1, 0, 2, 3))).to(self.device).view(-1, self.batch_size, self.action_dim)  # MxB
        batch_rewards = torch.FloatTensor(np.transpose(batch_rewards, (1, 0))).to(self.device)  # MxB
        batch_terminals = torch.FloatTensor(np.transpose(batch_terminals, (1, 0))).to(self.device)  # MxB

        actor_lstm_tuple = (
            Variable(torch.zeros(self.lstm_layers, self.batch_size, self.lstm_dim).float().to(self.device)),
            Variable(torch.zeros(self.lstm_layers, self.batch_size, self.lstm_dim).float().to(self.device))
        )
        target_actor_lstm_tuple = (
            Variable(torch.zeros(self.lstm_layers, self.batch_size, self.lstm_dim).float().to(self.device)),
            Variable(torch.zeros(self.lstm_layers, self.batch_size, self.lstm_dim).float().to(self.device))
        )

        for i in range(self.min_history):
            with torch.no_grad():
                _, actor_lstm_tuple = self.actor(
                    batch_images[i], batch_vectors[i], actor_lstm_tuple)
                _, target_actor_lstm_tuple = self.target_actor(
                    batch_n_images[0], batch_n_vectors[1], hidden=target_actor_lstm_tuple)

        critic_losses = []
        actor_losses = []
        for i in range(self.min_history, self.min_history + self.state_to_update):
            self.global_step += 1
            batch_image = batch_images[i].detach()
            batch_vector = batch_vectors[i].detach()
            batch_n_image = batch_n_images[i].detach()
            batch_n_vector = batch_n_vectors[i].detach()
            batch_action = batch_actions[i].detach()
            batch_r = batch_rewards[i].view(-1, 1).detach()
            batch_done = batch_terminals[i].view(-1, 1).detach()

            action, actor_lstm_tuple = self.actor(
                batch_image, batch_vector,
                hidden=(actor_lstm_tuple[0].detach(), actor_lstm_tuple[1].detach()))
            q_value = self.critic(
                batch_image, batch_vector,
                action=batch_action
            )

            n_action, target_actor_lstm_tuple = self.target_actor(
                batch_n_image, batch_n_vector,
                hidden=(target_actor_lstm_tuple[0].detach(), target_actor_lstm_tuple[1].detach()))
            n_q_value = self.target_critic(
                batch_n_image, batch_n_vector,
                action=n_action,
            )

            q_target = batch_r + self.gamma * n_q_value * (1 - batch_done)
            critic_loss = F.mse_loss(q_target.detach(), q_value)

            actor_loss = - torch.mean(self.critic(
                batch_image, batch_vector,
                action=action,
            ))
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            actor_loss.backward()
            self.critic_optimizer.step()
            self.actor_optimizer.step()

            critic_losses.append(critic_loss.item())
            actor_losses.append(actor_loss.item())

        self.soft_update(target_network=self.target_critic, source_network=self.critic)
        self.soft_update(target_network=self.target_actor, source_network=self.actor)

        return np.mean(actor_losses), np.mean(critic_losses)

    def seed(self, s):
        np.random.seed(s)
        torch.manual_seed(s)
        if self.device.type == 'gpu':
            torch.cuda.manual_seed(s)

    def mode_transfer(self, is_training: bool):
        self.is_train = is_training
        if not self.is_train:
            self.actor.eval()
            self.target_actor.eval()
            self.critic.eval()
            self.target_critic.eval()
        else:
            self.actor.train()
            self.target_actor.train()
            self.critic.train()
            self.target_critic.train()

    def save_model(self, model_path=''):
        print('=> saving network to {} ...'.format(model_path))
        checkpoint = {'episodes': self.episode_counter,
                      'global_step': self.global_step,
                      'actor': self.actor.state_dict(),
                      'target_actor': self.target_actor.state_dict(),
                      'critic': self.critic.state_dict(),
                      'target_critic': self.target_critic.state_dict(),
                      'actor_lr_scheduler': self.actor_lr_scheduler.state_dict(),
                      'critic_lr_scheduler': self.critic_lr_scheduler.state_dict()
                      }
        torch.save(checkpoint, model_path, _use_new_zipfile_serialization=False)
        print('=> model params is saved!')

    def reload_model(self, model_path='', is_train=False):
        if os.path.exists(model_path):
            print('=> restoring model from {} ...'.format(model_path))
            checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage.cpu())
            self.actor.load_state_dict(checkpoint['actor'])
            self.target_actor.load_state_dict(checkpoint['target_actor'])
            self.actor_lr_scheduler.load_state_dict(checkpoint['actor_lr_scheduler'])
            self.critic.load_state_dict(checkpoint['critic'])
            self.target_critic.load_state_dict(checkpoint['target_critic'])
            self.critic_lr_scheduler.load_state_dict(checkpoint['critic_lr_scheduler'])
            self.episode_counter = checkpoint['episodes']
            self.global_step = checkpoint['global_step'] + 1
        else:
            raise ValueError('No model file is found in {}'.format(model_path))
        self.mode_transfer(is_train)

    def choose_action(self, state_dict):
        state = self.state_dict_transfer(state_dict)
        with torch.no_grad():
            action, self.eval_lstm_tuple = self.actor(
                state[0].to(self.device), state[1].to(self.device), self.eval_lstm_tuple)
            action = action.detach().cpu().numpy()
        if self.is_train:
            epsilon = self.start_epsilon * np.exp(- 0.1 * self.episode_counter)
            epsilon = max(self.end_epsilon, epsilon)
            action += np.random.normal(0, epsilon, action.shape)
        return action, 1

    def reset(self, is_record='', record_path=''):
        self.eval_lstm_tuple = (Variable(torch.zeros(self.lstm_layers, 1, self.lstm_dim).float().to(self.device)),
                                Variable(torch.zeros(self.lstm_layers, 1, self.lstm_dim).float().to(self.device)))

        return

    def store_transition(self, state_dict, action, reward, next_state_dict, terminal):
        image = state_dict['image']
        image = self.image_normalize(image)
        if self.state_type == 'OrientImage':
            vector = state_dict['orientation']
        elif self.state_type == 'PosImage' or self.state_type == 'Position':
            vector = state_dict['position']
        else:
            vector = state_dict['pose']

        transition = Transition(image, vector, action, reward, terminal)
        self.memory.push(transition)

    def image_normalize(self, image):
        """
        Normalize depth channel of state
        :param state_dict: the state dict including Image(NxMxWxH), Position(NxMx3), Orientation(NxMx3), and Pose(NxMx6)
        :param env: the simulation env
        :return: norm_image
        """
        if self.image_type == 'Color':
            norm_image = image
        elif self.image_type == 'Depth':
            norm_image = image / self.cam_far_distance
        elif self.image_type == 'RGBD':
            image[:, ..., -1] = image[:, ..., -1] / self.cam_far_distance
            norm_image = image
        else:
            raise ValueError('Unsupported image type!')
        norm_image = np.concatenate(np.transpose(norm_image, (0, 3, 1, 2)))
        return norm_image

    def state_dict_transfer(self, state_dict: dict, aug_funcs=None) -> Tuple[
        torch.Tensor, torch.Tensor]:
        image = state_dict['image']
        if self.image_type == 'Color':
            norm_image = image
        elif self.image_type == 'Depth':
            norm_image = image / self.cam_far_distance
        elif self.image_type == 'RGBD':
            image[:, ..., -1] = image[:, ..., -1] / self.cam_far_distance
            norm_image = image
        else:
            raise ValueError('Unsupported image type!')
        image = np.concatenate(np.transpose(norm_image, (0, 3, 1, 2)))

        if self.state_type == 'Image':
            vector = np.zeros(3)
        elif self.state_type == 'PosImage':
            vector = state_dict['position']
        elif self.state_type == 'OrientImage':
            vector = state_dict['orientation']
        elif self.state_type == 'PoseImage':
            vector = state_dict['pose']
        else:
            vector = state_dict['pose']

        if aug_funcs is not None:
            for i, aug_name in enumerate(aug_funcs):
                if aug_name == 'SensorNoise':
                    vector = aug_funcs[aug_name](vector, sigma=0.2)
                else:
                    image = aug_funcs[aug_name](image)

        state = (
            torch.unsqueeze(torch.from_numpy(image), 0),
            torch.unsqueeze(torch.from_numpy(vector.astype(np.float32)), 0)
        )
        return state

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

    @staticmethod
    def hard_update(target_network: nn.Module, source_network: nn.Module):
        for target_param, source_param in zip(target_network.parameters(), source_network.parameters()):
            target_param.data.copy_(source_param.data)

    def soft_update(self, target_network: nn.Module, source_network: nn.Module):
        for target_param, source_param in zip(target_network.parameters(), source_network.parameters()):
            target_param.data.copy_(target_param.data * (1 - self.tau) + self.tau * source_param.data)
