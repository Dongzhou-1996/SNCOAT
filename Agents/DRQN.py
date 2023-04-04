import os

import gym
import numpy as np
import glob
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import Utils.data_augs as rad
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from Models.critic_models import Conv_LSTM, Conv_LSTM_A, Conv_LSTM_MHA, FC_LSTM, Multimodal_LSTM
# from Utils.replay_memory import Memory, Transition
from Utils.structured_memory import Memory, Transition
from Config.env_setting import IMAGE_CHANNELS, DISCRETE_ACTIONS
from ptflops import get_model_complexity_info
from Envs.SNCOAT_Env_v2 import SNCOAT_Env_v2
from evaluation import AVTEval

aug_to_func = {
    'Crop': rad.random_crop,
    'Cutout': rad.random_cutout,
    'CutoutColor': rad.random_cutout_color,
    'Flip': rad.random_flip,
    'Rotate': rad.random_rotation,
    'SensorNoise': rad.random_sensor_noise,
    'No': rad.no_aug,
}


class DRQN(object):
    """
        DRQN Algorithm
    """

    def __init__(
            self, name='drqn', log_dir='log',
            state_type='Image', image_type='Color',
            action_type='Discrete', actuator_type='Position',
            attention_type='No', data_augs=None,
            model_path=None, backbone_path=None,
            action_dim=11, gpu_idx=0, seed=1,
            replay_buffer_size=50000,
            init_buffer_size=10000,
            batch_size=64, lr=1e-4, gamma=0.99,
            max_ep_len=20, episode_nums=500,
            start_epsilon=0.9, end_epsilon=0.1,
            update_interval=1000, cam_far_distance=20,
            min_history=4, state_to_update=4,
            lstm_dim=1024, lstm_layers=2, head_num=8,
            restore=False, restore_backbone=False,
            headless=True, is_train=True, with_SE=True, vis=False):
        self.name = name
        assert state_type in ['Image', 'PosImage', 'OrientImage', 'PoseImage', 'Pose', 'Position'], print(
            'Unsupported state type!')
        self.state_type = state_type
        assert image_type in ['Color', 'Depth', 'RGBD'], print('Unsupported image type!')
        self.image_type = image_type
        assert action_type in ['Discrete', 'Continuous'], print('Unsupported action type!')
        self.action_type = action_type
        assert actuator_type in ['Position', 'Velocity', 'Force'], print('Unsupported actuator type!')
        self.actuator_type = actuator_type
        assert attention_type in ['No', 'Add', 'DotProd', 'MHA'], print('Unsupported attention mechanism!')
        self.attention_type = attention_type

        self.action_dim = action_dim
        self.device = torch.device('cuda', gpu_idx) if gpu_idx < torch.cuda.device_count() \
            else torch.device('cpu', gpu_idx)
        self.seed(seed)

        self.data_augs = {}
        for aug_name in data_augs:
            assert aug_name in aug_to_func, 'invalid data augmentation method'
            self.data_augs[aug_name] = aug_to_func[aug_name]

        self.headless = headless
        self.vis = vis
        self.lr = lr
        self.gamma = gamma
        self.batch_size = batch_size
        self.start_epsilon = start_epsilon
        self.end_epsilon = end_epsilon
        self.update_interval = update_interval
        self.replay_buffer_size = replay_buffer_size
        self.init_buffer_size = init_buffer_size
        self.max_ep_len = max_ep_len
        self.episode_nums = episode_nums
        self.history_len = 1
        self.min_history = min_history
        self.state_to_update = state_to_update
        self.log_dir = log_dir
        self.lstm_dim = lstm_dim
        self.lstm_layers = lstm_layers
        self.head_num = head_num
        self.with_SE = with_SE

        self.episode_counter = 0
        self.global_step = 0

        self.memory = Memory(max_episode_records=1000, max_replay_experiences=self.replay_buffer_size,
                             aug_funcs=self.data_augs, episode_len=self.max_ep_len,
                             history_len=self.min_history + self.state_to_update)
        self.cam_far_distance = cam_far_distance
        self.image_channels = IMAGE_CHANNELS[self.image_type]

        if self.state_type == 'Image':
            if self.attention_type == 'No':
                self.eval_net = Conv_LSTM(
                    image_channels=self.image_channels, action_dim=self.action_dim,
                    vis=self.vis, vis_dir=self.log_dir, with_SE=self.with_SE,
                    lstm_dim=self.lstm_dim, lstm_layers=self.lstm_layers
                )
                self.target_net = Conv_LSTM(
                    image_channels=self.image_channels, action_dim=self.action_dim,
                    vis=False, vis_dir=self.log_dir, with_SE=self.with_SE,
                    lstm_dim=self.lstm_dim, lstm_layers=self.lstm_layers
                )
            elif self.attention_type == 'MHA':
                self.eval_net = Conv_LSTM_MHA(
                    image_channels=self.image_channels, action_dim=self.action_dim,
                    vis=self.vis, vis_dir=self.log_dir, with_SE=self.with_SE, head_num=self.head_num,
                    lstm_dim=self.lstm_dim, lstm_layers=self.lstm_layers
                )
                self.target_net = Conv_LSTM_MHA(
                    image_channels=self.image_channels, action_dim=self.action_dim,
                    vis=False, vis_dir=self.log_dir, with_SE=self.with_SE, head_num=self.head_num,
                    lstm_dim=self.lstm_dim, lstm_layers=self.lstm_layers
                )
            else:
                self.eval_net = Conv_LSTM_A(
                    image_channels=self.image_channels, action_dim=self.action_dim,
                    attention_type=self.attention_type, vis=self.vis, vis_dir=self.log_dir,
                    lstm_dim=self.lstm_dim, lstm_layers=self.lstm_layers, with_SE=self.with_SE
                )
                self.target_net = Conv_LSTM_A(
                    image_channels=self.image_channels, action_dim=self.action_dim,
                    attention_type=self.attention_type, vis=False, vis_dir=self.log_dir,
                    lstm_dim=self.lstm_dim, lstm_layers=self.lstm_layers, with_SE=self.with_SE
                )
        elif self.state_type == 'Position':
            self.eval_net = FC_LSTM(vec_dim=3, action_dim=self.action_dim,
                                    lstm_dim=self.lstm_dim, lstm_layers=self.lstm_layers)
            self.target_net = FC_LSTM(vec_dim=3, action_dim=self.action_dim,
                                      lstm_dim=self.lstm_dim, lstm_layers=self.lstm_layers)
        elif self.state_type == 'Pose':
            self.eval_net = FC_LSTM(vec_dim=6, action_dim=self.action_dim,
                                    lstm_dim=self.lstm_dim, lstm_layers=self.lstm_layers)
            self.target_net = FC_LSTM(vec_dim=6, action_dim=self.action_dim,
                                      lstm_dim=self.lstm_dim, lstm_layers=self.lstm_layers)
        elif self.state_type == 'PoseImage':
            self.eval_net = Multimodal_LSTM(
                image_channels=self.image_channels, vec_dim=6, action_dim=self.action_dim,
                lstm_dim=self.lstm_dim, lstm_layers=self.lstm_layers, with_SE=self.with_SE
            )
            self.target_net = Multimodal_LSTM(
                image_channels=self.image_channels, vec_dim=6, action_dim=self.action_dim,
                lstm_dim=self.lstm_dim, lstm_layers=self.lstm_layers, with_SE=self.with_SE
            )
        else:
            self.eval_net = Multimodal_LSTM(
                image_channels=self.image_channels, vec_dim=3, action_dim=self.action_dim,
                lstm_dim=self.lstm_dim, lstm_layers=self.lstm_layers, with_SE=self.with_SE
            )
            self.target_net = Multimodal_LSTM(
                image_channels=self.image_channels, vec_dim=3, action_dim=self.action_dim,
                lstm_dim=self.lstm_dim, lstm_layers=self.lstm_layers, with_SE=self.with_SE
            )

        self.eval_lstm_tuple = (Variable(torch.zeros(self.lstm_layers, 1, self.lstm_dim).float().to(self.device)),
                                Variable(torch.zeros(self.lstm_layers, 1, self.lstm_dim).float().to(self.device)))

        # self.model_params(self.eval_net, model_name=self.eval_net.name, input_shape=(self.image_channels, 255, 255))

        self.optimizer = optim.Adam(self.eval_net.parameters(), lr=self.lr)
        self.lr_scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9)
        self.loss_func = nn.MSELoss()

        if restore:
            if model_path is None:
                model_path = self.get_latest_model_path()
            if os.path.exists(model_path):
                self.reload_model(model_path, is_train)
            else:
                raise ValueError('No model file is found in {}'.format(model_path))

        if restore_backbone:
            if backbone_path is None:
                raise ValueError('=> The path of backbone is not specified!')
            elif os.path.exists(backbone_path):
                print('=> restoring backbone model from  {} ...'.format(backbone_path))
                checkpoint = torch.load(backbone_path, map_location=lambda storage, loc: storage.cpu())
                pretrained_dict = checkpoint['state_dict']
                pretrained_backbone_dict = {'.'.join(k.split('.')[1:]): v for k, v in pretrained_dict.items() if
                                            'backbone' in k.split('.')}
                backbone_dict = self.eval_net.embedding.state_dict()
                pretrained_backbone_dict = {k: v for k, v in pretrained_backbone_dict.items() if k in backbone_dict}
                backbone_dict.update(pretrained_backbone_dict)
                self.eval_net.embedding.load_state_dict(backbone_dict)
                self.target_net.embedding.load_state_dict(backbone_dict)
                # frozen backbone params
                self.eval_net.embedding.requires_grad_(False)
                self.target_net.embedding.requires_grad_(False)

                if self.attention_type != 'No':
                    pretrained_attention_dict = {'.'.join(k.split('.')[1:]): v for k, v in pretrained_dict.items() if
                                                 'attention' in k.split('.')}
                    attention_dict = self.eval_net.attention.state_dict()
                    pretrained_attention_dict = {k: v for k, v in pretrained_attention_dict.items() if
                                                 k in attention_dict}
                    attention_dict.update(pretrained_attention_dict)
                    self.eval_net.attention.load_state_dict(attention_dict)
                    self.target_net.attention.load_state_dict(attention_dict)
                    # frozen attention params
                    self.eval_net.attention.requires_grad_(False)
                    self.target_net.attention.requires_grad_(False)

        self.eval_net.to(self.device)
        self.target_net.to(self.device)

        self.is_train = is_train
        if not self.is_train:
            self.eval_net.eval()
            self.target_net.eval()
        else:
            self.eval_net.train()
            self.target_net.train()
            self.summary_writer = SummaryWriter(log_dir=self.log_dir)

    def mode_transfer(self, is_train: bool):
        self.is_train = is_train
        if not self.is_train:
            self.eval_net.eval()
            self.target_net.eval()
        else:
            self.eval_net.train()
            self.target_net.train()

    def save_model(self, model_path):
        print('=> saving network to {} ...'.format(model_path))
        checkpoint = {'episodes': self.episode_counter,
                      'steps': self.global_step,
                      'eval_net': self.eval_net.state_dict(),
                      'target_net': self.target_net.state_dict(),
                      'lr_scheduler': self.lr_scheduler.state_dict()}
        torch.save(checkpoint, model_path, _use_new_zipfile_serialization=False)
        print('=> model params is saved!')

    def reload_model(self, model_path='', is_train=False):
        if os.path.exists(model_path):
            print('=> restoring model from {} ...'.format(model_path))
            checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage.cpu())
            self.eval_net.load_state_dict(checkpoint['eval_net'])
            self.target_net.load_state_dict(checkpoint['target_net'])
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            self.episode_counter = checkpoint['episodes']
            self.global_step = checkpoint['steps'] + 1
        else:
            raise ValueError('No model file is found in {}'.format(model_path))
        self.mode_transfer(is_train)

    def choose_action(self, state_dict):
        state = self.state_dict_transfer(state_dict)

        if self.is_train:
            epsilon = self.start_epsilon * np.exp(- 0.1 * self.episode_counter)
            epsilon = max(self.end_epsilon, epsilon)
            self.summary_writer.add_scalar('epsilon_factor', epsilon, self.episode_counter)

            if np.random.rand() >= epsilon:
                # greedy policy
                with torch.no_grad():
                    action_value, self.eval_lstm_tuple = self.eval_net.forward(
                        state[0].to(self.device), state[1].to(self.device), self.eval_lstm_tuple)
                indx = torch.argmax(action_value, 1)[0].item()
            else:
                # random policy
                indx = np.random.randint(0, len(DISCRETE_ACTIONS))
        else:
            # greedy policy
            with torch.no_grad():
                action_value, self.eval_lstm_tuple = self.eval_net.forward(
                    state[0].to(self.device), state[1].to(self.device), self.eval_lstm_tuple)
            indx = torch.argmax(action_value, 1)[0].item()

        return indx, 1

    def learn(self, evaluator: AVTEval, scene_paths: list,
              init_buffer_size=None, episode_nums=None,
              dis_penalty_f=0.1, outview_penalty=5,
              eval_interval=50, record_interval=10, multi_scene=True):
        if init_buffer_size is not None:
            self.init_buffer_size = init_buffer_size
        if episode_nums is not None:
            self.episode_nums = episode_nums
        self.memory_initialization(self.init_buffer_size, scene_paths, multi_scene)

        scene_path = scene_paths[np.random.randint(0, len(scene_paths))]
        env = SNCOAT_Env_v2(
            name='SNCOAT_Env', scene_path=scene_path, log_dir=self.log_dir,
            state_type=self.state_type, image_type=self.image_type,
            action_type=self.action_type, actuator_type=self.actuator_type,
            history_len=self.history_len, wait_done_steps=10,
            dis_penalty_f=dis_penalty_f, outview_penalty=outview_penalty,
            headless=self.headless, clear_record=True,
            actuator_noise=False, time_delay=False, image_blur=False)

        for i in range(self.episode_counter, self.episode_nums):
            self.episode_counter += 1
            print('=> training on {}th episode ...'.format(self.episode_counter))
            if (i + 1) % self.update_interval == 0:
                print('=> copying eval net params to target net ...')
                self.target_net.load_state_dict(self.eval_net.state_dict())

            if (i + 1) % eval_interval == 0:
                env.stop()
                env.shutdown()

                model_path = os.path.join(self.log_dir, 'model_ep_{:03d}.pth'.format(self.episode_counter + 1))
                self.save_model(model_path)

                self.mode_transfer(is_train=False)
                evaluator.eval(
                    [self], action_type=self.action_type, actuator_type=self.actuator_type,
                    eval_episode=self.episode_counter, max_episode_len=self.max_ep_len,
                    image_blur=False, actuator_noise=False, time_delay=False,
                    headless=True, overwrite=True
                )
                self.mode_transfer(is_train=True)

                if multi_scene:
                    scene_path = scene_paths[np.random.randint(0, len(scene_paths))]
                else:
                    scene_path = scene_paths[0]

                print('=> reloading scene in {} ...'.format(scene_path))
                env = SNCOAT_Env_v2(
                    name='SNCOAT_Env', scene_path=scene_path, log_dir=self.log_dir,
                    state_type=self.state_type, image_type=self.image_type,
                    action_type=self.action_type, actuator_type=self.actuator_type,
                    history_len=self.history_len, wait_done_steps=10,
                    dis_penalty_f=dis_penalty_f, outview_penalty=outview_penalty,
                    headless=self.headless, clear_record=True,
                    actuator_noise=False, time_delay=False, image_blur=False)

            if i % record_interval == 0:
                record = True
            else:
                record = False
            # Reset the environment
            state_dict = env.reset(is_record=record)
            self.reset()

            ep_reward = 0
            while True:
                print('\r=> step: {}'.format(env.step_count), end="")
                act_idx, action_prob = self.choose_action(state_dict)
                next_state_dict, reward, done, _ = env.env_step(act_idx)
                # Save transition to replay memory
                self.store_transition(state_dict, act_idx, reward, next_state_dict, done)
                ep_reward += reward

                loss = self.update()

                if done or env.step_count > self.max_ep_len:
                    print("\nepisode: {}, episode reward: {}, episode length: {}, loss: {}".format(
                        self.episode_counter, round(ep_reward, 3), env.step_count, loss))
                    self.reset()
                    break

                state_dict = next_state_dict
        evaluator.train_report([self.name], actuator_type=self.actuator_type, plot=True)
        return

    def memory_initialization(self, init_buffer_num: int, scene_paths: list,
                              multi_scene: bool, dis_penalty_f=0.1, outview_penalty=5,):
        scene_path = scene_paths[np.random.randint(0, len(scene_paths))]
        env = SNCOAT_Env_v2(
            name='SNCOAT_Env', scene_path=scene_path, log_dir=self.log_dir,
            state_type=self.state_type, image_type=self.image_type,
            action_type=self.action_type, actuator_type=self.actuator_type,
            history_len=self.history_len, wait_done_steps=10,
            dis_penalty_f=dis_penalty_f, outview_penalty=outview_penalty,
            headless=self.headless, clear_record=True,
            actuator_noise=False, time_delay=False, image_blur=False)
        state_dict = env.reset()
        print("=> Collecting Experience....")
        while True:
            if self.memory.total_count < init_buffer_num:
                print('\r=> experience {}/{}'.format(self.memory.total_count, init_buffer_num), end="")
            else:
                env.stop()
                env.shutdown()
                break
            action_idx, action_prob = self.choose_action(state_dict)
            next_state_dict, reward, done, _ = env.env_step(action_idx)

            self.store_transition(state_dict, action_idx, reward, next_state_dict, done)
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
                    state_type=self.state_type, image_type=self.image_type,
                    action_type=self.action_type, actuator_type=self.actuator_type,
                    dis_penalty_f=dis_penalty_f, outview_penalty=outview_penalty,
                    history_len=self.history_len, wait_done_steps=10,
                    headless=self.headless, clear_record=True,
                    actuator_noise=False, time_delay=False, image_blur=False)
                state_dict = env.reset()
                self.reset()

    def update(self):
        batch_images, batch_vectors, batch_actions, batch_rewards, \
            batch_next_images, batch_next_vectors, batch_terminals = self.memory.sample(
                self.batch_size)  # BxMxCxHxW

        batch_images = torch.FloatTensor(np.transpose(batch_images, (1, 0, 2, 3, 4))).to(self.device)  # MxBxCxHxW
        batch_vectors = torch.FloatTensor(np.transpose(batch_vectors, (1, 0, 2, 3))).to(self.device)  # MxBx1xN
        batch_next_images = torch.FloatTensor(np.transpose(
            batch_next_images, (1, 0, 2, 3, 4))).to(self.device)  # MxBxCxHxW
        batch_next_vectors = torch.FloatTensor(np.transpose(batch_next_vectors, (1, 0, 2, 3))).to(
            self.device)  # MxBx1xN

        batch_actions = torch.LongTensor(np.transpose(batch_actions, (1, 0))).to(self.device)  # MxB
        batch_rewards = torch.FloatTensor(np.transpose(batch_rewards, (1, 0)), ).to(self.device)  # MxB
        batch_terminals = torch.FloatTensor(np.transpose(batch_terminals, (1, 0))).to(self.device)  # MxB

        eval_lstm_tuple = (
            Variable(torch.zeros(self.lstm_layers, self.batch_size, self.lstm_dim).float().to(self.device)),
            Variable(torch.zeros(self.lstm_layers, self.batch_size, self.lstm_dim).float().to(self.device))
        )

        target_lstm_tuple = (
            Variable(torch.zeros(self.lstm_layers, self.batch_size, self.lstm_dim).float().to(self.device)),
            Variable(torch.zeros(self.lstm_layers, self.batch_size, self.lstm_dim).float().to(self.device))
        )

        for i in range(self.min_history):
            with torch.no_grad():
                _, eval_lstm_tuple = self.eval_net(batch_images[i], batch_vectors[i], eval_lstm_tuple)
                _, target_lstm_tuple = self.target_net(batch_next_images[i], batch_next_vectors[i], target_lstm_tuple)

        total_loss = []
        total_q_eval = []

        for i in range(self.min_history, self.min_history + self.state_to_update):
            self.global_step += 1
            batch_image = batch_images[i].detach()
            batch_vector = batch_vectors[i].detach()
            batch_next_image = batch_next_images[i].detach()
            batch_next_vector = batch_next_vectors[i].detach()
            q_eval, eval_lstm_tuple = self.eval_net(
                batch_image, batch_vector, (eval_lstm_tuple[0].detach(), eval_lstm_tuple[1].detach()))
            q_eval = q_eval.gather(1, batch_actions[i].reshape(-1, 1))
            q_next, target_lstm_tuple = self.target_net(
                batch_next_image, batch_next_vector, (target_lstm_tuple[0].detach(), target_lstm_tuple[1].detach()))
            q_next = torch.max(q_next, 1)[0]
            q_target = batch_rewards[i] + self.gamma * q_next
            loss = self.loss_func(q_eval, q_target.reshape(-1, 1))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.summary_writer.add_scalar('loss/value_loss', loss, self.global_step)
            self.summary_writer.add_scalar('Q/action_value', q_eval.sum() / self.batch_size, self.global_step)

            total_loss.append(loss)
            total_q_eval.append(q_eval.sum())

        if self.global_step % 1e5 == 0:
            self.lr_scheduler.step()

        total_loss = torch.stack(total_loss).mean()
        # self.optimizer.zero_grad()
        # total_loss.backward()
        # self.optimizer.step()

        # self.summary_writer.add_scalar('loss/value_loss', total_loss, self.global_step)
        # self.summary_writer.add_scalar('Q/action_value', sum(total_q_eval) / self.state_to_update, self.global_step)

        return total_loss.item()

    def seed(self, s):
        np.random.seed(s)
        torch.manual_seed(s)
        if self.device.type == 'gpu':
            torch.cuda.manual_seed(s)

    def reset(self, is_record=False, record_path=''):
        self.eval_lstm_tuple = (Variable(torch.zeros(self.lstm_layers, 1, self.lstm_dim).float().to(self.device)),
                                Variable(torch.zeros(self.lstm_layers, 1, self.lstm_dim).float().to(self.device)))

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
        :param image: Image(NxMxWxH),
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

    def state_dict_transfer(self, state_dict: dict, aug_funcs=None):
        image = state_dict['image']

        image = self.image_normalize(image)

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

        # if aug_funcs is not None:
        #     for i, aug_name in enumerate(aug_funcs):
        #         if aug_name == 'SensorNoise':
        #             vector = aug_funcs[aug_name](vector, sigma=0.2)
        #         else:
        #             image = aug_funcs[aug_name](image)

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

    def get_available_model_path(self):
        model_paths = glob.glob(os.path.join(self.log_dir, 'model_*.pth'))
        if len(model_paths) > 0:
            print('=> found {} models in {}'.format(len(model_paths), self.log_dir))
            return sorted(model_paths)
        else:
            raise ValueError('No checkpoint file has been found!')

    @staticmethod
    def model_params(eval_net, model_name='ConvQNet', input_shape=(3, 255, 255)):
        flops, params = get_model_complexity_info(eval_net, input_shape,
                                                  as_strings=True, print_per_layer_stat=True)
        print('Model name: ' + model_name)
        print('Flops:  ' + flops)
        print('Params: ' + params)
        print('=========================================================')
