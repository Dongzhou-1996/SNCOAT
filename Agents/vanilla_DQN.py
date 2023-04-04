import os
import numpy as np
import glob
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from Models.embedding_models import BasicBlock
from Models.critic_models import ConvQNet, ResQNet, FCNet, MultiModalQNet
from Utils.replay_memory import Memory, Transition
import Utils.data_augs as rad
# from Utils.structured_memory import Memory, Transition
from Envs.SNCOAT_Env_v2 import DISCRETE_ACTIONS, SNCOAT_Env_v2
from ptflops import get_model_complexity_info
from Config.env_setting import IMAGE_CHANNELS
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


class VanillaDQN(object):
    """DQN algorithm"""

    def __init__(
            self, name='vanilla_dqn', log_dir='log',
            state_type='Image', image_type='Color',
            action_type='Discrete', actuator_type='Position',
            backbone='ConvNet', model_path=None,
            attention_type='No', data_augs=None,
            gpu_idx=0, action_dim=11, cam_far_distance=20,
            replay_buffer_size=50000, init_buffer_size=10000,
            episode_nums=500, max_ep_len=1000,
            batch_size=32, lr=1e-4, gamma=0.99,
            start_epsilon=0.9, end_epsilon=0.1,
            update_interval=1000, history_len=3,
            with_SE=False, restore=True, is_train=True, vis=False,
            headless=True, seed=1,
    ):
        super(VanillaDQN, self).__init__()
        self.name = name
        assert state_type in ['Image', 'Pose', 'Position', 'PoseImage', 'PosImage'], print('Unsupported state type!')
        self.state_type = state_type
        assert image_type in ['Color', 'Depth', 'RGBD'], print('Unsupported image type!')
        self.image_type = image_type
        assert action_type in ['Discrete', 'Continuous'], print('Unsupported action type!')
        self.action_type = action_type
        assert actuator_type in ['Position', 'Velocity', 'Force'], print('Unsupported actuator type')
        self.actuator_type = actuator_type
        assert backbone in ['ConvNet', 'ResNet18', 'ResNet34'], print('Unsupported backbone type!')
        self.backbone = backbone
        self.action_dim = action_dim

        self.device = torch.device('cuda', gpu_idx) if gpu_idx < torch.cuda.device_count() \
            else torch.device('cpu', gpu_idx)
        self.seed(seed)
        self.headless = headless
        self.vis = vis
        self.lr = lr
        self.gamma = gamma
        self.batch_size = batch_size
        self.start_epsilon = start_epsilon
        self.end_epsilon = end_epsilon
        self.update_interval = update_interval
        self.init_buffer_size = init_buffer_size
        self.replay_buffer_size = replay_buffer_size
        self.episode_nums = episode_nums
        self.max_ep_len = max_ep_len
        self.history_len = history_len
        self.log_dir = log_dir

        self.episode_counter = 0
        self.global_step = 0
        self.memory_counter = 0
        self.cam_far_distance = cam_far_distance
        self.input_channels = IMAGE_CHANNELS[self.image_type] * self.history_len

        if self.state_type == 'Position':
            self.eval_net = FCNet(vec_dim=3, output_dim=self.action_dim, history_len=self.history_len).to(self.device)
            self.target_net = FCNet(vec_dim=3, output_dim=self.action_dim, history_len=self.history_len).to(self.device)
        elif self.state_type == 'Pose':
            self.eval_net = FCNet(vec_dim=6, output_dim=self.action_dim, history_len=self.history_len).to(self.device)
            self.target_net = FCNet(vec_dim=6, output_dim=self.action_dim, history_len=self.history_len).to(self.device)
        elif self.state_type == 'Image':
            if self.backbone == 'ConvNet':
                self.eval_net = ConvQNet(
                    image_channels=self.input_channels, output_dim=self.action_dim,
                    with_SE=with_SE, vis=self.vis, vis_dir=self.log_dir).to(self.device)
                self.target_net = ConvQNet(
                    image_channels=self.input_channels, output_dim=self.action_dim,
                    with_SE=with_SE, vis=self.vis, vis_dir=self.log_dir).to(self.device)
            elif self.backbone == 'ResNet18':
                self.eval_net = ResQNet(
                    image_channels=self.input_channels, output_dim=self.action_dim,
                    block=BasicBlock, num_blocks=[2, 2, 2, 2]).to(self.device)
                self.target_net = ResQNet(
                    image_channels=self.input_channels, output_dim=self.action_dim,
                    block=BasicBlock, num_blocks=[2, 2, 2, 2]).to(self.device)
            elif self.backbone == 'ResNet34':
                self.eval_net = ResQNet(
                    image_channels=self.input_channels, output_dim=self.action_dim,
                    block=BasicBlock, num_blocks=[3, 4, 6, 3]).to(self.device)
                self.target_net = ResQNet(
                    image_channels=self.input_channels, output_dim=self.action_dim,
                    block=BasicBlock, num_blocks=[3, 4, 6, 3]).to(self.device)
            else:
                raise ValueError('Unsupported backbone type!')
        elif self.state_type == 'PosImage':
            self.eval_net = MultiModalQNet(
                image_channels=self.input_channels, vec_dim=3, action_dim=self.action_dim,
                hidden_dim=128, vis=self.vis, with_SE=with_SE, history_len=self.history_len
            )
            self.target_net = MultiModalQNet(
                image_channels=self.input_channels, vec_dim=3, action_dim=self.action_dim,
                hidden_dim=128, vis=self.vis, with_SE=with_SE, history_len=self.history_len
            )
        else:
            self.eval_net = MultiModalQNet(
                image_channels=self.input_channels, vec_dim=6, action_dim=self.action_dim,
                hidden_dim=128, vis=self.vis, with_SE=with_SE, history_len=self.history_len
            )
            self.target_net = MultiModalQNet(
                image_channels=self.input_channels, vec_dim=6, action_dim=self.action_dim,
                hidden_dim=128, vis=self.vis, with_SE=with_SE, history_len=self.history_len
            )

        # self.model_params(self.eval_net, model_name=self.eval_net.name, input_shape=(self.input_channels, 255, 255))

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

        # When we store the memory, we put the state, action, reward and next_state in the memory
        # here reward and action is a number, state is a ndarray
        self.memory = Memory(self.replay_buffer_size)

        self.is_train = is_train
        if not self.is_train:
            self.eval_net.eval()
            self.target_net.eval()
        else:
            self.summary_writer = SummaryWriter(log_dir=self.log_dir)

    def mode_transfer(self, is_train: bool):
        self.is_train = is_train
        self.eval_net.to(self.device)
        self.target_net.to(self.device)
        if not self.is_train:
            self.eval_net.eval()
            self.target_net.eval()
        else:
            self.eval_net.train()
            self.target_net.train()

    def choose_action(self, state_dict: dict):
        state = self.state_dict_transfer(state_dict)

        if self.is_train:
            epsilon = self.start_epsilon * np.exp(- 0.1 * self.episode_counter)
            epsilon = max(self.end_epsilon, epsilon)
            self.summary_writer.add_scalar('epsilon_factor', epsilon, self.episode_counter)

            if np.random.rand() >= epsilon:
                # greedy policy
                action_value = self.eval_net.forward(state[0].to(self.device), state[1].to(self.device))
                action_idx = torch.argmax(action_value, 1)[0].item()
            else:
                # random policy
                action_idx = np.random.randint(0, len(DISCRETE_ACTIONS))
        else:
            # greedy policy
            action_value = self.eval_net.forward(state[0].to(self.device), state[1].to(self.device))
            action_idx = torch.argmax(action_value, 1)[0].item()

        return action_idx, 1

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
                    history_len=self.history_len, wait_done_steps=10,
                    dis_penalty_f=dis_penalty_f, outview_penalty=outview_penalty,
                    headless=self.headless, clear_record=True,
                    actuator_noise=False, time_delay=False, image_blur=False)
                state_dict = env.reset()
                self.reset()

    def learn(self, evaluator: AVTEval, scene_paths: list,
              init_buffer_size=None, episode_nums=None,
              dis_penalty_f=0.1, outview_penalty=5,
              eval_interval=10, record_interval=10, multi_scene=True):
        if init_buffer_size is not None:
            self.init_buffer_size = init_buffer_size
        if episode_nums is not None:
            self.episode_nums = episode_nums
        self.memory_initialization(self.init_buffer_size, scene_paths, multi_scene,
                                   dis_penalty_f, outview_penalty)

        if multi_scene:
            scene_path = scene_paths[np.random.randint(0, len(scene_paths))]
        else:
            scene_path = scene_paths[0]
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
            if (i+1) % self.update_interval == 0:
                print('=> copying eval net params to target net ...')
                self.target_net.load_state_dict(self.eval_net.state_dict())

            if (i + 1) % eval_interval == 0:
                env.stop()
                env.shutdown()

                if multi_scene:
                    scene_path = scene_paths[np.random.randint(0, len(scene_paths))]
                else:
                    scene_path = scene_paths[0]

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
                act_idx, action = self.choose_action(state_dict)
                next_state_dict, reward, done, _ = env.env_step(action)
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

    def update(self):
        # update the parameters
        self.global_step += 1

        batch_images, batch_vectors, batch_action_idxs, batch_rewards,\
            batch_next_images, batch_next_vectors, batch_dones = self.memory.sample(
                self.batch_size)

        batch_images = torch.FloatTensor(batch_images).to(self.device)
        batch_vectors = torch.FloatTensor(batch_vectors).to(self.device)
        batch_action_idxs = torch.LongTensor(batch_action_idxs).unsqueeze(1).to(self.device)
        batch_rewards = torch.FloatTensor(batch_rewards).to(self.device)
        batch_next_images = torch.FloatTensor(batch_next_images).to(self.device)
        batch_next_vectors = torch.FloatTensor(batch_next_vectors).to(self.device)

        # q_eval
        q_eval = self.eval_net(batch_images, batch_vectors).gather(1, batch_action_idxs).squeeze(1)
        q_next = torch.max(self.target_net(batch_next_images, batch_next_vectors).detach(), 1)[0]

        q_target = batch_rewards + self.gamma * q_next
        loss = self.loss_func(q_eval, q_target.to(self.device))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.global_step % 1e4 == 0:
            self.lr_scheduler.step()
        lr = self.lr_scheduler.get_last_lr()[0]
        self.summary_writer.add_scalar('loss/lr', lr, self.global_step)
        self.summary_writer.add_scalar('loss/value_loss', loss, self.global_step)
        self.summary_writer.add_scalar('Q/action_value', q_eval.sum() / self.batch_size, self.global_step)

        return loss.item()

    def save_model(self, model_path=""):
        print('=> saving network to {} ...'.format(model_path))
        checkpoint = {'episodes': self.episode_counter,
                      'steps': self.global_step,
                      'eval_net': self.eval_net.state_dict(),
                      'target_net': self.target_net.state_dict(),
                      'lr_scheduler': self.lr_scheduler.state_dict()}
        torch.save(checkpoint, model_path, _use_new_zipfile_serialization=False)
        print('=> model params is saved!')
        return

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

    def reset(self, is_record='', record_path=''):
        return

    def store_transition(self, state_dict, action, reward, next_state_dict, terminal):
        image = state_dict['image']
        image = self.image_normalize(image)
        next_image = next_state_dict['image']
        next_image = self.image_normalize(next_image)
        if self.state_type == 'OrientImage':
            vector = state_dict['orientation']
            next_vector = next_state_dict['orientation']
        elif self.state_type == 'PosImage' or self.state_type == 'Position':
            vector = state_dict['position']
            next_vector = next_state_dict['position']
        else:
            vector = state_dict['pose']
            next_vector = next_state_dict['pose']

        transition = Transition(image, vector, action, reward, next_image, next_vector, terminal)
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

    def state_dict_transfer(self, state_dict: dict, aug_funcs=None):
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

    def seed(self, s):
        np.random.seed(s)
        torch.manual_seed(s)
        if self.device.type == 'gpu':
            torch.cuda.manual_seed(s)

    @staticmethod
    def action_pos_2_force(pos_action, env):
        return 2 * env.chaser_mass * pos_action / (env.delta_time ** 2)

    @staticmethod
    def action_vel_2_force(vel_action, env):
        return vel_action / env.delta_time

    @staticmethod
    def model_params(eval_net, model_name='ConvQNet', input_shape=(3, 255, 255)):
        flops, params = get_model_complexity_info(eval_net, input_shape,
                                                  as_strings=True, print_per_layer_stat=True)
        print('Model name: ' + model_name)
        print('Flops:  ' + flops)
        print('Params: ' + params)
        print('=========================================================')
