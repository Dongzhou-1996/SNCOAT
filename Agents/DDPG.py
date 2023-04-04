import os

import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from collections import namedtuple
from Models.actor_models import ConvActor
from Models.critic_models import ConvQNet
from Utils.replay_memory import Memory, Transition
from Config.env_setting import IMAGE_CHANNELS
from Envs.SNCOAT_Env_v2 import SNCOAT_Env_v2
from evaluation import AVTEval
from typing import Tuple, Union


class RandomProcess(object):
    def reset_states(self):
        pass


class AnnealedGaussianProcess(RandomProcess):
    def __init__(self, mu, sigma, sigma_min, n_steps_annealing):
        self.mu = mu
        self.sigma = sigma
        self.n_steps = 0

        if sigma_min is not None:
            self.m = - float(sigma - sigma_min) / float(n_steps_annealing)
            self.c = sigma
            self.sigma_min = sigma_min
        else:
            self.m = 0
            self.c = sigma
            self.sigma_min = sigma

    @property
    def current_sigma(self):
        sigma = max(self.sigma_min, self.m * float(self.n_steps) + self.c)
        return sigma


class OrnsteinUhlenbeckProcess(AnnealedGaussianProcess):
    def __init__(self, theta, mu=0.0, sigma=1.0, dt=1e-2, x0=None, size=1, sigma_min=None, n_steps_annealing=1000):
        super(OrnsteinUhlenbeckProcess, self).__init__(mu=mu, sigma=sigma, sigma_min=sigma_min,
                                                       n_steps_annealing=n_steps_annealing)
        self.theta = theta
        self.mu = mu
        self.dt = dt
        self.x0 = x0
        self.size = size
        self.x_prev = None
        self.reset_states()

    def sample(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt \
            + self.current_sigma * np.sqrt(self.dt) * np.random.normal(size=self.size)
        self.x_prev = x
        self.n_steps += 1
        return x

    def reset_states(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros(self.size)


class DDPG(object):
    def __init__(
            self, name: str, log_dir='./log',
            state_type='Image', image_type='Color',
            action_type='Continuous', actuator_type='Velocity',
            pretrained_model=None,
            action_dim=3, gpu_idx=0, seed=1,
            replay_buffer_size=50000,
            init_buffer_size=10000,
            episode_nums=500,
            history_len=4, max_ep_len=1000, max_action=2,
            start_epsilon=0.9, end_epsilon=0.1,
            epsilon_decay_steps=100000,
            lr=1e-4, tau=1e-4, gamma=0.99,
            batch_size=32, cam_far_distance=20,
            is_train=True, restore=False,
            with_SE=True, headless=True,
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

        self.lr = lr
        self.tau = tau
        self.discount_factor = gamma
        self.replay_buffer_size = replay_buffer_size
        self.init_buffer_size = init_buffer_size
        self.max_action = max_action
        self.max_ep_len = max_ep_len
        self.episode_nums = episode_nums
        self.history_len = history_len
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
        self.device = torch.device('cuda', gpu_idx) if gpu_idx < torch.cuda.device_count() \
            else torch.device('cpu', gpu_idx)

        self.seed(seed)
        self.cam_far_distance = cam_far_distance
        self.image_channels = IMAGE_CHANNELS[self.image_type] * self.history_len

        ## create memory pool
        self.memory = Memory(self.replay_buffer_size)

        self.actor = ConvActor(
            self.image_channels, self.action_dim, with_SE=with_SE, activation='Tanh'
        ).to(device=self.device)
        self.target_actor = ConvActor(
            self.image_channels, self.action_dim, with_SE=with_SE, activation='Tanh'
        ).to(device=self.device)
        self.critic = ConvQNet(
            self.image_channels, output_dim=1, action_dim=self.action_dim, with_SE=with_SE
        ).to(device=self.device)
        self.target_critic = ConvQNet(
            self.image_channels, output_dim=1, action_dim=self.action_dim, with_SE=with_SE
        ).to(device=self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=50 * self.lr)

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

            checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage.cuda())
            self.actor.load_state_dict(checkpoint['actor'])
            self.target_actor.load_state_dict(checkpoint['target_actor'])
            self.actor_lr_scheduler.load_state_dict(checkpoint['actor_lr_scheduler'])
            self.critic.load_state_dict(checkpoint['critic'])
            self.target_critic.load_state_dict(checkpoint['target_critic'])
            self.critic_lr_scheduler.load_state_dict(checkpoint['critic_lr_scheduler'])
            self.episode_counter = checkpoint['episodes']
            self.global_step = checkpoint['global_step'] + 1

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
              multi_scene=True, eval_interval=50, record_interval=10):
        if init_buffer_size is not None:
            self.init_buffer_size = init_buffer_size
        if episode_nums is not None:
            self.episode_nums = episode_nums
        self.memory_initialization(self.init_buffer_size, scene_paths, multi_scene)

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
            wait_done_steps=10, history_len=self.history_len,
            dis_penalty_f=dis_penalty_f, outview_penalty=outview_penalty,
            actuator_noise=False, time_delay=False, image_blur=False)

        for i in range(self.episode_counter, self.episode_nums):
            self.episode_counter += 1
            print('=> training on {}th episode ...'.format(self.episode_counter))
            if (i + 1) % eval_interval == 0:
                env.stop()
                env.shutdown()
                model_path = os.path.join(self.log_dir, 'model_ep_{:03d}.pth'.format(self.episode_counter + 1))
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
                    action_type=self.action_type, state_type=self.state_type, image_type=self.image_type,
                    actuator_type=self.actuator_type, headless=self.headless, clear_record=False,
                    wait_done_steps=10, history_len=self.history_len,
                    dis_penalty_f=dis_penalty_f, outview_penalty=outview_penalty,
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
                action, action_prob = self.choose_action(state_dict)
                print('\r=> step: {}'.format(env.step_count), end="")
                next_state_dict, reward, done, _ = env.env_step(action)

                # Save transition to replay memory
                self.store_transition(state_dict, action, reward, next_state_dict, done)
                ep_reward += reward

                critic_loss, actor_loss = self.update()

                if done or env.step_count > self.max_ep_len:
                    print(
                        "\n[Train]: episode: {}, episode reward: {}, episode length: {}, critic loss: {:0.3f}, actor loss: {:0.3f}".format(
                            self.episode_counter, round(ep_reward, 3), env.step_count, critic_loss, actor_loss))
                    self.reset()
                    break

                state_dict = next_state_dict

            self.summary_writer.add_scalar('episode_len', env.step_count, global_step=self.episode_counter)
            self.summary_writer.add_scalar('episode_reward', ep_reward, global_step=self.episode_counter)
            env.save_records()
        evaluator.train_report([self.name], actuator_type=self.actuator_type, plot=True)

        return

    def memory_initialization(self, init_buffer_num: int, scene_paths: list,
                              multi_scene: bool, dis_penalty_f=0.1, outview_penalty=5):
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


    def update(self):
        self.global_step += 1
        batch_images, batch_vectors, batch_actions, batch_rewards, \
        batch_next_images, batch_next_vectors, batch_terminals = self.memory.sample(self.batch_size)  # BxMxCxHxW

        batch_images = torch.FloatTensor(batch_images).to(self.device)  # BxCxHxW
        # batch_vectors = torch.FloatTensor(np.transpose(batch_vectors, (1, 0, 2, 3))).to(self.device)  # Bx1xN
        batch_next_images = torch.FloatTensor(batch_next_images).to(self.device)  # MxBxCxHxW
        # batch_next_vectors = torch.FloatTensor(np.transpose(batch_next_vectors, (1, 0, 2, 3))).to(
        #     self.device)  # MxBx1xN

        batch_actions = torch.FloatTensor(batch_actions).to(self.device)  # MxB
        batch_rewards = torch.FloatTensor(batch_rewards).view(-1, 1).to(self.device)  # MxB
        batch_terminals = torch.FloatTensor(batch_terminals).view(-1, 1).to(self.device)  # MxB

        ## update critic
        q_values = self.critic(batch_images, action=batch_actions)
        next_q_values = self.target_critic(batch_next_images, action=self.target_actor(batch_next_images))
        target_q_values = batch_rewards + self.discount_factor * (1 - batch_terminals) * next_q_values

        critic_loss = F.mse_loss(target_q_values.detach(), q_values)
        self.summary_writer.add_scalar('Loss/critic_loss', critic_loss, self.global_step)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()
        critic_lr = self.critic_lr_scheduler.get_last_lr()[0]
        self.summary_writer.add_scalar('LR/critic_lr', critic_lr, self.global_step)

        ## update actor
        actor_loss = - self.critic(batch_images, action=self.actor(batch_images)).mean()
        self.summary_writer.add_scalar('Loss/actor_loss', actor_loss, self.global_step)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()
        actor_lr = self.actor_lr_scheduler.get_last_lr()[0]
        self.summary_writer.add_scalar('LR/actor_lr', actor_lr, self.global_step)

        # if self.global_step % 1e5 == 0:
        #     self.actor_lr_scheduler.step()
        #     self.critic_lr_scheduler.step()

        self.soft_update(target_network=self.target_critic, source_network=self.critic)
        self.soft_update(target_network=self.target_actor, source_network=self.actor)

        return critic_loss.item(), actor_loss.item()

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

    def reload_model(self, model_path='', is_training=False):
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
        self.mode_transfer(is_training)

    def choose_action(self, state_dict):
        state = self.state_dict_transfer(state_dict)
        with torch.no_grad():
            action = self.actor(state[0].to(self.device)).squeeze(0).detach().cpu().numpy()
        if self.is_train:
            epsilon = self.start_epsilon * np.exp(- 0.1 * self.episode_counter)
            epsilon = max(self.end_epsilon, epsilon)
            action += np.random.normal(0, epsilon, action.shape)
        return action, 1

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
