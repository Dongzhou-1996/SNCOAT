import os
import random
import sys

import numpy as np
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import Utils.data_augs as rad
from Config.env_setting import IMAGE_CHANNELS, TRAIN_SCENES_DIR, CONTINUOUS_ACTION_DIM, DISCRETE_ACTION_DIM
from Models.actor_critic_models import ActorCritic
from Utils.replay_memory import Transition
from torch.utils.tensorboard import SummaryWriter
from Envs.SNCOAT_Env_v2 import SNCOAT_Env_v2, DISCRETE_ACTIONS
from evaluation import AVTEval
from typing import Union, Tuple, List

aug_to_func = {
    'Crop': rad.random_crop,
    'Cutout': rad.random_cutout,
    'CutoutColor': rad.random_cutout_color,
    'Flip': rad.random_flip,
    'Rotate': rad.random_rotation,
    'SensorNoise': rad.random_sensor_noise,
    'No': rad.no_aug,
}


class Memory(object):
    """
    Code based on:
    https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
    Expects namedtuple(state, action, reward, next_state, done)
    """

    def __init__(self, replay_memory_size=10000):
        self.storage = []
        self.total_count = 0
        self.max_size = replay_memory_size

    @property
    def is_full(self):
        if self.total_count >= self.max_size:
            return True
        else:
            return False

    def push(self, data: Transition):
        if self.is_full:
            self.storage.pop(0)
            self.storage.append(data)
            return True
        else:
            self.storage.append(data)
            self.total_count += 1
            return True

    def sample(self, batch_size):
        imgs, vecs, actions, rewards, n_imgs, n_vecs, dones = map(np.array, zip(*self.storage[-batch_size:]))
        return imgs, vecs, actions, rewards, n_imgs, n_vecs, dones

    def clear(self):
        self.total_count = 0
        del self.storage[:]
        return

    def __len__(self):
        return len(self.storage)



class PPO(object):
    def __init__(
            self, name='PPO', log_dir='log', model_path=None,
            state_type='Image', image_type='Depth', seed=1,
            action_type='Discrete', actuator_type='Velocity',
            attention_type='No', data_augs=None,
            gpu_idx=0, action_dim=3, history_len=3,
            cam_far_distance=20, max_action=2,
            episode_nums=500, max_ep_len=1000,
            batch_size=32, lr=1e-4, gamma=0.99, epoch_num=10,
            replay_buffer_size=10000, clip_f=0.2,
            restore=False, is_train=True, with_SE=True,
            vis=False, headless=True
    ):
        self.name = name
        self.log_dir = log_dir
        assert state_type in ['Image', 'Pose', 'PoseImage', 'Position', 'PosImage'], print('Unsupported state type!')
        self.state_type = state_type
        assert image_type in ['Color', 'Depth', 'RGBD'], print('Unsupported image type!')
        self.image_type = image_type
        assert action_type in ['Discrete', 'Continuous'], print('Unsupported action type!')
        self.action_type = action_type
        assert actuator_type in ['Position', 'Velocity', 'Force'], print('Unsupported actuator type')
        self.actuator_type = actuator_type
        assert attention_type in ['MHA', 'No'], print('Unsupported attention type!')
        self.attention_type = attention_type

        if self.action_type == 'Continuous':
            self.continuous = True
        else:
            self.continuous = False

        self.action_dim = action_dim
        self.device = torch.device('cuda', gpu_idx) if gpu_idx < torch.cuda.device_count() \
            else torch.device('cpu', gpu_idx)
        self.seed(seed)
        self.data_augs = {}
        for aug_name in data_augs:
            assert aug_name in aug_to_func, 'invalid data augmentation method'
            self.data_augs[aug_name] = aug_to_func[aug_name]

        self.lr = lr
        self.gamma = gamma
        self.clip_f = clip_f
        self.batch_size = batch_size
        self.max_ep_len = max_ep_len
        self.max_action = max_action
        self.history_len = history_len
        self.vis = vis
        self.with_SE = with_SE
        self.headless = headless
        self.epoch_num = epoch_num
        self.episode_counter = 0
        self.global_step = 0
        self.episode_nums = episode_nums
        self.memory_counter = 0
        self.cam_far_distance = cam_far_distance
        self.input_channels = IMAGE_CHANNELS[self.image_type] * self.history_len
        self.memory = Memory(replay_memory_size=replay_buffer_size)
        self.model = ActorCritic(
            input_channels=self.input_channels,
            output_dim=self.action_dim,
            max_action=self.max_action,
            continuous=self.continuous,
            with_SE=self.with_SE,
            vis=self.vis,
        )

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.lr_scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9)

        if restore:
            if model_path is None:
                model_path = self.get_latest_model_path()
            if os.path.exists(model_path):
                self.reload_model(model_path, is_training=True)
            else:
                raise ValueError('No model file is found in {}'.format(model_path))

        self.model.to(self.device)
        self.is_training = is_train
        if not self.is_training:
            self.model.eval()
        else:
            self.model.train()
            self.summary_writer = SummaryWriter(log_dir=self.log_dir)

    def choose_action(self, state_dict: dict):
        state = self.state_dict_transfer(state_dict)
        with torch.no_grad():
            if self.continuous:
                mu, sigma = self.model.actor(state[0].to(self.device), state[1].to(self.device))
                dist = torch.distributions.Normal(mu, sigma)
                action = dist.sample()
                action_probs = dist.log_prob(action)
                return action.detach().cpu().numpy(), action_probs.detach().cpu().numpy()
            else:
                action_probs = self.model.actor(state[0].to(self.device), state[1].to(self.device))
                dist = torch.distributions.Categorical(action_probs)
                action_idx = dist.sample()
                return action_idx.detach().cpu().numpy(), action_probs.item()

    def learn(self, evaluator: AVTEval, scene_paths: list, eval_interval=100,
              dis_penalty_f=0.1, outview_penalty=5,
              episode_nums=None, epoch_num=None):
        if episode_nums is not None:
            self.episode_nums = episode_nums
        if epoch_num is not None:
            self.epoch_num = epoch_num

        scene_path = scene_paths[np.random.randint(0, len(scene_paths))]
        print('\n=> Start training ...')
        env = SNCOAT_Env_v2(
            name='SNCOAT_Env', scene_path=scene_path, log_dir=self.log_dir,
            action_type=self.action_type, state_type=self.state_type, image_type=self.image_type,
            actuator_type=self.actuator_type, headless=self.headless,
            dis_penalty_f=dis_penalty_f, outview_penalty=outview_penalty,
            clear_record=True, wait_done_steps=20, history_len=self.history_len,
            actuator_noise=False, time_delay=False, image_blur=False, visualization=False)
        reward_list = []
        now_reward = 0
        best_reward = -np.inf
        for ep in range(self.episode_counter, self.episode_nums):
            self.episode_counter += 1
            ep_reward = 0
            ep_len = 0
            state_dict = env.reset()
            while True:
                action, action_prob = self.choose_action(state_dict)
                n_state_dict, reward, done, _ = env.env_step(action)
                # print('\r=> action: {}, reward: {}'.format(action, reward), end="")
                self.store_transition(state_dict, action, reward, n_state_dict, done)
                ep_len += 1
                ep_reward += reward
                state_dict = n_state_dict
                if done or ep_len >= self.max_ep_len:
                    break
            loss = self.update()
            self.memory.clear()
            reward_list.append(ep_reward)
            now_reward = np.mean(reward_list[-10:])
            print('\n[Train]: episode num: {}, episode length:{}, last reward: {:0.3f}, best reward: {:0.3f}, loss: {:0.3f}, '.format(
                ep+1, ep_len, now_reward, best_reward, loss))
            if now_reward > best_reward:
                best_reward = now_reward
                best_model = os.path.join(self.log_dir, 'best_model.pth')
                self.save_model(best_model)
            if (ep+1) % eval_interval == 0:
                env.stop()
                env.shutdown()
                model_path = os.path.join(self.log_dir, 'model_ep_{:03d}.pth'.format(self.episode_counter))
                self.save_model(model_path)
                self.mode_transfer(is_training=False)
                evaluator.eval(
                    [self], action_type=self.action_type, actuator_type=self.actuator_type,
                    eval_episode=self.episode_counter, blur_level=2, max_episode_len=self.max_ep_len,
                    overwrite=True, headless=self.headless, image_blur=False, actuator_noise=False, time_delay=False
                )
                self.mode_transfer(is_training=True)
                scene_path = scene_paths[np.random.randint(0, len(scene_paths))]
                print('=> reloading scene in {} ...'.format(scene_path))
                env = SNCOAT_Env_v2(
                    name='SNCOAT_Env', scene_path=scene_path, log_dir=self.log_dir,
                    action_type=self.action_type, state_type=self.state_type, image_type=self.image_type,
                    actuator_type=self.actuator_type, headless=self.headless,
                    dis_penalty_f=dis_penalty_f, outview_penalty=outview_penalty,
                    clear_record=True, wait_done_steps=20, history_len=self.history_len,
                    actuator_noise=False, time_delay=False, image_blur=False, visualization=False)

        env.stop()
        env.shutdown()
        evaluator.train_report([self.name], actuator_type=self.actuator_type, plot=True)
        return

    def update(self):
        img, vec, a, r, n_img, n_vec, done = self.memory.sample(self.batch_size)
        img = torch.FloatTensor(img).to(self.device)
        vec = torch.FloatTensor(vec).to(self.device)
        if self.continuous:
            a = torch.tensor(a).view(-1, self.action_dim).to(self.device)
        else:
            a = torch.tensor(a).view(-1, 1).to(self.device)
        r = (r+8.0)/8.0
        r = torch.tensor(r).view(-1, 1).to(self.device)
        # r = (r - r.mean()) / (r.std()+1e-3)
        n_img = torch.FloatTensor(n_img).to(self.device)
        n_vec = torch.FloatTensor(n_vec).to(self.device)
        done = torch.FloatTensor(done).view(-1, 1).to(self.device)

        td_target = r + self.gamma * self.model.critic(n_img, n_vec) * (1 - done)
        td_error = td_target - self.model.critic(img, vec)
        adv = self.compute_advantage(self.gamma, 0.9, td_error.cpu()).to(self.device)
        if self.continuous:
            mu, sig = self.model.actor(img, vec)
            dist = torch.distributions.Normal(mu, sig)
            action_prob_old = dist.log_prob(a).detach()
        else:
            prob = self.model.actor(img, vec)
            dist = torch.distributions.Categorical(prob)
            action_prob_old = dist.log_prob(a.view(-1)).detach()
        losses = []
        for epoch in range(self.epoch_num):
            if self.continuous:
                mu, sig = self.model.actor(img, vec)
                dist = torch.distributions.Normal(mu, sig)
                action_prob_new = dist.log_prob(a)
            else:
                prob = self.model.actor(img, vec)
                dist = torch.distributions.Categorical(prob)
                action_prob_new = dist.log_prob(a.view(-1))

            ratio = torch.exp(action_prob_new - action_prob_old)
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1-self.clip_f, 1+self.clip_f) * adv
            actor_loss = torch.mean(-torch.min(surr1, surr2)).float()
            critic_loss = F.mse_loss(self.model.critic(img, vec).float(), td_target.detach().float()).float()
            loss = 0.1 * actor_loss + critic_loss
            if torch.isnan(loss).item():
                print('=> reward: {}'.format(r))
                print('=> action: {}'.format(a))
                print('=> old action prob: {}'.format(action_prob_old))
                print('=> new action prob: {}'.format(action_prob_new))
                print('=> mu: {}'.format(mu))
                print('=> sigma: {}'.format(sig))
                print('=> advantage: {}'.format(adv))
                sys.exit(0)
            if loss.item() > 50:
                continue
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            self.optimizer.step()
            losses.append(loss.item())
            print('=> epoch num: {}, loss: {:0.3f}'.format(epoch, loss.item()))
        return np.mean(losses)

    def store_transition(self, state_dict, action, reward, next_state_dict, terminal):
        image = self.image_transform(state_dict['image'])
        next_image = self.image_transform(next_state_dict['image'])
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

    def image_transform(self, image):
        """
        Normalize depth channel of state
        :param image: Image(NxMxWxH)
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
        if self.data_augs is not None:
            for i, aug_name in enumerate(self.data_augs):
                if aug_name == 'SensorNoise':
                    continue
                else:
                    image = self.data_augs[aug_name](image)

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

    def seed(self, s):
        np.random.seed(s)
        torch.manual_seed(s)
        if self.device.type == 'gpu':
            torch.cuda.manual_seed(s)

    def mode_transfer(self, is_training: bool):
        self.is_training = is_training
        self.model.to(self.device)
        if not self.is_training:
            self.model.eval()
        else:
            self.model.train()

    def save_model(self, model_path=''):
        print('=> saving network to {} ...'.format(model_path))
        checkpoint = {'steps': self.global_step,
                      'episodes': self.episode_counter,
                      'actor_critic': self.model.state_dict(),
                      'lr_scheduler': self.lr_scheduler.state_dict()}
        try:
            torch.save(checkpoint, model_path, _use_new_zipfile_serialization=False)
            print('=> model params is saved!')
        except:
            print('=> failed to save model!')

        return

    def reload_model(self, model_path='', is_training=False):
        if os.path.exists(model_path):
            print('=> restoring model from {} ...'.format(model_path))
            checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage.cpu())
            self.model.load_state_dict(checkpoint['actor_critic'], strict=False)
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            self.episode_counter = checkpoint['episodes']
            self.global_step = checkpoint['steps'] + 1
        else:
            raise ValueError('No model file is found in {}'.format(model_path))
        self.mode_transfer(is_training)

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

    @staticmethod
    def compute_advantage(gamma, lmbda, td_delta):
        td_delta = td_delta.detach().numpy()
        adv_list = []
        adv = 0
        for delta in td_delta[::-1]:
            adv = gamma * lmbda * adv + delta
            adv_list.append(adv)
        adv_list.reverse()
        return torch.FloatTensor(adv_list)