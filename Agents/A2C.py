import numpy as np
import torch
import os
import glob
import torch.nn as nn
import torch.optim as optim
import Utils.data_augs as rad
from Config.env_setting import IMAGE_CHANNELS
from Models.actor_critic_models import ActorCritic
from Utils.replay_memory import Memory, Transition
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


class A2C(object):
    def __init__(
            self, name='A2C', log_dir='log', seed=1,
            state_type='Image', image_type='Color', model_path=None,
            action_type='Continuous', actuator_type='Velocity',
            attention_type='No', data_augs=None,
            action_dim=3, gpu_idx=0,
            min_history=4, state_to_update=8,
            lstm_dim=512, lstm_layers=2,
            global_steps=50000, batch_size=32,
            lr=1e-4, gamma=0.99, cam_far_distance=20,
            max_ep_len=1000, history_len=3,
            restore=True, is_train=True,
            with_SE=True, vis=False, headless=True):
        self.name = name
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
        self.batch_size = batch_size
        self.max_ep_len = max_ep_len
        self.history_len = history_len
        self.log_dir = log_dir
        self.vis = vis
        self.with_SE = with_SE
        self.headless = headless
        self.min_history = min_history
        self.state_to_update = state_to_update
        self.lstm_dim = lstm_dim
        self.lstm_layers = lstm_layers

        self.episode_counter = 0
        self.global_step = 0
        self.global_steps = global_steps
        self.memory_counter = 0
        self.cam_far_distance = cam_far_distance
        self.input_channels = IMAGE_CHANNELS[self.image_type] * self.history_len

        self.model = ActorCritic(input_channels=self.input_channels, output_dim=self.action_dim,
                                 with_SE=self.with_SE, vis=self.vis)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.lr_scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9)

        if restore:
            if model_path is None:
                model_path = self.get_latest_model_path()
            if os.path.exists(model_path):
                print('=> restoring model from {} ...'.format(model_path))
                checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage.cpu())
                self.model.load_state_dict(checkpoint['actor_critic'], strict=False)
                self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
                self.global_step = checkpoint['step'] + 1
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
            dist, value = self.model(state[0].to(self.device), state[1].to(self.device))
            action_idx = dist.sample().item()
        return action_idx, DISCRETE_ACTIONS[action_idx]

    def rollout(self, env: SNCOAT_Env_v2, init_state_dict, sample_num=5) -> Tuple[list, list, dict]:
        actor_loss = []
        critic_loss = []
        state_dict = init_state_dict
        for i in range(sample_num):
            state = self.state_dict_transfer(state_dict, aug_funcs=self.data_augs)
            dist, value = self.model(state[0].to(self.device), state[1].to(self.device))
            action_idx = dist.sample()
            action_prob = dist.log_prob(action_idx)
            action = DISCRETE_ACTIONS[action_idx.item()]
            next_state_dict, reward, done, info = env.env_step(action)
            state_dict = next_state_dict
            next_state = self.state_dict_transfer(next_state_dict, aug_funcs=self.data_augs)
            next_value = self.model.critic(next_state[0].to(self.device), next_state[1].to(self.device))
            reward = torch.Tensor([reward]).unsqueeze(0).to(self.device)
            torch_done = torch.FloatTensor([done]).unsqueeze(0).to(self.device)
            g = reward + self.gamma * next_value * (1 - torch_done)
            adv = g - value
            actor_loss.append(-action_prob * adv)
            critic_loss.append(adv.pow(2))
            if done or env.step_count > self.max_ep_len:
                state_dict = env.reset()
        return actor_loss, critic_loss, state_dict

    def learn(self, evaluator: AVTEval, scene_paths: list, eval_interval=1000,
              rollout_step=5, global_steps=None,
              dis_penalty_f=0.1, outview_penalty=5):
        if global_steps is not None:
            self.global_steps = global_steps

        scene_path = scene_paths[np.random.randint(0, len(scene_paths))]
        print('\n=> Start training ...')
        env = SNCOAT_Env_v2(
            name='SNCOAT_Env', scene_path=scene_path, log_dir=self.log_dir,
            dis_penalty_f=dis_penalty_f, outview_penalty=outview_penalty,
            action_type=self.action_type, state_type=self.state_type, image_type=self.image_type,
            actuator_type=self.actuator_type, headless=self.headless,
            clear_record=True, wait_done_steps=10, history_len=self.history_len,
            actuator_noise=False, time_delay=False, image_blur=False, visualization=False)
        state_dict = env.reset()
        for i in range(self.global_step, self.global_steps):
            self.global_step += 1
            actor_loss, critic_loss, final_state = self.rollout(env, state_dict, rollout_step)
            actor_loss = torch.cat(actor_loss)
            critic_loss = torch.cat(critic_loss)
            loss = 0.1 * actor_loss.mean() + critic_loss.mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            print('\r=> Train: step {}/{}, loss: {}'.format(self.global_step, self.global_steps, loss.item()), end="")

            if (i + 1) % eval_interval == 0:
                env.stop()
                env.shutdown()
                model_path = os.path.join(self.log_dir, 'model_ep_{:03d}.pth'.format(self.global_step + 1))
                print('\n=> saving network to {} ...'.format(model_path))
                checkpoint = {'step': self.global_step,
                              'actor_critic': self.model.state_dict(),
                              'lr_scheduler': self.lr_scheduler.state_dict()}
                torch.save(checkpoint, model_path, _use_new_zipfile_serialization=False)
                print('=> model params is saved!')

                self.mode_transfer(is_training=False)
                evaluator.eval(
                    [self], action_type=self.action_type, actuator_type=self.actuator_type,
                    eval_episode=self.global_step, blur_level=2, max_episode_len=self.max_ep_len,
                    overwrite=True, headless=True, image_blur=False, actuator_noise=False, time_delay=False
                )
                self.mode_transfer(is_training=True)
                scene_path = scene_paths[np.random.randint(0, len(scene_paths))]
                print('=> reloading scene in {} ...'.format(scene_path))
                env = SNCOAT_Env_v2(
                    name='SNCOAT_Env', scene_path=scene_path, log_dir=self.log_dir,
                    dis_penalty_f=dis_penalty_f, outview_penalty=outview_penalty,
                    action_type=self.action_type, state_type=self.state_type, image_type=self.image_type,
                    actuator_type=self.actuator_type, headless=self.headless,
                    clear_record=True, wait_done_steps=10, history_len=self.history_len,
                    actuator_noise=False, time_delay=False, image_blur=False, visualization=False)
                state_dict = env.reset()
        evaluator.train_report([self.name], actuator_type=self.actuator_type, plot=True)

    def state_transform(self, state_dict):
        """
        Normalize depth channel of state
        :param state_dict: the state dict including Image(NxMxWxH), Position(NxMx3), Orientation(NxMx3), and Pose(NxMx6)
        :param env: the simulation env
        :return: norm_image
        """
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
        norm_image = np.concatenate(np.transpose(norm_image, (0, 3, 1, 2)))
        state_dict['image'] = norm_image
        return state_dict

    def state_dict_transfer(self, state_dict: dict, aug_funcs=None) -> Tuple[torch.Tensor, torch.Tensor]:
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
