import os
import time
import math
import argparse
import imageio
import numpy as np
import glob
import shutil
import matplotlib.pyplot as plt

from pyrep import PyRep
from pyrep.backend import sim
from pyrep.objects.vision_sensor import VisionSensor
from multiprocessing import Process
from gym import spaces

SCENES = [
    'SNCOAT-Asteroid-v0.ttt', 'SNCOAT-Asteroid-v1.ttt', 'SNCOAT-Asteroid-v2.ttt',
    'SNCOAT-Asteroid-v3.ttt', 'SNCOAT-Asteroid-v4.ttt', 'SNCOAT-Asteroid-v5.ttt',
    'SNCOAT-Capsule-v0.ttt', 'SNCOAT-Capsule-v1.ttt', 'SNCOAT-Capsule-v2.ttt',
    'SNCOAT-Rocket-v0.ttt', 'SNCOAT-Rocket-v1.ttt', 'SNCOAT-Rocket-v2.ttt',
    'SNCOAT-Satellite-v0.ttt', 'SNCOAT-Satellite-v1.ttt', 'SNCOAT-Satellite-v2.ttt',
    'SNCOAT-Station-v0.ttt', 'SNCOAT-Station-v1.ttt', 'SNCOAT-Station-v2.ttt',
]

PROCESS_NUM = 5
SCENES_DIR = '/home/group1/dzhou/SNCOAT/Scenes/eval'

# action format: force in x, y, z axis, and yaw & pitch velocity
DISCRETE_ACTIONS = np.array([
    [0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0],
    [-1, 0, 0, 0, 0],
    [0, 1, 0, 0, 0],
    [0, -1, 0, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, -1, 0, 0],
    [0, 0, 0, 1, 0],
    [0, 0, 0, -1, 0],
    [0, 0, 0, 0, 1],
    [0, 0, 0, 0, -1]
])


class SNCOAT_Env_v2(PyRep):
    def __init__(self, scene_path='', headless=True, action_type='Discrete', observation_type='Color',
                 log_dir='', wait_done_steps=10, visualization=False):
        super(SNCOAT_Env_v2, self).__init__()
        self.name = 'SNCOAT_Env_v2'

        self.action_types = ['Discrete', 'Continuous']
        self.observation_types = ['Color', 'Depth', 'RGBD']

        self.action_type = action_type
        assert self.action_type in self.action_types

        self.observation_type = observation_type
        assert self.observation_type in observation_type

        self.target_pos = None
        self.target_ang = None
        self.target_motion_vel = None
        self.target_rotate_vel = None

        self.chaser_pos = np.zeros(3, dtype=np.float32)
        self.chaser_ang = np.zeros(3, dtype=np.float32)
        self.chaser_motion_vel = np.zeros(3)
        self.chaser_rotation_vel = np.zeros(3)

        self.delta_time = 0.2
        self.expected_pos = np.array([0, 0, 5])
        self.step_count = 0
        self.done_count = 0
        self.wait_done_steps = wait_done_steps
        self.is_running = False

        self.record = False
        self.record_imgs = []
        self.record_dir = os.path.join(log_dir, self.name, 'record')
        if not os.path.exists(self.record_dir):
            print('=> not found record directory, it will be created soon ...')
            os.makedirs(self.record_dir)
        else:
            print('=> found record directory, it will be deleted soon ...')
            shutil.rmtree(self.record_dir)
            print('=> creating record directory again ...')
            os.makedirs(self.record_dir)

        self.record_file_path = None

        self.launch(scene_path, headless=headless)

        # get Target, Chaser and camera handle
        self.target_handle = sim.simGetObjectHandle('Target')
        self.chaser_handle = sim.simGetObjectHandle('Chaser')
        self.camera_handle = sim.simGetObjectHandle('camera')
        self.camera = VisionSensor(self.camera_handle)

        self.camera_resolution = sim.simGetVisionSensorResolution(self.camera_handle)
        self.cam_near_distance = sim.simGetObjectFloatParameter(self.camera_handle,
                                                                sim.sim_visionfloatparam_near_clipping)
        self.cam_far_distance = sim.simGetObjectFloatParameter(self.camera_handle,
                                                               sim.sim_visionfloatparam_far_clipping)
        # print('=> camera resolution: ({}, {})'.format(self.camera_resolution[0], self.camera_resolution[1]))
        # print('=> camera near/far plane: {:0.2f}/{:0.2f} m'.format(self.cam_near_distance, self.cam_far_distance))
        # get camera fov_angle
        self.fov_ang_x, self.fov_ang_y = self._get_camera_fov_angle()
        # print('=> fov angle: ({:0.2f}, {:0.2f})'.format(self.fov_ang_x, self.fov_ang_y))
        self.chaser_mass = sim.simGetObjectFloatParameter(self.chaser_handle, sim.sim_shapefloatparam_mass)
        # print('=> the mass of chaser: {}'.format(self.chaser_mass))

        print('=> start env ...')
        self.start()

        if self.action_type == 'Discrete':
            self.action_space = spaces.Discrete(len(DISCRETE_ACTIONS))
        elif self.action_type == 'Continuous':
            self.action_space = spaces.Box(low=[-1, -1, -1, -1, -1],
                                           high=[1, 1, 1, 1, 1], shape=(5,))
        else:
            raise ValueError('=> Error action type. Only \'Discrete\' and \'Continuous\' are supported!')

        self.observation_space = self._define_observation(self.observation_type)

        self.visualization = visualization

    def reset(self, is_record=False):
        self.record = is_record
        self.record_imgs.clear()
        self.record_file_path = os.path.join(self.record_dir, 'SNCOAT_{}.gif'.format(time.time()))

        # print('=> resetting env ...')
        start_time = time.time()
        # re-initial chaser position and angle
        # print('=> re-initializing chaser position and angle ...')
        self.chaser_pos = np.zeros(3)
        self.chaser_ang = np.zeros(3)
        self.chaser_motion_vel = np.zeros(3)
        sim.simSetObjectPosition(self.chaser_handle, -1, list(self.chaser_pos))
        sim.simSetObjectOrientation(self.chaser_handle, -1, list(self.chaser_ang))

        # re-init target position and angle
        # print('=> re-initializing target position and angle ...')
        self.target_pos = [np.random.randn(), np.random.randn(), np.random.randint(2, 8)]
        self.target_ang = np.random.randint(-90, 90, 3).astype(np.float32)
        # print('=> initial position: {}'.format(self.target_pos))
        # print('=> initial angle: {}'.format(self.target_ang))

        sim.simSetObjectPosition(self.target_handle, self.camera_handle, list(self.target_pos))
        sim.simSetObjectOrientation(self.target_handle, self.camera_handle, list(self.target_ang))

        self.target_pos = np.array(sim.simGetObjectPosition(self.target_handle, -1))
        self.target_ang = np.array(sim.simGetObjectOrientation(self.target_handle, -1))

        # re-initial target motion and rotation velocity
        self.target_motion_vel = np.random.randn(3) * 0.2
        self.target_rotate_vel = np.random.randn(3) * 0.1

        # re-inital done condition count
        self.done_count = 0
        self.step_count = 0

        end_time = time.time()
        print('=> env is reset!')
        reset_time = end_time - start_time
        # print('=> time for reset env: {}s'.format(reset_time))
        state = self._get_camera_image()
        if self.visualization:
            plt.close()
            plt.figure()
            plt.ion()
            plt.imshow(state)
            plt.pause(0.001)

        return state

    def env_step(self, action):
        self.step_count += 1
        single_step_reward = 0
        # target part
        delta_target_pos = self.target_motion_vel * self.delta_time
        delta_target_angle = self.target_rotate_vel * self.delta_time
        self.target_pos += delta_target_pos
        self.target_ang += delta_target_angle
        sim.simSetObjectPosition(self.target_handle, -1, list(self.target_pos))
        sim.simSetObjectOrientation(self.target_handle, -1, list(self.target_ang))

        # action implementation
        # ac_idx = np.argmax(action)
        # real_action = np.array(DISCRETE_ACTIONS[ac_idx])
        # print('=> real action ({}th class): {}'.format(ac_idx, real_action))
        acceleration = action[:3] * 100 / self.chaser_mass
        delta_chaser_vel = acceleration * self.delta_time
        new_chaser_motion_vel = self.chaser_motion_vel + delta_chaser_vel
        # print('=> chaser velocity: {}'.format(new_chaser_motion_vel))
        delta_chaser_pos = (self.chaser_motion_vel + new_chaser_motion_vel) * self.delta_time / 2
        self.chaser_pos += delta_chaser_pos
        # print('=> chaser position: {}'.format(self.chaser_pos))
        sim.simSetObjectPosition(self.chaser_handle, -1, list(self.chaser_pos))
        self.chaser_motion_vel = new_chaser_motion_vel

        self.chaser_ang[0] += action[3] * 5
        self.chaser_ang[1] += action[4] * 5

        # delta_chaser_ang = self.chaser_rotation_vel * self.delta_time
        # self.chaser_ang += delta_chaser_ang
        # print('=> chaser angle in degrees: {}'.format(self.chaser_ang))
        sim.simSetObjectOrientation(self.chaser_handle, -1, list(np.radians(self.chaser_ang)))

        self.step()

        # get new state
        state = self._get_camera_image()

        if self.record:
            self.record_imgs.append(state)

        if self.visualization:
            plt.imshow(state)
            plt.pause(0.001)

        # reward calculation
        # propel constraint
        if (action == 0).any():
            single_step_reward += 5
        else:
            single_step_reward += 1

        # normal reward: whether the target is in the field of view
        target_pos_in_camera = np.array(sim.simGetObjectPosition(self.target_handle, self.camera_handle))
        in_view = self._is_target_in_camera_view(target_pos_in_camera)
        if in_view:
            single_step_reward += 5
        else:
            single_step_reward -= 5
            # print('=> target is out of view!')

        # distance constraint
        distance_res = np.linalg.norm(self.expected_pos - target_pos_in_camera, ord=2)
        # print('=> distance to expected position ({}): {} m'.format(self.expected_pos, distance_res))
        single_step_reward -= distance_res

        if not in_view or distance_res > 15:
            self.done_count += 1
        else:
            self.done_count = 0

        if self.done_count > self.wait_done_steps:
            is_done = True
            if self.record and len(self.record_imgs) != 0:
                print('=> saving video to {} ...'.format(self.record_file_path))
                imageio.mimsave(self.record_file_path, self.record_imgs, 'GIF', duration=self.delta_time)
        else:
            is_done = False

        return state, single_step_reward, is_done, {}

    def _get_camera_fov_angle(self):
        cam_fov_angle = sim.simGetObjectFloatParameter(self.camera_handle, sim.sim_visionfloatparam_perspective_angle)

        ratio = self.camera_resolution[0] / self.camera_resolution[1]
        if ratio > 1:
            fov_angle_x = cam_fov_angle
            fov_angle_y = 2 * math.atan(math.tan(cam_fov_angle / 2) / ratio)
        else:
            fov_angle_x = 2 * math.atan(math.tan(cam_fov_angle / 2) / ratio)
            fov_angle_y = cam_fov_angle

        return fov_angle_x, fov_angle_y

    def _get_camera_image(self):
        if self.observation_type == 'Color':
            # image = self.camera.capture_rgb()
            image = sim.simGetVisionSensorImage(self.camera_handle, self.camera_resolution)
            # image = image / 255.0
            return image

        elif self.observation_type == 'Depth':
            depth_img = sim.simGetVisionSensorDepthBuffer(self.camera_handle, self.camera_resolution, True)
            depth_img = 1 - depth_img / depth_img.max()
            return depth_img

        elif self.observation_type == 'RGBD':
            image = sim.simGetVisionSensorImage(self.camera_handle, self.camera_resolution)
            # image = image / 255.0
            depth_img = sim.simGetVisionSensorDepthBuffer(self.camera_handle, self.camera_resolution, True)
            depth_img = 1 - depth_img / depth_img.max()
            rgbd = np.append(image, np.expand_dims(depth_img, 2), axis=2)
            return rgbd

    def _is_target_in_camera_view(self, object_pos):
        # the position of object should be in camera coordinate system
        x, y, z = object_pos
        # print('x_lim:({}, {})'.format(- z * math.tan(self.fov_ang_x / 2), z * math.tan(self.fov_ang_x / 2)))
        # print('y_lim:({}, {})'.format(- z * math.tan(self.fov_ang_y / 2), z * math.tan(self.fov_ang_y / 2)))
        if (self.cam_near_distance < z < self.cam_far_distance) and \
                (- z * math.tan(self.fov_ang_x / 2) < x < z * math.tan(self.fov_ang_x / 2)) and \
                (- z * math.tan(self.fov_ang_y / 2) < y < z * math.tan(self.fov_ang_y / 2)):
            flag = True
        else:
            flag = False

        return flag

    def _define_observation(self, observation_type):
        # note that when use this api
        # the client_id and camera handle should be got
        # that is, observation definition should behind the initialization of CoppeliaSim
        state = self._get_camera_image()

        if observation_type == 'Color':
            observation_space = spaces.Box(low=0, high=255, shape=state.shape, dtype=np.float32)
        elif observation_type == 'Depth':
            observation_space = spaces.Box(low=0, high=1, shape=state.shape, dtype=np.float32)
        elif observation_type == 'RGBD':
            s_high = state
            s_high[:, :, -1] = 1
            s_high[:, :, :-1] = 255
            s_low = np.zeros(state.shape)
            observation_space = spaces.Box(low=s_low, high=s_high, dtype=np.float32)
        else:
            raise ValueError('=> input unsupported observation type!')

        return observation_space

    def run(self, episode_num=30):
        """
        Run the test simulation without any learning algorithm for debugging purposes
        """
        try:
            for t in range(episode_num):
                if t % 5 == 0:
                    self.reset(True)
                else:
                    self.reset(False)
                start_time = time.time()
                while True:
                    state, reward, done, _ = self.env_step(self.action_space.sample())
                    # print(reward)
                    if done:
                        break
                end_time = time.time()
                elapsed_time = end_time - start_time
                print('=> speed for one episode with {} timesteps: {:0.2f} Hz'.format(self.step_count,
                                                                                      self.step_count / elapsed_time))
        except KeyboardInterrupt:
            pass


def run(thread_num=0):
    scenes_paths = sorted(glob.glob(os.path.join(SCENES_DIR, '*.ttt')))
    env = SNCOAT_Env_v2(scenes_paths[thread_num], headless=True, action_type='Discrete', observation_type='Depth',
                        log_dir='log/log_{:02d}'.format(thread_num + 1), wait_done_steps=10)
    env.run(episode_num=20)
    env.stop()


if __name__ == '__main__':
    # PROCESS_NUM = len(glob.glob(os.path.join(SCENS_DIR, '*.ttt')))
    # print('=> Found {} scenes in {}'.format(PROCESS_NUM, SCENS_DIR))
    PROCESS_NUM = 1
    processes = [Process(target=run, args=(i,)) for i in range(PROCESS_NUM)]
    [p.start() for p in processes]
    [p.join() for p in processes]


