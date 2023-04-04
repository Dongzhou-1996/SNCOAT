import os
import sys
import cv2
import time
import math
import imageio
import glob
import shutil
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

sys.path.append('/home/group1/dzhou/RAMAVT')

from Config.env_setting import DISCRETE_ACTIONS, TEST_SCENES_DIR
from pyrep import PyRep
from pyrep.backend import sim
from pyrep.objects.vision_sensor import VisionSensor
from multiprocessing import Process
from gym import spaces

def runtime(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print('=> execution time: {:0.3f} ms'.format((end - start) * 1000))
        return result

    return wrapper


class SNCOAT_Env_v2(PyRep):
    def __init__(
            self, name='SNCOAT_Env', log_dir='', scene_path='',
            state_type='Image', image_type='Color',
            action_type='Discrete', actuator_type='Force',
            headless=True, history_len=4, max_output=5, dis_penalty_f=1, outview_penalty=5,
            actuator_noise=False, time_delay=False, image_blur=False, blur_level=1,
            clear_record=False, wait_done_steps=10, visualization=False):
        super(SNCOAT_Env_v2, self).__init__()

        self.action_types = ['Discrete', 'Continuous']
        self.state_types = ['Image', 'Pose', 'PoseImage', 'Position', 'PosImage', 'OrientImage']
        self.image_types = ['Color', 'Depth', 'RGBD']
        self.actuator_types = ['Force', 'Velocity', 'Position']

        self.action_type = action_type
        assert self.action_type in self.action_types

        self.state_type = state_type
        assert self.state_type in self.state_types

        self.image_type = image_type
        assert self.image_type in self.image_types

        self.actuator_type = actuator_type
        assert self.actuator_type in self.actuator_types

        self.name = name + '_' + actuator_type

        self.actuator_noise = actuator_noise
        self.time_delay = time_delay
        self.image_blur = image_blur
        assert blur_level in range(1, 5)
        self.blur_level = blur_level + 1
        self.history_len = history_len
        self.dis_penalty_f = dis_penalty_f
        self.outview_penalty = outview_penalty
        self.target_pos = None
        self.target_pos_in_camera = None
        self.target_ang = None
        self.target_ang_in_camera = None
        self.target_motion_vel = None
        self.target_rotate_vel = None

        self.chaser_pos = np.zeros(3, dtype=np.float32)
        self.chaser_ang = np.zeros(3, dtype=np.float32)
        self.chaser_motion_vel = np.zeros(3)
        self.chaser_rotation_vel = np.zeros(3)

        self.delta_time = 0.1
        # self.random_init = None
        # self.random_freq = None
        self.expected_pos = np.array([0, 0, 5])
        # self.expected_ang_vel = np.array([0, 0, 0])
        self.step_count = 0
        self.done_count = 0
        self.wait_done_steps = wait_done_steps
        self.is_running = False

        self.pre_images = []
        self.record = False
        self.record_imgs = []
        self.record_trajectory = []
        self.record_actions = []

        self.record_dir = os.path.join(log_dir, self.name, 'record')
        if not os.path.exists(self.record_dir):
            print('=> not found record directory, it will be created soon ...')
            os.makedirs(self.record_dir)
        elif clear_record:
            print('=> found record directory, it will be deleted soon ...')
            shutil.rmtree(self.record_dir)
            print('=> creating record directory again ...')
            os.makedirs(self.record_dir)

        self.record_video_path = None
        self.record_trajectory_path = None
        self.record_action_path = None

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
        self.camera_params_o3d, self.camera_matrix = self._get_o3d_camera_intrinsic_matrix()

        # get camera fov_angle
        self.fov_ang_x, self.fov_ang_y = self._get_camera_fov_angle()
        # print('=> fov angle: ({:0.2f}, {:0.2f})'.format(self.fov_ang_x, self.fov_ang_y))
        self.chaser_mass = sim.simGetObjectFloatParameter(self.chaser_handle, sim.sim_shapefloatparam_mass)
        # print('=> the mass of chaser: {}'.format(self.chaser_mass))

        print('=> start env ...')
        self.start()
        self.max_output = max_output
        if self.action_type == 'Discrete':
            self.action_space = spaces.Discrete(len(DISCRETE_ACTIONS))
        elif self.action_type == 'Continuous':
            self.action_space = spaces.Box(low=np.array([-1, -1, -1, -1, -1, -1]) * max_output,
                                           high=np.array([1, 1, 1, 1, 1, 1]) * max_output, dtype=float)
        else:
            raise ValueError('=> Error action type. Only \'Discrete\' and \'Continuous\' are supported!')

        self.observation_space = self._define_observation(self.image_type)

        self.visualization = visualization
        pass

    def reset(self, is_record=False) -> dict:
        # re-inital done condition count
        self.done_count = 0
        self.step_count = 0
        # self.random_init = np.random.randint(0, 360, 2)
        # self.random_freq = np.random.randint(1, 5, 2)

        self.record = is_record
        self.record_imgs.clear()
        self.record_trajectory.clear()
        self.record_actions.clear()
        record_time = time.time()
        self.record_video_path = os.path.join(self.record_dir, 'SNCOAT_{}_original.gif'.format(record_time))
        self.record_trajectory_path = os.path.join(self.record_dir, 'SNCOAT_{}_trajectory.txt'.format(record_time))
        self.record_action_path = os.path.join(self.record_dir, 'SNCOAT_{}_action.txt'.format(record_time))
        self.pre_images = []

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
        # self.target_pos = np.array([np.random.randn(), np.random.randn(), np.random.randint(2, 12)])
        # self.target_ang = np.random.randint(-90, 90, 3).astype(np.float32)
        self.target_pos = [2, 2, 7]
        self.target_ang = np.array([30, 30, 60])
        # print('=> initial position: {}'.format(self.target_pos))
        # print('=> initial angle: {}'.format(self.target_ang))

        sim.simSetObjectPosition(self.target_handle, self.camera_handle, list(self.target_pos))
        sim.simSetObjectOrientation(self.target_handle, self.camera_handle, list(self.target_ang))

        self.target_pos = np.array(sim.simGetObjectPosition(self.target_handle, -1))
        self.target_pos_in_camera = np.array(sim.simGetObjectPosition(self.target_handle, self.camera_handle))
        self.target_ang = np.array(sim.simGetObjectOrientation(self.target_handle, -1))
        self.target_ang_in_camera = np.array(sim.simGetObjectOrientation(self.target_handle, self.camera_handle))

        self.record_trajectory.append(np.array(
            [self.step_count] + list(self.target_pos) + list(self.target_ang) +
            list(self.chaser_pos) + list(self.chaser_ang))
        )

        # re-initial target motion and rotation velocity
        # self.target_motion_vel = np.random.randn(3) * 0.3
        # self.target_rotate_vel = np.random.randn(3) * 0.1

        self.target_motion_vel = np.array([0.1, 0.2, 0.1])
        self.target_rotate_vel = np.zeros(3)

        end_time = time.time()
        # print('=> env is reset!')
        # reset_time = end_time - start_time
        # print('=> time for reset env: {}s'.format(reset_time))
        self.step()

        image = self._get_camera_image()
        pos = self.target_pos_in_camera.astype(np.float32)
        orient = self.target_ang_in_camera.astype(np.float32)
        stack_image = np.array([image] * self.history_len)
        stack_position = np.array([pos] * self.history_len)
        stack_orient = np.array([orient] * self.history_len)
        stack_pose = np.array([np.concatenate([pos, orient])] * self.history_len)

        self.pre_images.append(image)

        if self.visualization:
            plt.close()
            plt.figure()
            plt.ion()
            plt.imshow(image)
            plt.pause(0.001)

        state_dict = {
            'image': np.array(stack_image),
            'position': np.array(stack_position, dtype=np.float32),
            'orientation': np.array(stack_orient, dtype=np.float32),
            'pose': np.array(stack_pose, dtype=np.float32)
        }

        return state_dict

    # @runtime
    def env_step(self, action: np.ndarray or int):
        if self.action_type == 'Discrete':
            action = DISCRETE_ACTIONS[action]
        else:
            action = action.reshape(-1)
        if action.shape[0] == 3:
            action = np.concatenate([action, np.zeros(3)], axis=0)
        assert action.shape[0] == 6, print('Wrong action dim: {}'.format(action.shape))
        action = np.clip(action, -5, 5)

        multi_step_reward = 0
        stack_image = []
        stack_position = []
        stack_orient = []
        stack_pose = []

        is_done = False
        self.step_count += 1

        if self.time_delay:
            delta_time = self.delta_time + np.random.rand() * 0.5
        else:
            delta_time = self.delta_time

        if self.actuator_noise:
            action = action * (1 + np.random.randn() * 0.3)

        for _ in range(self.history_len):
            single_step_reward = 0
            self.record_actions.append(action)

            # target part
            delta_target_pos = self.target_motion_vel * delta_time
            delta_target_angle = self.target_rotate_vel * delta_time
            self.target_pos += delta_target_pos
            self.target_ang += delta_target_angle
            sim.simSetObjectPosition(self.target_handle, -1, list(self.target_pos))
            sim.simSetObjectOrientation(self.target_handle, -1, list(self.target_ang))

            if self.actuator_type == 'Force':
                acceleration = action[:3] * 10.0 / self.chaser_mass
                # torque = action[3:]
                delta_chaser_t_vel = acceleration * delta_time
                # delta_chaser_r_vel = torque * delta_time
                new_chaser_motion_vel = self.chaser_motion_vel + delta_chaser_t_vel
                # new_chaser_rotation_vel = self.target_rotate_vel + delta_chaser_r_vel
                # print('=> chaser velocity: {}'.format(new_chaser_motion_vel))
                delta_chaser_pos = (self.chaser_motion_vel + new_chaser_motion_vel) * delta_time / 2
                # delta_chaser_ang = (self.chaser_rotation_vel + new_chaser_rotation_vel) * delta_time / 2

            elif self.actuator_type == 'Velocity':
                # delta_chaser_vel = action[:3] * 0.5
                # new_chaser_motion_vel = self.chaser_motion_vel + delta_chaser_vel
                new_chaser_motion_vel = action[:3] * 1.5
                # new_chaser_rotation_vel = action[3:]
                delta_chaser_pos = (self.chaser_motion_vel + new_chaser_motion_vel) * delta_time / 2
                # delta_chaser_ang = (self.chaser_rotation_vel + new_chaser_rotation_vel) * delta_time / 2

            else:
                new_chaser_motion_vel = np.zeros(3)
                # new_chaser_rotation_vel = np.zeros(3)
                delta_chaser_pos = action[:3]
                # delta_chaser_ang = action[3:]

            self.chaser_pos += delta_chaser_pos
            # self.chaser_ang += delta_chaser_ang
            # print('=> chaser position: {}'.format(self.chaser_pos))
            sim.simSetObjectPosition(self.chaser_handle, -1, list(self.chaser_pos))
            # sim.simSetObjectOrientation(self.chaser_handle, -1, list(self.chaser_ang))
            self.chaser_motion_vel = new_chaser_motion_vel
            # self.chaser_rotation_vel = new_chaser_rotation_vel

            self.record_trajectory.append(np.array(
                [self.step_count] + list(self.target_pos) + list(self.target_ang) +
                list(self.chaser_pos) + list(self.chaser_ang)
            ))

            self.step()
            # get new state

            image = self._get_camera_image()
            pos = self.target_pos_in_camera.astype(np.float32)
            orient = self.target_ang_in_camera.astype(np.float32)
            pose = np.concatenate([pos, orient])

            if self.image_blur:
                if len(self.pre_images) < self.blur_level:
                    self.pre_images.append(image)
                else:
                    blur_image = np.mean(np.array(self.pre_images[-self.blur_level:]), axis=0)
                    self.pre_images.append(image)
                    self.pre_images.pop(0)
                    image = blur_image

            # stack lasted history states
            stack_image.append(image)
            stack_position.append(pos)
            stack_orient.append(orient)
            stack_pose.append(pose)

            if self.image_type == 'RGBD':
                record_image = np.array(image[:, :, :3] * 255, dtype=np.uint8)
            else:
                record_image = np.array(image * 255, dtype=np.uint8)
            self.record_imgs.append(record_image)

            if self.visualization:
                plt.imshow(image)
                plt.pause(0.001)

            # reward calculation
            # propel constraint only for Force and Velocity mode
            # if self.actuator_type == 'Force' or self.actuator_type == 'Velocity':
            #     if (action == 0).any():
            #         single_step_reward += 1
            #     else:
            #         single_step_reward -= 1

            # normal reward: whether the target is in the field of view
            self.target_pos_in_camera = np.array(sim.simGetObjectPosition(self.target_handle, self.camera_handle))
            in_view = self._is_target_in_camera_view(self.target_pos_in_camera)
            if in_view:
                single_step_reward += 1
            else:
                single_step_reward -= self.outview_penalty
                # print('=> target is out of view!')

            # distance constraint
            distance_res = np.linalg.norm(self.expected_pos - self.target_pos_in_camera, ord=2)
            single_step_reward -= distance_res * self.dis_penalty_f
            # print('=> distance to expected position ({}): {} m'.format(self.expected_pos, distance_res))

            multi_step_reward += single_step_reward

            if not in_view or distance_res > 15:
                self.done_count += 1
            else:
                self.done_count = 0

        if self.done_count > self.wait_done_steps:
            is_done = True
        else:
            is_done = False

        state_dict = {
            'image': np.array(stack_image),
            'position': np.array(stack_position, dtype=np.float32),
            'orientation': np.array(stack_orient, dtype=np.float32),
            'pose': np.array(stack_pose, dtype=np.float32)
        }

        return state_dict, multi_step_reward / self.history_len, is_done, {}

    def get_target_bbox2d(self, depth_image):
        w, h = depth_image.shape
        bbox2d = [0, 0, w, h]
        refine_bbox2d = self.bbox2d_calc(bbox2d, depth_image.astype(np.uint8))
        return refine_bbox2d

    def save_records(self):
        if self.record and len(self.record_imgs) != 0:
            # print('=> saving action to {} ...'.format(self.record_action_path))
            np.savetxt(self.record_action_path, np.array(self.record_actions), delimiter=',', fmt='%.3f')
            # print('=> saving trajectory to {} ...'.format(self.record_trajectory_path))
            np.savetxt(self.record_trajectory_path, np.array(self.record_trajectory), delimiter=',', fmt='%.3f')
            # print('=> saving video to {} ...'.format(self.record_video_path))
            imageio.mimsave(self.record_video_path, self.record_imgs, 'GIF', duration=self.delta_time)

    @staticmethod
    def euler_vector_2_rotation_matrix(euler_vector):
        # euler vector format: [alpha, beta, gamma]
        alpha, beta, gamma = euler_vector
        r_x = np.array([[1, 0, 0],
                        [0, np.cos(alpha), -np.sin(alpha)],
                        [0, np.sin(alpha), np.cos(alpha)]])
        r_y = np.array([[np.cos(beta), 0, np.sin(beta)],
                        [0, 1, 0],
                        [-np.sin(beta), 0, np.cos(beta)]])
        r_z = np.array([[np.cos(gamma), -np.sin(gamma), 0],
                        [np.sin(gamma), np.cos(gamma), 0],
                        [0, 0, 1]])
        rotation_matrix = np.matmul(r_x, np.matmul(r_y, r_z))
        return rotation_matrix

    @staticmethod
    def bbox2d_calc(bbox2d, depth_img):
        x, y, w, h = bbox2d
        obj_depth_img = depth_img[y:y + h, x:x + w]
        threshold, binary_depth_img = cv2.threshold(obj_depth_img, 0.5, 255, cv2.THRESH_BINARY)

        # plt.imshow(binary_depth_img)
        # plt.show()

        # horizontal scanning
        horizontal_stack = np.sum(binary_depth_img, axis=1)
        top_bound = 0
        bottom_bound = 0
        if horizontal_stack[0] > 1:
            top_bound = 0
        if horizontal_stack[h - 1] > 1:
            bottom_bound = h
        for i in range(0, h - 1):
            if horizontal_stack[i] < 1 <= horizontal_stack[i + 1]:
                top_bound = i
            if horizontal_stack[i] >= 1 > horizontal_stack[i + 1]:
                bottom_bound = i
        # print('=> top bound: {} \t bottom bound: {}'.format(top_bound, bottom_bound))
        height = bottom_bound - top_bound
        if height < 0:
            print('=> target horizontally locate error!')
            height = 0

        # vertical scanning
        vertical_stack = np.sum(binary_depth_img, axis=0)
        left_bound = 0
        right_bound = 0
        if vertical_stack[0] > 1:
            left_bound = 0
        if vertical_stack[w - 1] > 1:
            right_bound = w
        for i in range(0, w - 1):
            if vertical_stack[i] < 1 <= vertical_stack[i + 1]:
                left_bound = i
            if vertical_stack[i] >= 1 > vertical_stack[i + 1]:
                right_bound = i
        # print('=> left bound: {} \t right bound: {}'.format(left_bound, right_bound))
        width = right_bound - left_bound
        if width < 0:
            print('=> target vertically locate error!')
            width = 0

        return [x + left_bound, y + top_bound, width, height]

    def _get_o3d_camera_intrinsic_matrix(self):
        # get left camera matrix
        cam_pers_angle = sim.simGetObjectFloatParameter(self.camera_handle,
                                                        sim.sim_visionfloatparam_perspective_angle)
        # print('=> left camera perspective angle: {}'.format(math.degrees(cam_pers_angle)))

        ratio = self.camera_resolution[0] / self.camera_resolution[1]
        if ratio > 1:
            angle_x = cam_pers_angle
            angle_y = 2 * math.atan(math.tan(cam_pers_angle / 2) / ratio)

        else:
            angle_x = 2 * math.atan(math.tan(cam_pers_angle / 2) * ratio)
            angle_y = cam_pers_angle
        # print('agnleX: {}, angleY:{}'.format(math.degrees(angle_x), math.degrees(angle_y)))

        camera_params = o3d.camera.PinholeCameraIntrinsic()
        camera_params.set_intrinsics(self.camera_resolution[0], self.camera_resolution[1],
                                     self.camera_resolution[0] / (2 * math.tan(angle_x / 2)),
                                     self.camera_resolution[1] / (2 * math.tan(angle_y / 2)),
                                     self.camera_resolution[0] / 2, self.camera_resolution[1] / 2
                                     )
        intrinsic_matrix = np.array(
            [[self.camera_resolution[0] / (2 * math.tan(angle_x / 2)), 0, self.camera_resolution[0] / 2],
             [0, self.camera_resolution[1] / (2 * math.tan(angle_y / 2)), self.camera_resolution[1] / 2],
             [0, 0, 1]])
        rt_matrix = np.hstack([np.eye(3), np.zeros([3, 1])])
        camera_matrix = np.matmul(intrinsic_matrix, rt_matrix)
        # print('=> camera matrix: {}'.format(camera_matrix))
        return camera_params, camera_matrix

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
        """
        retrieve different types of images from CoppeliaSim
        'Color': Color image in [H, W, 3]
        'Depth': Depth image in [H, W, 1]
        'RGBD':  RGBD image in [H, W, 3 + 1]
        :return:
        """
        if self.image_type == 'Color':
            image = sim.simGetVisionSensorImage(self.camera_handle, self.camera_resolution)
            return image

        elif self.image_type == 'Depth':
            depth_img = sim.simGetVisionSensorDepthBuffer(self.camera_handle, self.camera_resolution, True)
            depth_img[np.where(depth_img > self.cam_far_distance)] = 0
            depth_img = np.expand_dims(depth_img, -1)
            return depth_img

        elif self.image_type == 'RGBD':
            image = sim.simGetVisionSensorImage(self.camera_handle, self.camera_resolution)
            depth_img = sim.simGetVisionSensorDepthBuffer(self.camera_handle, self.camera_resolution, True)
            depth_img[np.where(depth_img > self.cam_far_distance)] = 0
            rgbd = np.append(image, np.expand_dims(depth_img, 2), axis=2)
            return rgbd
        else:
            raise ValueError('Unsupported image type!')

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

    def _define_observation(self, image_type):
        # note that when use this api
        # the client_id and camera handle should be got
        # that is, observation definition should behind the initialization of CoppeliaSim
        image = self._get_camera_image()

        if image_type == 'Color':
            observation_space = spaces.Box(low=0, high=1, shape=image.shape, dtype=np.float32)
        elif image_type == 'Depth':
            observation_space = spaces.Box(low=0, high=self.cam_far_distance, shape=image.shape, dtype=np.float32)
        elif image_type == 'RGBD':
            s_high = image
            s_high[:, :, -1] = self.cam_far_distance
            s_high[:, :, :-1] = 1
            s_low = np.zeros(image.shape)
            observation_space = spaces.Box(low=s_low, high=s_high, shape=image.shape, dtype=np.float32)
        else:
            raise ValueError('=> input unsupported observation type!')

        return observation_space

    @runtime
    def run(self, episode_num=30):
        """
        Run the test simulation without any learning algorithm for debugging purposes
        """
        try:
            for t in range(episode_num):
                rewards = []
                if t % 1 == 0:
                    state_dict = self.reset(True)
                else:
                    state_dict = self.reset(False)
                start_time = time.time()
                while True:
                    state_dict, reward, done, _ = self.env_step(DISCRETE_ACTIONS[self.action_space.sample()])
                    # state, reward, done, _ = self.env_step(DISCRETE_ACTIONS[0])
                    # print(reward)
                    rewards.append(reward)
                    if done or self.step_count > 1000:
                        print('=> episode: {}, episode length: {}, episode rewards: {}'.format(
                            t + 1, self.step_count, np.sum(rewards)
                        ))
                        self.save_records()
                        break
                end_time = time.time()
                elapsed_time = end_time - start_time
                print('=> speed for one episode with {} timesteps: {:0.2f} Hz'.format(
                    self.step_count, self.step_count / elapsed_time))
            self.save_records()
        except KeyboardInterrupt:
            pass


def run(thread_num):
    # np.random.seed(thread_num)
    scenes_paths = sorted(glob.glob(os.path.join(TEST_SCENES_DIR, '*.ttt')))
    if len(scenes_paths) > 0:
        print('=> {} scenes file found in {}'.format(len(scenes_paths), TEST_SCENES_DIR))
    else:
        raise ValueError('No scenes file found!')

    # for i in range(5):
    #     thread_num = i
    scene_path = scenes_paths[thread_num % len(scenes_paths)]
    print('=> reloading scene in {} ...'.format(scene_path))
    env = SNCOAT_Env_v2(name='scene_{:02d}'.format(thread_num + 1), scene_path=scene_path, log_dir='log',
                        action_type='Discrete', image_type='Color', state_type='PoseImage', actuator_type='Velocity',
                        clear_record=True, headless=True, wait_done_steps=20, history_len=4,
                        actuator_noise=False, time_delay=False, image_blur=False, blur_level=1)
    env.run(episode_num=10)
    # env.stop()
    env.shutdown()


if __name__ == '__main__':
    PROCESS_NUM = len(glob.glob(os.path.join(TEST_SCENES_DIR, '*.ttt')))
    print('=> Found {} scenes in {}'.format(PROCESS_NUM, TEST_SCENES_DIR))
    # PROCESS_NUM = 10
    # processes = [Process(target=run, args=(i,)) for i in range(PROCESS_NUM)]
    # [p.start() for p in processes]
    # [p.join() for p in processes]
    run(0)
