import argparse
import os
import time
import cv2
import imageio
import numpy as np
import open3d as o3d
import torch.cuda
import evaluation as eval
from Envs.SNCOAT_Env_v2 import SNCOAT_Env_v2, DISCRETE_ACTIONS
from Config.env_setting import CAM_PARAMS, CAM_MATRIX

import matplotlib.pyplot as plt
from mono_trackers import MonoTracker


class PBVS(object):
    def __init__(self, mono_tracker, pos_ctrl_params=[0.5, 0.1, 0.1], saturation_thresh=1, history_len=1,
                 state_type='Image', image_type='RGBD', action_type='Discrete', cam_far_distance=20):
        assert state_type in ['Image', 'PosImage', 'PoseImage', 'Pos', 'Pose']
        assert image_type in ['RGBD', 'Depth'], print('=> unsupported observation type!')
        assert action_type in ['Discrete', 'Continuous']

        self.name = 'PBVS_{}'.format(mono_tracker.name)
        self.tracker = mono_tracker
        self.state_type = state_type
        self.image_type = image_type
        self.action_type = action_type
        self.history_len = history_len
        self.INIT = True
        self.cam_far_distance = cam_far_distance
        self.expected_pos = np.array([0, 0, 5])

        # PID controller
        self.saturation_thresh = saturation_thresh
        self.pos_pid = pos_ctrl_params
        self.pre_ctrl_cmd = 0
        self.err_1 = np.zeros(5)
        self.err_2 = np.zeros(5)

        self.tracking_results_2d = []
        # self.record_img_dir = os.path.join('PBVS', self.name)
        # if not os.path.exists(self.record_img_dir):
        #     print('=> recoding dir is not existed, it will be created soon ...')
        #     os.makedirs(self.record_img_dir)

    def init(self, state, init_bbox):
        """
        Init monocular tracker with initial bounding box and image state
        :param state: an image retrieved from env which may be Depth or RGB-D
        :param init_bbox: 2D bounding box with [x, y, w, h] format
        :return:
        """
        print('=> start to initial monocular tracker')
        self.tracker.init(state, init_bbox)
        self.INIT = False


    def choose_action(self, state_dict):
        """
        Choose action with Continuous/Discrete PID algorithm
        there are two control loop, one is for position control, another is angular control
        :param state_dict: image retrieved from ENV
        :return:
        """
        state = np.squeeze(state_dict['image'], axis=0)

        if self.image_type == 'Depth':
            depth = state
            image = state
            o3d_depth = o3d.geometry.Image(depth)
            o3d_pcd = o3d.geometry.PointCloud.create_from_depth_image(o3d_depth, CAM_PARAMS)
        else:
            depth = np.array(state[:, ..., -1])
            image = np.array(state[:, ..., :-1] * 255, dtype=np.uint8)

            # plt.pause(0.1)
            o3d_image = o3d.geometry.Image(image)
            o3d_depth = o3d.geometry.Image(depth)
            rgbd_img = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d_image, o3d_depth, depth_scale=1,
                                                                          depth_trunc=45)
            o3d_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_img, CAM_PARAMS)

            # if env.step_count % 10 == 0:
            #     record_time = time.time()
            #     plt.figure(1)
            #     plt.imshow(image)
            #     plt.xticks([])
            #     plt.yticks([])
            #     plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
            #     plt.savefig(os.path.join(self.record_img_dir, '{}_color.jpg'.format(record_time)),
            #                 bbox_inches='tight', dpi=300)
            #     plt.figure(2)
            #     plt.imshow(depth)
            #     plt.xticks([])
            #     plt.yticks([])
            #     plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
            #     plt.savefig(os.path.join(self.record_img_dir, '{}_depth.jpg'.format(record_time)),
            #                 bbox_inches='tight', dpi=300)
            #     flip_transform = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
            #     o3d_pcd.transform(flip_transform)
            #     # self.vis.add_geometry(o3d_pcd)
            #     # self.vis.poll_events()
            #     # self.vis.update_renderer()
            #     # self.vis.clear_geometries()
            #     o3d.visualization.draw_geometries([o3d_pcd])

        point_cloud = np.asarray(o3d_pcd.points)

        if self.INIT:
            init_bbox = self.get_target_bbox2d(depth)
            print('=> init bbox: {}'.format(init_bbox))
            _, _, w, h = init_bbox
            if w < 5 or h < 5:
                print('=> fail to init ...')
                ctrl_cmd = np.zeros(6)
                # print('=> control command: {}'.format(ctrl_cmd))
                return ctrl_cmd, 1
            self.init(image, init_bbox)
            bbox2d = init_bbox
            self.INIT = False
        else:
            bbox2d = self.tracker.update(image)

        cv2.rectangle(image, pt1=(int(bbox2d[0]), int(bbox2d[1])),
                      pt2=(int(bbox2d[0] + bbox2d[2]), int(bbox2d[1] + bbox2d[3])),
                      thickness=2, color=(255, 0, 255))

        self.tracking_results_2d.append(image)

        frustum_pcd, frustum_pcd_pixel = self.get_frustum_point_cloud(point_cloud, CAM_MATRIX, bbox2d,
                                                                      min_distance=0.5, max_distance=45)
        if frustum_pcd is not None:
            # object_point = env.target_pos_in_camera
            object_point = np.nanmean(frustum_pcd, axis=0)
            # object_point[:2] = - object_point[:2]

            # print('=> measure point: {}'.format(object_point))
            err = self.expected_pos - object_point
            # print('=> error: {}'.format(err))

            pos_ctrl_cmd = self.pos_pid[0] * (err[:3] - self.err_1[:3]) + self.pos_pid[1] * err[:3] + \
                           self.pos_pid[2] * (err[:3] - 2 * self.err_1[:3] + self.err_2[:3])

            delta_ctrl_cmd = np.concatenate([pos_ctrl_cmd, [0, 0, 0]])
            ctrl_cmd = self.pre_ctrl_cmd + delta_ctrl_cmd
            ctrl_cmd = np.clip(ctrl_cmd, -self.saturation_thresh, self.saturation_thresh)
            # print('=> control command: {}'.format(ctrl_cmd))
            self.err_2 = self.err_1
            self.err_1 = err
            self.pre_ctrl_cmd = ctrl_cmd
            return ctrl_cmd, 1
        else:
            ctrl_cmd = np.zeros(6)
            # print('=> control command: {}'.format(ctrl_cmd))
            return ctrl_cmd, 1

    def get_target_bbox2d(self, depth_image):
        w, h = depth_image.shape
        bbox2d = [0, 0, w, h]
        refine_bbox2d = self.bbox2d_calc(bbox2d, depth_image.astype(np.uint8))
        return refine_bbox2d

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

    @staticmethod
    def get_frustum_point_cloud(pcd=np.array([]), calibration_matrix=np.zeros([3, 4]),
                                roi=np.array([]), min_distance=1.0, max_distance=45.0):
        assert calibration_matrix.shape == (3, 4), 'Err: the shape of calibration matrix should be (3, 4)!'

        x, y, w, h = roi
        xmax = x + w
        xmin = x
        ymax = y + h
        ymin = y

        # project point cloud data to image plane with calibration matrix
        points_num = len(pcd)
        if points_num == 0:
            return None, None

        one = np.ones(points_num).reshape(points_num, 1)
        expand_pcd = np.concatenate((pcd, one), axis=1)

        pcd_2d = np.dot(expand_pcd, np.transpose(calibration_matrix))
        pcd_2d[:, 0] /= pcd_2d[:, 2]
        pcd_2d[:, 1] /= pcd_2d[:, 2]

        pcd_uv = pcd_2d[:, 0:2]

        # get the index of points of interest in roi region
        fov_inds = (pcd_uv[:, 0] <= xmax) & (pcd_uv[:, 0] >= xmin) & \
                   (pcd_uv[:, 1] <= ymax) & (pcd_uv[:, 1] >= ymin)

        fov_inds = fov_inds & (pcd[:, 2] > min_distance) & (pcd[:, 2] < max_distance)

        # get frustum point cloud
        frustum_pcd = pcd[fov_inds, :]
        pcd_uv = pcd_uv[fov_inds, :]

        if len(frustum_pcd) > 0:
            return frustum_pcd, pcd_uv
        else:
            return None, None

    @staticmethod
    def pos2state(object_pos: np.array([])):
        x, y, z = object_pos
        if x == 0 and y == 0 and z == 0:
            return np.zeros(5)
        else:
            alpha = np.arcsin(y / np.sqrt(x ** 2 + y ** 2 + z ** 2))
            beta = np.arcsin(x / np.sqrt(x ** 2 + z ** 2))
            return np.array([x, y, z, np.degrees(alpha), np.degrees(beta)])

    def reset(self, is_record=False, record_path=''):
        self.pre_ctrl_cmd = 0
        self.err_1 = np.zeros(5)
        self.err_2 = np.zeros(5)
        self.INIT = True

        if is_record and len(self.tracking_results_2d) > 0:
            print('=> saving video to {} ...'.format(record_path))
            imageio.mimsave(record_path, self.tracking_results_2d, 'GIF', duration=0.1)

        self.tracking_results_2d = []


parser = argparse.ArgumentParser('PBVS Baseline Algorithm')
parser.add_argument('--scene_dir', type=str, default='Scenes/eval',
                    help='The directory of scenes')
parser.add_argument('--image_type', type=str, default='RGBD',
                    help='the type of observation for env in [Color, Depth, RGBD]')
parser.add_argument('--actuator_type', type=str, default='Force',
                    help='the actuator type of controller in [Force, Velocity, Position]')
parser.add_argument('--tracker', type=str, default='SiamRPN',
                    help='the name of 2D monocular tracker')
parser.add_argument('--pid_param', type=list, default=[-1, -0.001, -13],
                    help='the params of PID controller')
parser.add_argument('--repetition', type=int, default=10,
                    help='The repetition time for evaluation')
parser.add_argument('--saturation', type=int, default=50,
                    help='the saturation value of PID controller, \
                    50 for force control, 5 for velocity control')
parser.add_argument('--max_episode_len', type=int, default=1000,
                    help='the maximum length of one episode for training')
parser.add_argument('--blur_level', type=int, default=1,
                    help='[1-4] levels of blurring')
parser.add_argument('--headless', type=bool, default=True,
                    help='headless mode')
parser.add_argument('--overwrite', type=bool, default=True,
                    help='overwrite previous evaluation results')
parser.add_argument('--actuator_noise', type=bool, default=False)
parser.add_argument('--time_delay', type=bool, default=False)
parser.add_argument('--image_blur', type=bool, default=False)
args = parser.parse_args()


if __name__ == '__main__':
    # force_pid = [-1, -0.001, -13]
    # velocity_pid = [-2, -0.1, -3]
    # position_pid = [-1, -0.5, -0.5]
    # scene_dir = os.environ['HOME'] + '/SNCOAT/Scenes/eval'

    evaluator = eval.AVTEval(scenes_dir=args.scene_dir,
                             scenes_num=5, wait_steps=20,
                             repetition=args.repetition)

    mono_trackers = [
        MonoTracker(args.tracker)
    ]

    agents = []
    for mono_tracker in mono_trackers:
        pbvs = PBVS(mono_tracker, pos_ctrl_params=args.pid_param,
                    saturation_thresh=args.saturation, image_type=args.image_type, action_type='Continuous')
        agents.append(pbvs)

    evaluator.eval(agents, action_type='Continuous', actuator_type=args.actuator_type,
                   headless=args.headless, overwrite=True, max_episode_len=args.max_episode_len,
                   actuator_noise=args.actuator_noise, time_delay=args.time_delay,
                   image_blur=args.image_blur, blur_level=args.blur_level)

    agent_names = [agent.name for agent in agents]
    evaluator.report(agent_names, actuator_type=args.actuator_type)
