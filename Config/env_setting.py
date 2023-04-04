import numpy as np
import os
import sys
import math
import open3d as o3d

sys.path.append('/home/group1/dzhou/RAMAVT')

os.environ['COPPELIASIM_ROOT'] = os.environ['HOME'] + '/CoppeliaSim4.2'
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = os.environ['COPPELIASIM_ROOT']

TRAIN_SCENES_DIR = os.environ['HOME'] + '/RAMAVT/Scenes/train'
TEST_SCENES_DIR = os.environ['HOME'] + '/RAMAVT/Scenes/eval'

SCENES = [
    'SNCOAT-Asteroid-v0.ttt', 'SNCOAT-Asteroid-v1.ttt', 'SNCOAT-Asteroid-v2.ttt',
    'SNCOAT-Asteroid-v3.ttt', 'SNCOAT-Asteroid-v4.ttt', 'SNCOAT-Asteroid-v5.ttt',
    'SNCOAT-Capsule-v0.ttt', 'SNCOAT-Capsule-v1.ttt', 'SNCOAT-Capsule-v2.ttt',
    'SNCOAT-Rocket-v0.ttt', 'SNCOAT-Rocket-v1.ttt', 'SNCOAT-Rocket-v2.ttt',
    'SNCOAT-Satellite-v0.ttt', 'SNCOAT-Satellite-v1.ttt', 'SNCOAT-Satellite-v2.ttt',
    'SNCOAT-Station-v0.ttt', 'SNCOAT-Station-v1.ttt', 'SNCOAT-Station-v2.ttt',
]

# action format: force in x, y, z axis

DISCRETE_ACTIONS = np.array([
    [0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0],
    [-1, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0],
    [0, -1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0],
    [0, 0, -1, 0, 0, 0],
    [1, 1, 0, 0, 0, 0],
    [-1, 1, 0, 0, 0, 0],
    [1, -1, 0, 0, 0, 0],
    [-1, -1, 0, 0, 0, 0],
])

# DISCRETE_ACTIONS = np.array([
#     [0, 0, 0, 0, 0, 0],
#     [1, 0, 0, 0, 0, 0],
#     [-1, 0, 0, 0, 0, 0],
#     [0, 1, 0, 0, 0, 0],
#     [0, -1, 0, 0, 0, 0],
#     [0, 0, 1, 0, 0, 0],
#     [0, 0, -1, 0, 0, 0],
#     [1, 1, 0, 0, 0, 0],
#     [-1, 1, 0, 0, 0, 0],
#     [1, -1, 0, 0, 0, 0],
#     [-1, -1, 0, 0, 0, 0],
#     [0, 1, -1, 0, 0, 0],
#     [0, -1, 1, 0, 0, 0],
#     [1, 0, -1, 0, 0, 0],
#     [-1, 0, 1, 0, 0, 0],
#     [1, 1, 1, 0, 0, 0],
#     [-1, -1, -1, 0, 0, 0]
# ])

DISCRETE_ACTION_DIM = len(DISCRETE_ACTIONS)
CONTINUOUS_ACTION_DIM = 3
CAM_FAR_DISTANCE = 20
IMAGE_CHANNELS = {
    'Color': 3,
    'Depth': 1,
    'RGBD': 4
}
CAM_RESOLUTION = [256, 256]
CAM_PERS_ANG = 60

ratio = CAM_RESOLUTION[0] / CAM_RESOLUTION[1]
if ratio > 1:
    angle_x = CAM_PERS_ANG
    angle_y = 2 * math.atan(math.tan(CAM_PERS_ANG / 2) / ratio)

else:
    angle_x = 2 * math.atan(math.tan(CAM_PERS_ANG / 2) * ratio)
    angle_y = CAM_PERS_ANG

CAM_PARAMS = o3d.camera.PinholeCameraIntrinsic()
CAM_PARAMS.set_intrinsics(CAM_RESOLUTION[0], CAM_RESOLUTION[1],
                          CAM_RESOLUTION[0] / (2 * math.tan(angle_x / 2)),
                          CAM_RESOLUTION[1] / (2 * math.tan(angle_y / 2)),
                          CAM_RESOLUTION[0] / 2, CAM_RESOLUTION[1] / 2)
intrinsic_matrix = np.array(
    [[CAM_RESOLUTION[0] / (2 * math.tan(angle_x / 2)), 0, CAM_RESOLUTION[0] / 2],
     [0, CAM_RESOLUTION[1] / (2 * math.tan(angle_y / 2)), CAM_RESOLUTION[1] / 2],
     [0, 0, 1]])
rt_matrix = np.hstack([np.eye(3), np.zeros([3, 1])])
CAM_MATRIX = np.matmul(intrinsic_matrix, rt_matrix)
