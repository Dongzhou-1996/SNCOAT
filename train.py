import os
import glob
import numpy as np
import argparse
from Agents.DRQN import DRQN
from Agents.vanilla_DQN import VanillaDQN
from Agents.RDDPG import RDDPG
from Agents.DDPG import DDPG
from Agents.A2C import A2C
from Agents.PPO import PPO
from evaluation import AVTEval
from Config.env_setting import TRAIN_SCENES_DIR, TEST_SCENES_DIR, \
    DISCRETE_ACTION_DIM, CONTINUOUS_ACTION_DIM, CAM_FAR_DISTANCE


os.environ['COPPELIASIM_ROOT'] = os.environ['HOME'] + '/CoppeliaSim4.2'
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = os.environ['COPPELIASIM_ROOT']


def save_arguments(arguments: argparse.Namespace, save_path='args.txt'):
    print('=> saving all the arguments to {} ...'.format(save_path))
    with open(save_path, "w") as f:
        for arg in vars(arguments):
            f.write("{}: {}\n".format(arg, getattr(arguments, arg)))


parser = argparse.ArgumentParser('SNCOAT')
parser.add_argument('--gpu_idx', type=int, default=0,
                    help='gpu id')
parser.add_argument('--seed', type=int, default=1,
                    help='the seed for training')
parser.add_argument('--log_dir', type=str, default='train_v2/log',
                    help='the logging directory')
parser.add_argument('--algo_type', type=str, default='RDDPG',
                    help='the type of deep reinforcement learning algorithm')
parser.add_argument('--attention_type', type=str, default='MHA',
                    help='the type of attention module in [No, Add, DotProd, MHA]')
parser.add_argument('--state_type', type=str, default='Image',
                    help='the type of image for env in [Image, Position, PosImage, OrientImage, Pose, PoseImage]')
parser.add_argument('--image_type', type=str, default='Depth',
                    help='the type of image for env in [Color, Depth, RGBD]')
parser.add_argument('--action_type', type=str, default='Continuous',
                    help='the type of action for env in [Discrete, Continuous]')
parser.add_argument('--actuator_type', type=str, default='Velocity',
                    help='the type of actuator for chaser in [Force, Velocity, Position]')
parser.add_argument('--data_augs', type=str, default='No',
                    help='The types of data augmentations in [Crop, Cutout, CutoutColor, Flip, Rotate, No]')
parser.add_argument('--model_path', type=str,
                    help='the path of model file')
parser.add_argument('--backbone_path', type=str,
                    help='the path of backbone file')
parser.add_argument('--global_steps', type=int, default=50000,
                    help='the step nums for total training')
parser.add_argument('--replay_buffer_size', type=int, default=10000,
                    help='the size fo replay experience buffer')
parser.add_argument('--init_buffer_size', type=int, default=100,
                    help='the size fo initial replay experience buffer')
parser.add_argument('--episode_nums', type=int, default=500,
                    help='episode nums for DQN training')
parser.add_argument('--max_episode_len', type=int, default=1000,
                    help='the maximum length of one episode for training')
parser.add_argument('--history_len', type=int, default=1,
                    help='the num of history frames to be stacked as input')
parser.add_argument('--rollout_step', type=int, default=8,
                    help='the rollout step for each network update')
parser.add_argument('--min_history', type=int, default=4,
                    help='the num of history frames to initialize the hidden state of LSTM')
parser.add_argument('--state_to_update', type=int, default=8,
                    help='the num of history frames to update QNet')
parser.add_argument('--update_interval', type=int,
                    help='the update interval of target network')
parser.add_argument('--start_epsilon', type=float, default=0.9,
                    help='the start epsilon for epsilon-greedy algorithm')
parser.add_argument('--end_epsilon', type=float, default=0.1,
                    help='the end epsilon for epsilon-greedy algorithm')
parser.add_argument('--batch_size', type=int, default=64,
                    help='the batch size')
parser.add_argument('--lstm_dim', type=int, default=512,
                    help='the dim of LSTM layers')
parser.add_argument('--lstm_layers', type=int, default=2,
                    help='the layers of LSTM modules')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='the learning rate for DQN')
parser.add_argument('--gamma', type=float, default=0.99,
                    help='the discount factor of future return')
parser.add_argument('--dis_penalty_f', type=float, default=0.5,
                    help='the factor of distance penalty')
parser.add_argument('--outview_penalty', type=int, default=5,
                    help='the reward penalty when the target is out of view')
parser.add_argument('--eval_interval', type=int, default=20,
                    help='the interval of evaluation')
parser.add_argument('--record_interval', type=int, default=10,
                    help='the interval of video recording')
parser.add_argument('--headless', action='store_true',
                    help='headless mode for CoppeliaSim platform')
parser.add_argument('--restore', action='store_true',
                    help='restore model params from file')
parser.add_argument('--restore_backbone', action='store_true',
                    help='restore backbone params from file')
parser.add_argument('--single_scene', action='store_false',
                    help='use single scene to train')
parser.add_argument('--with_SE', action='store_true',
                    help='restore model params from file')
parser.add_argument('--visualize', action='store_true',
                    help='visualize the attention map of Q-network')
args = parser.parse_args()


def main(algo_type: str):
    assert algo_type in ['REINFORCE', 'A2C', 'DDPG', 'RDDPG', 'PPO', 'TD3', 'A3C', 'SAC',
                         'Vanilla_DQN', 'Double_DQN', 'Dueling_DQN', 'Prioritized_DQN', 'DRQN'], \
        print('Unsupported reinforcement learning algorithm!')
    data_augs = args.data_augs.split('_')
    scene_paths = sorted(glob.glob(os.path.join(TRAIN_SCENES_DIR, '*.ttt')))
    log_dir = args.log_dir + '_{}_{}_{}_{}_{}_attention_{}_data_augs_{}_seed_{}'.format(
        algo_type, args.state_type, args.image_type,
        args.action_type, args.actuator_type,
        args.attention_type, args.data_augs, args.seed
    )
    if args.with_SE:
        log_dir += '_SE'
    if os.path.exists(log_dir):
        print('=> logging directory is found in {}!'.format(log_dir))
    else:
        print('=> logging directory not found in {}, it will be created soon ...'.format(log_dir))
        os.makedirs(log_dir)

    args_file = os.path.join(log_dir, 'args.txt')
    save_arguments(args, args_file)

    evaluation_dir = os.path.join(log_dir, 'eval')
    if not os.path.exists(evaluation_dir):
        print('=> evaluation directory is not found in {}, it will be created soon ...'.format(evaluation_dir))
        os.makedirs(evaluation_dir)
    if args.single_scene:
        evaluator = AVTEval(eval_dir=evaluation_dir, scenes_dir=TEST_SCENES_DIR,
                            dis_penalty_f=0.1, outview_penalty=5,
                            scenes_num=5, repetition=10, wait_steps=20, is_train=True)
    else:
        evaluator = AVTEval(eval_dir=evaluation_dir, scenes_dir=TRAIN_SCENES_DIR,
                            dis_penalty_f=0.1, outview_penalty=5,
                            scenes_num=1, repetition=10, wait_steps=20, is_train=True)

    if algo_type == 'Vanilla_DQN':
        agent = VanillaDQN(
            name=algo_type, log_dir=log_dir, seed=args.seed,
            state_type=args.state_type, image_type=args.image_type,
            action_type=args.action_type, actuator_type=args.actuator_type,
            gpu_idx=args.gpu_idx, action_dim=DISCRETE_ACTION_DIM,
            cam_far_distance=CAM_FAR_DISTANCE,
            replay_buffer_size=args.replay_buffer_size,
            init_buffer_size=args.init_buffer_size,
            history_len=args.history_len,
            gamma=args.gamma, lr=args.lr, batch_size=args.batch_size,
            start_epsilon=args.start_epsilon, end_epsilon=args.end_epsilon,
            max_ep_len=args.max_episode_len, episode_nums=args.episode_nums,
            is_train=True, restore=args.restore,
            with_SE=args.with_SE, headless=True,
        )
        agent.learn(
            evaluator, scene_paths=scene_paths, init_buffer_size=args.init_buffer_size,
            episode_nums=args.episode_nums, multi_scene=args.single_scene,
            dis_penalty_f=args.dis_penalty_f, outview_penalty=args.outview_penalty,
            eval_interval=args.eval_interval, record_interval=args.record_interval
        )
    elif algo_type == 'DRQN':
        agent = DRQN(
            name=algo_type, log_dir=log_dir, seed=args.seed,
            state_type=args.state_type, image_type=args.image_type,
            action_type=args.action_type, actuator_type=args.actuator_type,
            attention_type=args.attention_type, data_augs=data_augs,
            gpu_idx=args.gpu_idx, action_dim=DISCRETE_ACTION_DIM,
            cam_far_distance=CAM_FAR_DISTANCE,
            replay_buffer_size=args.replay_buffer_size,
            init_buffer_size=args.init_buffer_size,
            min_history=args.min_history, state_to_update=args.state_to_update,
            batch_size=args.batch_size, lr=args.lr, gamma=args.gamma,
            max_ep_len=args.max_episode_len, episode_nums=args.episode_nums,
            start_epsilon=args.start_epsilon, end_epsilon=args.end_epsilon,
            lstm_dim=args.lstm_dim, lstm_layers=args.lstm_layers, head_num=8,
            restore=args.restore, restore_backbone=args.restore_backbone,
            headless=True, is_train=True, with_SE=args.with_SE
        )
        agent.learn(
            evaluator, scene_paths=scene_paths, multi_scene=args.single_scene,
            dis_penalty_f=args.dis_penalty_f, outview_penalty=args.outview_penalty,
            eval_interval=args.eval_interval, record_interval=args.record_interval)

    elif algo_type == 'DDPG':
        agent = DDPG(
            name=algo_type, log_dir=log_dir, seed=args.seed,
            state_type=args.state_type, image_type=args.image_type,
            action_type=args.action_type, actuator_type=args.actuator_type,
            gpu_idx=args.gpu_idx, max_action=2,
            replay_buffer_size=args.replay_buffer_size,
            init_buffer_size=args.init_buffer_size,
            action_dim=CONTINUOUS_ACTION_DIM, cam_far_distance=CAM_FAR_DISTANCE,
            history_len=args.history_len, lr=args.lr,
            tau=1e-4, gamma=args.gamma, batch_size=args.batch_size,
            max_ep_len=args.max_episode_len, episode_nums=args.episode_nums,
            is_train=True,
            restore=args.restore, headless=True,
        )
        agent.learn(
            evaluator, scene_paths=scene_paths, multi_scene=args.single_scene,
            dis_penalty_f=args.dis_penalty_f, outview_penalty=args.outview_penalty,
            eval_interval=args.eval_interval, record_interval=args.record_interval)

    elif algo_type == 'RDDPG':
        agent = RDDPG(
            name=algo_type, log_dir=log_dir, seed=args.seed,
            state_type=args.state_type, image_type=args.image_type,
            action_type=args.action_type, actuator_type=args.actuator_type,
            attention_type=args.attention_type, data_augs=data_augs,
            gpu_idx=args.gpu_idx, max_action=2, lr=args.lr,
            replay_buffer_size=args.replay_buffer_size,
            init_buffer_size=args.init_buffer_size,
            action_dim=CONTINUOUS_ACTION_DIM, cam_far_distance=CAM_FAR_DISTANCE,
            min_history=args.min_history, state_to_update=args.state_to_update,
            tau=1e-4, gamma=args.gamma, batch_size=args.batch_size,
            max_ep_len=args.max_episode_len, episode_nums=args.episode_nums,
            is_train=True,
            restore=args.restore, headless=True,
        )
        agent.learn(
            evaluator, scene_paths=scene_paths, multi_scene=args.single_scene,
            init_buffer_size=args.init_buffer_size,
            dis_penalty_f=args.dis_penalty_f, outview_penalty=args.outview_penalty,
            eval_interval=args.eval_interval, record_interval=args.record_interval)

    elif algo_type == 'A2C':
        if args.action_type == 'Discrete':
            action_dim = DISCRETE_ACTION_DIM
        else:
            action_dim = CONTINUOUS_ACTION_DIM

        agent = A2C(
            name=algo_type, log_dir=log_dir, seed=args.seed,
            state_type=args.state_type, image_type=args.image_type,
            actuator_type=args.actuator_type, action_type=args.action_type,
            attention_type=args.attention_type, data_augs=data_augs,
            gpu_idx=args.gpu_idx, global_steps=args.global_steps,
            action_dim=action_dim, cam_far_distance=CAM_FAR_DISTANCE,
            history_len=args.history_len, lr=args.lr, gamma=args.gamma,
            batch_size=args.batch_size, max_ep_len=args.max_episode_len,
            lstm_dim=args.lstm_dim, lstm_layers=2,
            restore=args.restore, is_train=True,
            with_SE=args.with_SE, headless=True
        )
        agent.learn(evaluator, scene_paths=scene_paths,
                    dis_penalty_f=args.dis_penalty_f, outview_penalty=args.outview_penalty,
                    eval_interval=args.eval_interval,
                    rollout_step=args.rollout_step)
    elif algo_type == 'PPO':
        if args.action_type == 'Discrete':
            action_dim = DISCRETE_ACTION_DIM
        else:
            action_dim = CONTINUOUS_ACTION_DIM
        agent = PPO(
            name=algo_type, log_dir=log_dir, seed=args.seed,
            state_type=args.state_type, image_type=args.image_type,
            action_type=args.action_type, actuator_type=args.actuator_type,
            attention_type=args.attention_type, data_augs=data_augs,
            gpu_idx=args.gpu_idx, replay_buffer_size=args.replay_buffer_size,
            action_dim=action_dim, cam_far_distance=CAM_FAR_DISTANCE,
            history_len=args.history_len, lr=args.lr, gamma=args.gamma,
            batch_size=args.batch_size, episode_nums=args.episode_nums,
            max_ep_len=args.max_episode_len, restore=args.restore,
            is_train=True, headless=True, max_action=3,
            with_SE=args.with_SE
        )
        agent.learn(
            evaluator, scene_paths, eval_interval=args.eval_interval,
            dis_penalty_f=args.dis_penalty_f, outview_penalty=args.outview_penalty,
        )
    else:
        print('=> no agent!')


if __name__ == '__main__':
    main(args.algo_type)
