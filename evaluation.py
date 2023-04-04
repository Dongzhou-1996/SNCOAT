import os
import glob
import shutil
import sys
import time
import numpy as np
import json
import argparse
import matplotlib
import matplotlib.pyplot as plt
from Envs.SNCOAT_Env_v2 import SNCOAT_Env_v2
from Config.env_setting import DISCRETE_ACTIONS, TEST_SCENES_DIR
import matplotlib.font_manager as fm

plt.switch_backend('agg')
plt.rcParams['pdf.fonttype'] = 42
# font_path = '/usr/share/fonts/TimesNewRoman/times.ttf'
# prop = fm.FontProperties(fname=font_path)
# plt.rcParams['font.family'] = prop.get_name()


def state_transform(state_dict, env):
    """
    Normalize depth channel of state
    :param state_dict: the state dict including Image(NxMxWxH), Position(NxMx3), Orientation(NxMx3), and Pose(NxMx6)
    :param env: the simulation env
    :return: norm_image
    """
    if env.state_type == 'Pose':
        return state_dict
    else:
        image = state_dict['image']
        if env.image_type == 'Color':
            norm_image = image
        elif env.image_type == 'Depth':
            norm_image = image / env.cam_far_distance
        elif env.image_type == 'RGBD':
            image[:, ..., -1] = image[:, ..., -1] / env.cam_far_distance
            norm_image = image
        else:
            raise ValueError('Unsupported image type!')
        norm_image = np.concatenate(np.transpose(norm_image, (0, 3, 1, 2)))
        state_dict['image'] = norm_image
        return state_dict


class AVTEval(object):
    def __init__(self, eval_dir='eval', scenes_dir='Scenes',
                 dis_penalty_f=0.1, outview_penalty=5,
                 scenes_num=5, repetition=100, wait_steps=20, is_train=False):
        super(AVTEval, self).__init__()

        self.result_dir = os.path.join(eval_dir, 'result')
        if os.path.exists(self.result_dir):
            print('=> found results directory!')
        else:
            print('=> results directory is not found! it will be created soon ...')
            os.makedirs(self.result_dir)

        self.report_dir = os.path.join(eval_dir, 'report')
        if os.path.exists(self.report_dir):
            print('=> found reporting directory!')
        else:
            print('=> reporting directory is not found! it will be created soon ...')
            os.makedirs(self.report_dir)

        self.scenes_dir = scenes_dir

        if os.path.exists(self.scenes_dir):
            scene_paths = sorted(glob.glob(os.path.join(self.scenes_dir, '*.ttt')))
            if len(scene_paths) > 0:
                print('=> {} scenes have been found in {}'.format(len(scene_paths), self.scenes_dir))
            else:
                print('=> no scenes file have been found in {}'.format(self.scenes_dir))
                sys.exit(1)
        else:
            print('=> scenes directory is not found! please check it again!')
            sys.exit(1)
        self.dis_penalty_f = dis_penalty_f
        self.outview_penalty = outview_penalty
        self.scene_paths = scene_paths[:scenes_num]
        self.repetition = repetition
        self.wait_steps = wait_steps
        self.is_train = is_train

    def eval(self, agents, action_type='Discrete', actuator_type='Force',
             eval_episode=0, blur_level=1, max_episode_len=1000,
             overwrite=False, headless=True, vis=False,
             image_blur=False, actuator_noise=False, time_delay=False):
        assert isinstance(agents, (list, tuple)), print('=> pass in agents with List() or Tuple() format')
        assert action_type in ['Discrete', 'Continuous'], print('=> wrong action type!')
        assert actuator_type in ['Force', 'Velocity', 'Position'], print('=> wrong actuator type')

        for agent in agents:
            print('=> eval {} ...'.format(agent.name))

            if not hasattr(agent, 'choose_action') or agent.action_type != action_type:
                print('=> unsupported agent with wrong action type or without choose_action() method!')
                print('=> skip to eval next agent ...')
                continue

            agent_dir = os.path.join(self.result_dir, agent.name)
            if not os.path.exists(agent_dir):
                os.makedirs(agent_dir)

            if self.is_train:
                episode_dir = os.path.join(agent_dir, 'train', 'ep_{:06d}'.format(eval_episode))
            else:
                episode_dir = os.path.join(agent_dir, 'test')

            if not os.path.exists(episode_dir):
                os.makedirs(episode_dir)

            for i, scene_path in enumerate(self.scene_paths):
                print('=> eval {} on scene {}'.format(agent.name, scene_path))
                scene_dir = os.path.join(episode_dir, 'scene_{:02d}_{}'.format(i + 1, actuator_type))
                if not os.path.exists(scene_dir):
                    os.makedirs(scene_dir)
                vis_dir = os.path.join(scene_dir, 'visualization')
                if os.path.exists(vis_dir):
                    print('=> visualization directory is existed! it will be recreated soon ...')
                    shutil.rmtree(vis_dir)
                    os.makedirs(vis_dir)
                else:
                    print('=> visualization directory is not existed! it will be created soon ...')
                    os.makedirs(vis_dir)

                episode_lens = []
                episode_lens_file = os.path.join(scene_dir, 'episode_lens.txt')
                episode_rewards = []
                episode_rewards_file = os.path.join(scene_dir, 'episode_rewards.txt')
                episode_speeds = []
                episode_speeds_file = os.path.join(scene_dir, 'episode_speeds.txt')

                if os.path.exists(episode_lens_file) and os.path.exists(episode_rewards_file) and not overwrite:
                    print('=> found recording files, skipping to next scenes')
                    continue

                # created Env instance with scene path
                env = SNCOAT_Env_v2(
                    name='scene_{:02d}'.format(i + 1), scene_path=scene_path, log_dir=episode_dir,
                    action_type=action_type, state_type=agent.state_type, image_type=agent.image_type,
                    actuator_type=actuator_type, image_blur=image_blur, blur_level=blur_level,
                    dis_penalty_f=self.dis_penalty_f, outview_penalty=self.outview_penalty,
                    clear_record=True, headless=headless, wait_done_steps=self.wait_steps,
                    actuator_noise=actuator_noise, time_delay=time_delay, history_len=agent.history_len)

                for r in range(self.repetition):
                    # print('=> eval {}/{} with agent {}'.format(r + 1, self.repetition, agent.name))
                    if r % 2 == 0:
                        record = True
                    else:
                        record = False

                    state_dict = env.reset(record)
                    # state_dict = state_transform(state_dict, env)

                    rewards = []
                    elapsed_time = 0
                    while True:
                        print('\r=> step: {}'.format(env.step_count), end='')
                        start_time = time.time()
                        action, action_prob = agent.choose_action(state_dict)
                        end_time = time.time()
                        elapsed_time += (end_time - start_time + 0.005 * agent.history_len)

                        state_dict, reward, done, _ = env.env_step(action)
                        # state_dict = state_transform(state_dict, env)

                        rewards.append(reward)

                        if record and env.step_count % 5 == 0 and vis:
                            if hasattr(agent, 'eval_net'):
                                agent.eval_net.visualization(vis_dir=vis_dir)
                        else:
                            if hasattr(agent, 'eval_net'):
                                agent.eval_net.visualization_clear()

                        if done or env.step_count > max_episode_len:
                            ep_reward = np.sum(rewards)
                            print('\n=> Episode: {}/{}, Length: {}, Rewards: {:0.3f}, Speed: {:0.3f} Hz'.format(
                                r + 1, self.repetition, env.step_count, ep_reward, env.step_count / elapsed_time))
                            env.save_records()
                            tracking_save_path = env.record_video_path[:env.record_video_path.rfind('_')] + '_tracking.gif'
                            agent.reset(is_record=record, record_path=tracking_save_path)
                            break

                    episode_rewards.append(ep_reward)
                    episode_lens.append(env.step_count)
                    episode_speeds.append(env.step_count / elapsed_time)
                    tracking_save_path = env.record_video_path[:env.record_video_path.rfind('_')] + '_tracking.gif'
                    agent.reset(is_record=record, record_path=tracking_save_path)

                # print('\n=> saving episode lens to {} ...'.format(episode_lens_file))
                np.savetxt(episode_lens_file, episode_lens, fmt='%d', delimiter=',')
                # print('=> saving episode lens to {} ...'.format(episode_rewards_file))
                np.savetxt(episode_rewards_file, episode_rewards, fmt='%.3f', delimiter=',')
                # print('=> saving episode speeds to {} ...'.format(episode_speeds_file))
                np.savetxt(episode_speeds_file, episode_speeds, fmt='%.3f', delimiter=',')

                print('=> closing env ...')
                env.shutdown()

    def eval_multi_checkpoints(self, agent, action_type='Discrete', actuator_type='Force',
                               blur_level=1, max_episode_len=1000, headless=True, image_blur=False,
                               actuator_noise=False, time_delay=False, overwrite=False):
        assert actuator_type in ['Force', 'Velocity', 'Position'], print('=> wrong actuator type!')
        assert hasattr(agent, 'reload_model'), print('Agent do not has reload_model() method!')

        result_dir = os.path.join(self.result_dir, agent.name, 'train')
        if not os.path.exists(result_dir):
            print('=> result directory is not found, it will be created soon ...')
            os.makedirs(result_dir)
        checkpoint_files = agent.get_available_model_path()

        for checkpoint_file in checkpoint_files:
            if not os.path.exists(checkpoint_file):
                raise ValueError('Checkpoint file is not found in {}!'.format(checkpoint_file))
            agent.reload_model(checkpoint_file, is_training=False)
            episode_num = agent.episode_counter
            episode_dir = os.path.join(result_dir, 'ep_{:06d}'.format(episode_num))
            if not os.path.exists(episode_dir):
                print('result directory for {}th episode is not found, it will be created soon ...'.format(episode_num))
                os.makedirs(episode_dir)

            for i, scene_path in enumerate(self.scene_paths):
                print('=> eval {} on scene {}'.format(agent.name, scene_path))
                scene_dir = os.path.join(episode_dir, 'scene_{:02d}_{}'.format(i + 1, actuator_type))
                if not os.path.exists(scene_dir):
                    os.makedirs(scene_dir)

                episode_lens = []
                episode_lens_file = os.path.join(scene_dir, 'episode_lens.txt')
                episode_rewards = []
                episode_rewards_file = os.path.join(scene_dir, 'episode_rewards.txt')
                episode_speeds = []
                episode_speeds_file = os.path.join(scene_dir, 'episode_speeds.txt')

                if os.path.exists(episode_lens_file) and os.path.exists(episode_rewards_file) and not overwrite:
                    print('=> found recording files, skipping to next scenes')
                    continue

                # created Env instance with scene path
                env = SNCOAT_Env_v2(
                    name='scene_{:02d}'.format(i + 1), scene_path=scene_path, log_dir=episode_dir,
                    action_type=action_type, state_type=agent.state_type, image_type=agent.image_type,
                    actuator_type=actuator_type, image_blur=image_blur, blur_level=blur_level,
                    dis_penalty_f=self.dis_penalty_f, outview_penalty=self.outview_penalty,
                    clear_record=True, headless=headless, wait_done_steps=self.wait_steps,
                    actuator_noise=actuator_noise, time_delay=time_delay, history_len=agent.history_len)

                for r in range(self.repetition):
                    # print('=> eval {}/{} with agent {}'.format(r + 1, self.repetition, agent.name))
                    if r % 2 == 0:
                        record = True
                    else:
                        record = False

                    state_dict = env.reset(record)
                    # state_dict = state_transform(state_dict, env)

                    rewards = []
                    elapsed_time = 0
                    while True:
                        print('\r=> step: {}'.format(env.step_count), end='')
                        start_time = time.time()

                        action, action_prob = agent.choose_action(state_dict, env)
                        state_dict, reward, done, _ = env.env_step(action)
                        # state_dict = state_transform(state_dict, env)

                        end_time = time.time()
                        elapsed_time += (end_time - start_time + 0.005 * agent.history_len)

                        rewards.append(reward)

                        # if record and env.step_count % 100 == 0:
                        #     if hasattr(agent, 'eval_net'):
                        #         agent.eval_net.visualization()
                        # else:
                        #     if hasattr(agent, 'eval_net'):
                        #         agent.eval_net.visualization_clear()

                        if done or env.step_count > max_episode_len:
                            ep_reward = np.sum(rewards)
                            print('\n=> Episode: {}/{}, Length: {}, Rewards: {:0.3f}, Speed: {:0.3f} Hz'.format(
                                r + 1, self.repetition, env.step_count, ep_reward, env.step_count / elapsed_time))
                            env.save_records()
                            agent.reset()
                            break

                    episode_rewards.append(ep_reward)
                    episode_lens.append(env.step_count)
                    episode_speeds.append(env.step_count / elapsed_time)
                    tracking_save_path = env.record_video_path[:env.record_video_path.rfind('_')] + '_tracking.gif'
                    agent.reset(is_record=record, record_path=tracking_save_path)

                # print('\n=> saving episode lens to {} ...'.format(episode_lens_file))
                np.savetxt(episode_lens_file, episode_lens, fmt='%d', delimiter=',')
                # print('=> saving episode lens to {} ...'.format(episode_rewards_file))
                np.savetxt(episode_rewards_file, episode_rewards, fmt='%.3f', delimiter=',')
                # print('=> saving episode speeds to {} ...'.format(episode_speeds_file))
                np.savetxt(episode_speeds_file, episode_speeds, fmt='%.3f', delimiter=',')

                print('=> closing env ...')
                env.shutdown()

    def test_report(self, agents, actuator_type='Force'):
        assert isinstance(agents, (list, tuple)), print('=> pass in agents with List() or Tuple() format')
        assert actuator_type in ['Force', 'Velocity', 'Position'], print('=> wrong actuator type!')

        report_dir = os.path.join(self.report_dir, 'test')
        if not os.path.exists(report_dir):
            print('=> report directory is not found, it will be created ...')
            os.makedirs(report_dir)
        performance = {}
        perf_file = os.path.join(report_dir, '{}_performance.json'.format(actuator_type))

        for agent in agents:
            performance[agent] = {}

            agent_perf = {}
            agent_report_dir = os.path.join(report_dir, agent)
            if not os.path.exists(agent_report_dir):
                print('=> report directory is not found, it will be created ...')
                os.makedirs(agent_report_dir)
            report_file = os.path.join(agent_report_dir, '{}_performance.json'.format(actuator_type))

            print('=> retrieve the eval results of {} ...'.format(agent))
            agent_dir = os.path.join(self.result_dir, agent)
            if not os.path.exists(agent_dir):
                print('=> the results directory of {} is not found!'.format(agent_dir))
                # raise ValueError('=> the results directory of {} is not found!'.format(agent_dir))
                continue

            overall_rewards = []
            overall_lens = []
            overall_speeds = []
            for i, scene_path in enumerate(self.scene_paths):
                print('=> eval {} on scene {}'.format(agent, scene_path))
                scene_name = 'scene_{:02d}_{}'.format(i + 1, actuator_type)
                scene_dir = os.path.join(agent_dir, 'test', scene_name)
                if not os.path.exists(scene_dir):
                    raise ValueError(
                        '=> the results directory on {}th scene of {} is not found!'.format(i + 1, agent_dir))
                agent_perf[scene_name] = {}

                episode_lens_file = os.path.join(scene_dir, 'episode_lens.txt')
                episode_rewards_file = os.path.join(scene_dir, 'episode_rewards.txt')
                episode_speeds_file = os.path.join(scene_dir, 'episode_speeds.txt')

                episode_lens = np.loadtxt(episode_lens_file, delimiter=',')
                episode_rewards = np.loadtxt(episode_rewards_file, delimiter=',')
                episode_speeds = np.loadtxt(episode_speeds_file, delimiter=',')

                overall_lens.append(episode_lens)
                overall_rewards.append(episode_rewards)
                overall_speeds.append(episode_speeds)

                agent_perf[scene_name].update({
                    'episode_lens': episode_lens.tolist(),
                    'episode_rewards': episode_rewards.tolist(),
                    'scene_average_len': np.mean(episode_lens),
                    'scene_average_reward': np.mean(episode_rewards),
                    'scene_average_speed': np.mean(episode_speeds)
                })

            overall_lens = np.concatenate(overall_lens)
            overall_rewards = np.concatenate(overall_rewards)
            overall_speeds = np.concatenate(overall_speeds)

            agent_perf['average_len'] = np.mean(overall_lens)
            agent_perf['average_reward'] = np.mean(overall_rewards)
            agent_perf['average_speed'] = np.mean(overall_speeds)

            performance[agent]['average_len'] = np.mean(overall_lens)
            performance[agent]['min_len'] = np.min(overall_lens)
            performance[agent]['max_len'] = np.max(overall_lens)
            performance[agent]['average_reward'] = np.mean(overall_rewards)
            performance[agent]['min_reward'] = np.min(overall_rewards)
            performance[agent]['max_reward'] = np.max(overall_rewards)
            performance[agent]['average_speed'] = np.mean(overall_speeds)

            with open(report_file, 'w') as f:
                print('=> report eval results of agent {} to {}'.format(agent, report_file))
                json.dump(agent_perf, f, indent=4)

        with open(perf_file, 'w') as f:
            print('=> report overall eval results to {}'.format(perf_file))
            json.dump(performance, f, indent=4)

    # the structure of json should also be improved
    def train_report(self, agents, actuator_type='Force', plot=True):
        assert isinstance(agents, (list, tuple)), print('=> pass in agents with List() or Tuple() format')
        assert actuator_type in ['Force', 'Velocity', 'Position'], print('=> wrong actuator type!')

        report_dir = os.path.join(self.report_dir, 'train')
        if not os.path.exists(report_dir):
            print('=> report directory is not found, it will be created ...')
            os.makedirs(report_dir)

        performance = {}
        perf_file = os.path.join(report_dir, '{}_performance.json'.format(actuator_type))

        for agent in agents:
            performance[agent] = {}
            agent_perf = {}

            agent_report_dir = os.path.join(report_dir, agent)
            if not os.path.exists(agent_report_dir):
                print('=> report directory of agent({}) is not found, it will be created ...'.format(agent))
                os.makedirs(agent_report_dir)
            report_file = os.path.join(agent_report_dir, '{}_performance.json'.format(actuator_type))

            print('=> retrieve the eval results of {} ...'.format(agent))
            agent_dir = os.path.join(self.result_dir, agent, 'train')
            if not os.path.exists(agent_dir):
                raise ValueError('=> the results directory of {} is not found!'.format(agent_dir))

            overall_rewards = []
            overall_lens = []
            overall_speeds = []
            episode_dirs = sorted(os.listdir(agent_dir))

            for ep_dir in episode_dirs:
                episode_name = ep_dir.split('/')[-1]
                episode_dir = os.path.join(agent_dir, ep_dir)
                agent_perf[episode_name] = {}
                performance[agent][episode_name] = {}

                ep_rewards = []
                ep_lens = []
                ep_speeds = []

                for i, scene_path in enumerate(self.scene_paths):
                    print('\r=> eval {} on episode {}, scene {}'.format(agent, episode_name, scene_path), end='')
                    scene_name = 'scene_{:02d}_{}'.format(i + 1, actuator_type)
                    scene_dir = os.path.join(episode_dir, scene_name)
                    if not os.path.exists(scene_dir):
                        raise ValueError(
                            '=> the results directory on {}th scene of {} is not found!'.format(i + 1, agent_dir))
                    agent_perf[episode_name][scene_name] = {}

                    scene_lens_file = os.path.join(scene_dir, 'episode_lens.txt')
                    scene_rewards_file = os.path.join(scene_dir, 'episode_rewards.txt')
                    scene_speeds_file = os.path.join(scene_dir, 'episode_speeds.txt')

                    scene_lens = np.loadtxt(scene_lens_file, delimiter=',')
                    scene_rewards = np.loadtxt(scene_rewards_file, delimiter=',')
                    scene_speeds = np.loadtxt(scene_speeds_file, delimiter=',')

                    agent_perf[episode_name][scene_name].update({
                        'scene_episode_lens': scene_lens.tolist(),
                        'scene_episode_rewards': scene_rewards.tolist(),
                        'scene_episode_speed': scene_speeds.tolist()
                    })

                    ep_lens.append(scene_lens)
                    ep_rewards.append(scene_rewards)
                    ep_speeds.append(scene_speeds)

                    overall_lens.append(scene_lens)
                    overall_rewards.append(scene_rewards)
                    overall_speeds.append(scene_speeds)

                avg_ep_len = np.mean(ep_lens)
                avg_ep_reward = np.mean(ep_rewards)
                avg_ep_speed = np.mean(ep_speeds)

                agent_perf[episode_name].update(
                    {
                        'avg_ep_len': avg_ep_len,
                        'avg_ep_reward': avg_ep_reward,
                        'avg_ep_speed': avg_ep_speed,
                    }
                )

                performance[agent][episode_name].update(
                    {
                        'avg_ep_len': avg_ep_len,
                        'min_ep_len': np.min(ep_lens),
                        'max_ep_len': np.max(ep_lens),
                        'avg_ep_reward': avg_ep_reward,
                        'min_ep_reward': np.min(ep_rewards),
                        'max_ep_reward': np.max(ep_rewards),
                        'avg_ep_speed': avg_ep_speed,
                        'min_ep_speed': np.min(ep_speeds),
                        'max_ep_speed': np.max(ep_speeds),
                    }
                )

            overall_lens = np.concatenate(overall_lens)
            overall_rewards = np.concatenate(overall_rewards)
            overall_speeds = np.concatenate(overall_speeds)

            agent_perf['avg_len'] = np.mean(overall_lens)
            agent_perf['min_len'] = np.min(overall_lens)
            agent_perf['max_len'] = np.max(overall_lens)

            agent_perf['avg_reward'] = np.mean(overall_rewards)
            agent_perf['min_reward'] = np.min(overall_rewards)
            agent_perf['max_reward'] = np.max(overall_rewards)

            agent_perf['avg_speed'] = np.mean(overall_speeds)
            agent_perf['min_speed'] = np.min(overall_speeds)
            agent_perf['max_speed'] = np.max(overall_speeds)

            performance[agent]['avg_len'] = np.mean(overall_lens)
            performance[agent]['min_len'] = np.min(overall_lens)
            performance[agent]['max_len'] = np.max(overall_lens)

            performance[agent]['avg_reward'] = np.mean(overall_rewards)
            performance[agent]['min_reward'] = np.min(overall_rewards)
            performance[agent]['max_reward'] = np.max(overall_rewards)

            performance[agent]['avg_speed'] = np.mean(overall_speeds)
            performance[agent]['min_speed'] = np.min(overall_speeds)
            performance[agent]['max_speed'] = np.max(overall_speeds)

            with open(report_file, 'w') as f:
                print('\n=> report eval results of agent {} to {}'.format(agent, report_file))
                json.dump(agent_perf, f, indent=4)

        with open(perf_file, 'w') as f:
            print('\n=> report overall evaluation results to {}'.format(perf_file))
            json.dump(performance, f, indent=4)

        if plot:
            episode_len_plot = os.path.join(report_dir, 'Episode_len.pdf')
            episode_reward_plot = os.path.join(report_dir, 'Episode_reward.pdf')

            self.draw_train_process(agents, perf_file, metric='Episode Length', save_path=episode_len_plot)
            self.draw_train_process(agents, perf_file, metric='Episode Reward', save_path=episode_reward_plot)

    @staticmethod
    def draw_train_process(agents, report_file, metric='', save_path=''):
        assert isinstance(agents, (list, tuple)), print('=> pass in agents with List() or Tuple() format')
        assert metric in ['Episode Length', 'Episode Reward'], print(
            '=> only support Episode_len or Episode_reward metrics')
        metrics = {
            'Episode Length': 'len',
            'Episode Reward': 'reward'
        }
        if os.path.exists(report_file):
            print('=> report file of agents have been found')
        else:
            raise ValueError('Report file of agents not found!')
        with open(report_file, 'r') as f:
            performance = json.load(f)

        cmap = plt.cm.get_cmap('Set2', len(agents))
        fig, ax = plt.subplots()
        lines = []
        legends = []

        for i, agent in enumerate(agents):
            print('=> plotting training process of agent({})'.format(agent))
            agent_perf = performance[agent]
            eps = [key for key in agent_perf.keys() if 'ep' in key]
            ep_nums = [int(ep.split('_')[-1]) for ep in eps]
            avg_curve = [agent_perf[ep]['avg_ep_{}'.format(metrics[metric])] for ep in eps]
            min_curve = [agent_perf[ep]['min_ep_{}'.format(metrics[metric])] for ep in eps]
            max_curve = [agent_perf[ep]['max_ep_{}'.format(metrics[metric])] for ep in eps]
            avg_line, = ax.plot(ep_nums, avg_curve, linewidth=2, color=cmap(i), alpha=1)
            ax.plot(ep_nums, min_curve, linewidth=1, color=cmap(i), alpha=0.7)
            ax.plot(ep_nums, max_curve, linewidth=1, color=cmap(i), alpha=0.7)
            ax.fill_between(ep_nums, min_curve, max_curve, color=cmap(i), alpha=0.25)

            lines.append(avg_line)
            legends.append(agent)

        ax.tick_params(labelsize=10, pad=2)
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]

        matplotlib.rcParams.update({
            'font.family': 'Times New Roman',
            'font.size': 10,

        })
        legend = ax.legend(lines, legends, loc='upper right',
                           bbox_to_anchor=(1, 1))

        matplotlib.rcParams.update({
            'font.family': 'Times New Roman',
            'font.size': 14,

        })
        ax.set(xlabel='Episodes',
               ylabel=metric)
        ax.grid(True)
        fig.tight_layout()
        print('Saving success plots to', save_path)
        fig.savefig(save_path,
                    bbox_extra_artists=(legend,),
                    bbox_inches='tight',
                    dpi=300)
        plt.close()

        return


class RandomAgent(object):
    def __init__(self, action_type='Discrete', state_type='Image', image_type='Color'):
        self.name = 'random_agent'
        self.action_type = action_type
        self.state_type = state_type
        self.image_type = image_type
        self.history_len = 1

    def mode_transfer(self, is_training=False):
        return

    def choose_action(self, state_dict: dict):
        if self.action_type == 'Discrete':
            action_idx = np.random.randint(0, len(DISCRETE_ACTIONS))
            return action_idx, DISCRETE_ACTIONS[action_idx]
        else:
            action = np.random.rand(3) * 2 - 1
            return action

    def reset(self, is_record, record_path):
        return


def draw_train_process(agents, report_file, metric='', save_path=''):
    assert isinstance(agents, (list, tuple)), print('=> pass in agents with List() or Tuple() format')
    assert metric in ['Episode Length', 'Episode Reward'], print(
        '=> only support Episode_len or Episode_reward metrics')
    metrics = {
        'Episode Length': 'len',
        'Episode Reward': 'reward'
    }
    if os.path.exists(report_file):
        print('=> report file of agents have been found')
    else:
        raise ValueError('Report file of agents not found!')
    with open(report_file, 'r') as f:
        performance = json.load(f)

    # cmap = plt.cm.get_cmap('rainbow', 10)
    cmap = ['r', 'k', 'b', 'g', 'm']
    linetypes = ['-', '--', ':']
    fig, ax = plt.subplots()
    lines = []
    legends = []

    for i, agent in enumerate(agents):
        print('=> plotting training process of agent({})'.format(agent))
        agent_perf = performance[agent]
        eps = [key for key in agent_perf.keys() if 'ep' in key]
        eps = eps[::2]
        ep_nums = [int(ep.split('_')[-1]) for ep in eps]
        avg_curve = [agent_perf[ep]['avg_ep_{}'.format(metrics[metric])] for ep in eps]
        min_curve = [agent_perf[ep]['min_ep_{}'.format(metrics[metric])] for ep in eps]
        max_curve = [agent_perf[ep]['max_ep_{}'.format(metrics[metric])] for ep in eps]
        avg_line, = ax.plot(ep_nums, avg_curve, linewidth=2, color=cmap[i // len(linetypes) % len(cmap)], alpha=0.7,
                            linestyle=linetypes[i % len(linetypes)])
        # ax.plot(ep_nums, min_curve, linewidth=1, color=cmap(i), alpha=0.7)
        # ax.plot(ep_nums, max_curve, linewidth=1, color=cmap(i), alpha=0.7)
        # ax.fill_between(ep_nums, min_curve, max_curve, color=cmap(i), alpha=0.25)

        lines.append(avg_line)
        legends.append(agent)

    ax.tick_params(labelsize=10, pad=2)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

    matplotlib.rcParams.update({
        'font.family': 'Times New Roman',
        'font.size': 10}
    )
    legend = ax.legend(lines, legends, loc='lower right',
                       bbox_to_anchor=(1, 0))

    matplotlib.rcParams.update({
        'font.family': 'Times New Roman',
        'font.size': 14
    })
    ax.set(xlabel='Episodes',
           ylabel=metric,
           xlim=(10, 400),
           )
    ax.grid(True)
    fig.tight_layout()
    print('Saving success plots to', save_path)
    fig.savefig(save_path,
                bbox_extra_artists=(legend,),
                bbox_inches='tight',
                dpi=300)
    plt.close()

    return


def multi_trains_report(result_dirs: dict, report_dir='./', actuator_type='Force', scene_num=3, plot=True):
    assert actuator_type in ['Force', 'Velocity', 'Position'], print('=> wrong actuator type!')

    performance = {}
    perf_file = os.path.join(report_dir, '{}_performance.json'.format(actuator_type))
    agents = []

    for agent, result_dir in result_dirs.items():
        agents.append(agent)
        overall_rewards = []
        overall_lens = []
        overall_speeds = []
        episode_dirs = sorted(os.listdir(result_dir))
        performance[agent] = {}

        for i, ep_dir in enumerate(episode_dirs):
            episode_name = ep_dir.split('/')[-1]
            episode_dir = os.path.join(result_dir, ep_dir)
            performance[agent][episode_name] = {}

            ep_rewards = []
            ep_lens = []
            ep_speeds = []

            for i in range(scene_num):
                print('\r=> agent: {}, episode: {} , scene: {}'.format(agent, episode_name, i + 1), end='')
                scene_name = 'scene_{:02d}_{}'.format(i + 1, actuator_type)
                scene_dir = os.path.join(episode_dir, scene_name)
                if not os.path.exists(scene_dir):
                    raise ValueError(
                        '=> the results directory on {}th scene of {} is not found!'.format(i + 1, episode_dir))

                scene_lens_file = os.path.join(scene_dir, 'episode_lens.txt')
                scene_rewards_file = os.path.join(scene_dir, 'episode_rewards.txt')
                scene_speeds_file = os.path.join(scene_dir, 'episode_speeds.txt')

                scene_lens = np.loadtxt(scene_lens_file, delimiter=',')
                scene_rewards = np.loadtxt(scene_rewards_file, delimiter=',')
                scene_speeds = np.loadtxt(scene_speeds_file, delimiter=',')

                ep_lens.append(scene_lens)
                ep_rewards.append(scene_rewards)
                ep_speeds.append(scene_speeds)

                overall_lens.append(scene_lens)
                overall_rewards.append(scene_rewards)
                overall_speeds.append(scene_speeds)

            avg_ep_len = np.mean(ep_lens)
            avg_ep_reward = np.mean(ep_rewards)
            avg_ep_speed = np.mean(ep_speeds)

            performance[agent][episode_name].update(
                {
                    'avg_ep_len': avg_ep_len,
                    'min_ep_len': np.min(ep_lens),
                    'max_ep_len': np.max(ep_lens),
                    'avg_ep_reward': avg_ep_reward,
                    'min_ep_reward': np.min(ep_rewards),
                    'max_ep_reward': np.max(ep_rewards),
                    'avg_ep_speed': avg_ep_speed,
                    'min_ep_speed': np.min(ep_speeds),
                    'max_ep_speed': np.max(ep_speeds),
                }
            )

        overall_lens = np.concatenate(overall_lens)
        overall_rewards = np.concatenate(overall_rewards)
        overall_speeds = np.concatenate(overall_speeds)

        performance[agent]['avg_len'] = np.mean(overall_lens)
        performance[agent]['min_len'] = np.min(overall_lens)
        performance[agent]['max_len'] = np.max(overall_lens)

        performance[agent]['avg_reward'] = np.mean(overall_rewards)
        performance[agent]['min_reward'] = np.min(overall_rewards)
        performance[agent]['max_reward'] = np.max(overall_rewards)

        performance[agent]['avg_speed'] = np.mean(overall_speeds)
        performance[agent]['min_speed'] = np.min(overall_speeds)
        performance[agent]['max_speed'] = np.max(overall_speeds)

    with open(perf_file, 'w') as f:
        print('\n=> report overall evaluation results to {}'.format(perf_file))
        json.dump(performance, f, indent=4)

    if plot:
        episode_len_plot = os.path.join(report_dir, 'Episode_len.pdf')
        episode_reward_plot = os.path.join(report_dir, 'Episode_reward.pdf')

        draw_train_process(agents, perf_file, metric='Episode Length', save_path=episode_len_plot)
        draw_train_process(agents, perf_file, metric='Episode Reward', save_path=episode_reward_plot)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DQN for SNCOAT')
    parser.add_argument('--gpu_idx', type=int, default=1,
                        help='gpu id')
    parser.add_argument('--scenes_num', type=int, default=6,
                        help='the num of scenes used to evaluation')
    parser.add_argument('--repetition', type=int, default=1,
                        help='the repetition times of evaluation on each scene')
    parser.add_argument('--log_dir', type=str, default='train_v3/log',
                        help='the logging directory')
    parser.add_argument('--dqn_type', type=str, default='Vanilla',
                        help='the type of dqn variants in [Vanilla, Double, Prioritized, Dueling]')
    parser.add_argument('--image_type', type=str, default='RGBD',
                        help='the type of image for env in [Color, Depth, RGBD]')
    parser.add_argument('--action_type', type=str, default='Discrete',
                        help='the type of action for env in [Discrete, Continuous]')
    parser.add_argument('--actuator_type', type=str, default='Position',
                        help='the type of actuator for chaser in [Force, Velocity, Position]')
    parser.add_argument('--saturation', type=int, default=50,
                        help='the saturation value of PID controller, \
                        50 for force control, 5 for velocity control')
    parser.add_argument('--backbone', type=str, default='ConvNet',
                        help='the type of backbone net for DQN in [ConvNet, ResNet18, ResNet34, ResNet50]')
    parser.add_argument('--max_episode_len', type=int, default=1000,
                        help='the maximum length of one episode for training')
    parser.add_argument('--headless', action='store_true',
                        help='headless mode for CoppeliaSim platform')

    args = parser.parse_args()

    from Agents.vanilla_DQN import VanillaDQN
    from Agents.DRQN import DRQN
    from Agents.DDPG import DDPG
    from PBVS import PBVS, MonoTracker

    evaluator = AVTEval(eval_dir='train',
                        scenes_dir=TEST_SCENES_DIR, scenes_num=args.scenes_num,
                        dis_penalty_f=1, outview_penalty=5,
                        repetition=args.repetition, wait_steps=20, is_train=False)

    drlavt_ddpg_depth = DDPG(
        log_dir='train_v2/log_DDPG_Image_Depth_Continuous_Velocity_attention_MHA_data_augs_Crop_Cutout_seed_0_SE',
        name='DRLAVT_DDPG_Depth', state_type='Image', image_type='Depth', action_type='Continuous',
        actuator_type='Velocity', gpu_idx=args.gpu_idx, seed=0, history_len=3, max_action=2, gamma=0.95,
        cam_far_distance=20, is_train=False, with_SE=True, headless=True, restore=True
    )

    drlavt_rgbd = VanillaDQN(
        log_dir='train/log_Vanilla_DQN_ConvNet_Image_RGBD_Velocity/SNCOAT_Env_Velocity',
        name='DRLAVT_RGBD', state_type='Image', image_type='RGBD', action_type='Discrete',
        actuator_type='Velocity', backbone='ConvNet', action_dim=len(DISCRETE_ACTIONS),
        gpu_idx=args.gpu_idx, history_len=4, restore=True, is_train=False, vis=True
    )

    drlavt_depth = VanillaDQN(
        log_dir='train/log_Vanilla_DQN_ConvNet_Image_Depth_Velocity/SNCOAT_Env_Velocity',
        name='DRLAVT_Depth', state_type='Image', image_type='Depth', action_type='Discrete',
        actuator_type='Velocity', backbone='ConvNet', action_dim=len(DISCRETE_ACTIONS),
        gpu_idx=args.gpu_idx, history_len=4, restore=True, is_train=False, vis=True
    )

    drlavt_color = VanillaDQN(
        log_dir='train/log_Vanilla_DQN_ConvNet_Image_Color_Velocity/SNCOAT_Env_Velocity',
        name='DRLAVT_Color', state_type='Image', image_type='Color', action_type='Discrete',
        actuator_type='Velocity', backbone='ConvNet', action_dim=len(DISCRETE_ACTIONS),
        gpu_idx=args.gpu_idx, history_len=4, restore=True, is_train=False, vis=True
    )

    ramavt_rgbd = DRQN(
        log_dir='train/log_Image_RGBD_Velocity_DRQN_attention_MHA_data_augs_No_SE/SNCOAT_Env_Velocity',
        model_path='train/log_Image_RGBD_Velocity_DRQN_attention_MHA_data_augs_No_SE/SNCOAT_Env_Velocity/model_ep_301.pth',
        name='RAMAVT_RGBD', state_type='Image', image_type='RGBD', action_type='Discrete', actuator_type='Velocity',
        attention_type='MHA', data_augs=['No'], vis=True,
        action_dim=len(DISCRETE_ACTIONS), gpu_idx=args.gpu_idx, restore=True, is_train=False, with_SE=True,
        lstm_dim=512, lstm_layers=2
    )
    ramavt_color = DRQN(
        log_dir='train/log_Image_Color_Velocity_DRQN_attention_MHA_data_augs_No_SE/SNCOAT_Env_Velocity',
        model_path='train/log_Image_Color_Velocity_DRQN_attention_MHA_data_augs_No_SE/SNCOAT_Env_Velocity/model_ep_241.pth',
        name='RAMAVT_Color', state_type='Image', image_type='Color', action_type='Discrete', actuator_type='Velocity',
        attention_type='MHA', data_augs=['No'], vis=True,
        action_dim=len(DISCRETE_ACTIONS), gpu_idx=args.gpu_idx, restore=True, is_train=False, with_SE=True,
        lstm_dim=512, lstm_layers=2
    )
    ramavt_depth = DRQN(
        log_dir='train/log_Image_Depth_Velocity_DRQN_attention_MHA_data_augs_No_SE/SNCOAT_Env_Velocity',
        model_path='train/log_Image_Depth_Velocity_DRQN_attention_MHA_data_augs_No_SE/SNCOAT_Env_Velocity/model_ep_241.pth',
        name='RAMAVT_Depth', state_type='Image', image_type='Depth', action_type='Discrete', actuator_type='Velocity',
        attention_type='MHA', data_augs=['No'], vis=True,
        action_dim=len(DISCRETE_ACTIONS), gpu_idx=args.gpu_idx, restore=True, is_train=False, with_SE=True,
        lstm_dim=512, lstm_layers=2
    )

    tracker = MonoTracker(mono_tracker_name='SiamRPN')
    pbvs_vel = PBVS(
        tracker, pos_ctrl_params=[-2, -0.1, -3],
        saturation_thresh=args.saturation, image_type='RGBD',
        action_type='Continuous'
    )

    rand_agent = RandomAgent(action_type='Discrete')
    
    d_agents = [
        rand_agent,
        ramavt_rgbd,
        ramavt_depth,
        ramavt_color,
        drlavt_rgbd,
        drlavt_depth,
        drlavt_color,
    ]

    c_agents = [
        pbvs_vel,
        drlavt_ddpg_depth,
    ]

    evaluator.eval(c_agents, actuator_type='Velocity', action_type='Continuous', max_episode_len=1000, blur_level=2,
                   image_blur=False, headless=True, overwrite=True, actuator_noise=False, time_delay=False, vis=False)
    evaluator.eval(d_agents, actuator_type='Velocity', action_type='Discrete', max_episode_len=1000, blur_level=2,
                   image_blur=False, headless=True, overwrite=True, actuator_noise=False, time_delay=False, vis=False)
    
    c_agent_names = [agent.name for agent in c_agents]
    d_agent_names = [agent.name for agent in d_agents]
    agent_names = c_agent_names + d_agent_names
    evaluator.test_report(agent_names, actuator_type='Velocity')

  
