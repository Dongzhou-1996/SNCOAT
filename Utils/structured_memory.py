import numpy as np
import random
import Utils.data_augs as rad
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

Transition = namedtuple('Transition', ['image', 'vector', 'action', 'reward', 'terminal'])
Episode_record = namedtuple('Episode', ['images', 'vectors', 'actions', 'rewards', 'terminals', 'episode_len'])


class Memory(object):
    def __init__(self, max_episode_records=500, max_replay_experiences=10000,
                 episode_len=1000, history_len=1, aug_funcs=None):
        self.episodes = []
        self.history_len = history_len
        self.memory_size = max_episode_records
        self.replay_experience_size = max_replay_experiences
        self.episode_len = episode_len
        self.aug_funcs = aug_funcs
        self.episode_count = 0
        self.total_count = 0
        self.timestep = 0

        self.episode_images = []
        self.episode_vectors = []
        self.episode_actions = []
        self.episode_rewards = []
        self.episode_terminals = []

    def push(self, data: Transition):
        # print('\r=>{}th episode, {}th step'.format(self.episode_count, self.timestep), end="")
        self.episode_images.append(data.image)
        self.episode_vectors.append(data.vector)
        self.episode_actions.append(data.action)
        self.episode_rewards.append(data.reward)
        self.episode_terminals.append(data.terminal)
        self.timestep += 1
        self.total_count += 1

        if data.terminal:
            self.episode_save()

    def episode_save(self):
        if self.timestep > self.history_len:
            if len(self.episodes) > self.memory_size or self.total_count > self.replay_experience_size:
                self.episode_count -= 1
                self.total_count -= self.episodes[0].episode_len
                self.episodes.pop(0)

            self.episodes.append(
                Episode_record(images=np.array(self.episode_images), vectors=np.array(self.episode_vectors),
                               actions=np.array(self.episode_actions), rewards=np.array(self.episode_rewards),
                               terminals=np.array(self.episode_terminals), episode_len=self.timestep))
            self.episode_count += 1
        else:
            self.total_count -= self.timestep

        self.episode_reset()

    def episode_reset(self):
        self.timestep = 0
        self.episode_images = []
        self.episode_vectors = []
        self.episode_actions = []
        self.episode_rewards = []
        self.episode_terminals = []

    def retrieve_one_sample_from_episode(self, ep_idx):
        episode_record = self.episodes[ep_idx]
        ep_len = episode_record.episode_len
        idx = max(random.randint(0, ep_len - 1), self.history_len)
        images = np.array(episode_record.images[idx - self.history_len:idx + 1])  # MxCxWxH
        vectors = np.array(episode_record.vectors[idx - self.history_len:idx + 1])  # MxN
        for _, aug_name in enumerate(self.aug_funcs):
            if aug_name == 'SensorNoise':
                vectors = self.aug_funcs[aug_name](vectors, sigma=0.2)
            else:
                images = self.aug_funcs[aug_name](images)
        cur_images = images[:-1]
        cur_vectors = vectors[:-1]
        next_images = images[1:]
        next_vectors = vectors[:-1]
        actions = np.array(episode_record.actions[idx - self.history_len:idx])
        rewards = np.array(episode_record.rewards[idx - self.history_len:idx])
        terminals = np.array(episode_record.terminals[idx - self.history_len:idx])
        return cur_images, cur_vectors, actions, rewards, next_images, next_vectors, terminals

    def sample(self, batch_size):
        start_time = time.time()
        idxs = np.random.randint(0, len(self.episodes), batch_size)
        batch_images = []
        batch_vectors = []
        batch_actions = []
        batch_rewards = []
        batch_next_images = []
        batch_next_vectors = []
        batch_terminals = []

        thread_pool = ThreadPoolExecutor(max_workers=batch_size)
        tasks = [thread_pool.submit(self.retrieve_one_sample_from_episode, idx) for idx in idxs]

        for future in as_completed(tasks):
            image, vector, action, reward, next_image, next_vector, terminal = future.result()
            batch_images.append(image)
            batch_vectors.append(vector)
            batch_actions.append(action)
            batch_rewards.append(reward)
            batch_next_images.append(next_image)
            batch_next_vectors.append(next_vector)
            batch_terminals.append(terminal)

        # for idx in idxs:
        #     state, action, reward, next_state, terminal = self.retrieve_one_sample_from_episode(idx)
        #     batch_states.append(state)
        #     batch_actions.append(action)
        #     batch_rewards.append(reward)
        #     batch_next_states.append(next_state)
        #     batch_terminals.append(terminal)
        end_time = time.time()
        # print('=> cost of sampling one batch: {}s'.format(end_time - start_time))
        return np.array(batch_images), np.array(batch_vectors), np.array(batch_actions), np.array(batch_rewards), \
               np.array(batch_next_images), np.array(batch_next_vectors), np.array(batch_terminals)


if __name__ == '__main__':
    episode_num = 5000
    episode_len = 1000
    image_shape = (4, 255, 255)
    aug_funcs = {
        'Crop': rad.random_crop,
        # 'Cutout': rad.random_cutout,
        # 'Cutout_color': rad.random_cutout_color,
        'Flip': rad.random_flip,
        'Rotate': rad.random_rotation,
        'SensorNoise': rad.random_sensor_noise,
        'No': rad.no_aug,
    }
    memory = Memory(max_episode_records=1000, max_replay_experiences=10000,
                    episode_len=episode_len, history_len=4, aug_funcs=aug_funcs)
    start_time = time.time()
    for i in range(episode_num):
        print('=> {}th episode ...'.format(i + 1))
        image = np.random.randn(*image_shape)
        vector = np.random.randn(3)
        for step in range(episode_len):
            action = np.random.randint(0, 10)
            next_image = np.random.randn(*image_shape)
            next_vector = np.random.randn(3)
            reward = np.random.randn() * 10
            if step == episode_len - 1 or np.random.rand() < 0.01:
                terminal = True
            else:
                terminal = False
            data = Transition(image=image, vector=vector, action=action, reward=reward, terminal=terminal)
            memory.push(data)

            if i > 20:
                batch_images, batch_vectors, batch_actions, batch_rewards, batch_next_images, batch_next_vector, batch_terminals = memory.sample(32)

            if terminal:
                print('=> episode length: {} , episode_nums: {}, total_frames: {}'.format(
                    step, memory.episode_count, memory.total_count))
                break
            else:
                image = next_image
    end_time = time.time()
    elapsed_time = end_time - start_time
    print('=> elapsed time: {}'.format(elapsed_time))
