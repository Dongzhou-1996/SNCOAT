import numpy as np
import random
import time
from collections import namedtuple

Transition = namedtuple('Transition', ['image', 'vector', 'action', 'reward', 'next_image', 'next_vector', 'terminal'])


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

    def push(self, data: Transition):
        if len(self.storage) >= self.max_size:
            self.storage.pop(0)
            self.total_count -= 1
        self.storage.append(data)
        self.total_count += 1


    def sample(self, batch_size):
        samples = random.sample(self.storage, batch_size)
        imgs, vecs, actions, rewards, n_imgs, n_vecs, dones = map(np.array, zip(*samples))
        return imgs, vecs, actions, rewards, n_imgs, n_vecs, dones

    def clear(self):
        self.storage.clear()
        self.total_count = 0

    def __len__(self):
        return len(self.storage)


if __name__ == '__main__':
    episode_num = 40
    episode_len = 10
    image_shape = (255, 255, 3)
    memory = Memory(replay_memory_size=1000)
    start_time = time.time()
    for i in range(episode_num):
        print('=> {}th episode ...'.format(i + 1))
        image = np.random.randn(*image_shape)
        vec = np.random.randn(3)
        for step in range(episode_len):
            action = np.random.randint(0, 10)
            next_image = np.random.randn(*image_shape)
            next_vec = np.random.randn(3)
            reward = np.random.randn() * 10
            if step == episode_len - 1 or np.random.rand() < 0.1:
                terminal = True
            else:
                terminal = False
            data = Transition(image=image, vector=vec, action=action, reward=reward,
                              next_image=next_image, next_vector=next_vec, terminal=terminal)
            memory.push(data)

            if i > 2:
                batch_images, batch_vectors, batch_actions, batch_rewards, batch_next_images, batch_next_vectors, batch_terminals = memory.sample(16)

            if terminal:
                print('=> episode length: {}'.format(step))
                break

    end_time = time.time()
    elapsed_time = end_time - start_time
    print('=> elapsed time: {}'.format(elapsed_time))