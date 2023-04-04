import numpy as np
from collections import namedtuple

Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done'])


class PrioritizedMemory(object):
    def __init__(self, capacity, prob_alpha=0.6):
        self.prob_alpha = prob_alpha
        self.capacity = capacity
        self.buffer = []
        self.priorities = []

    def push(self, data: Transition):
        max_prio = max(self.priorities) if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append(data)
            self.priorities.append(max_prio)
        else:
            self.buffer.pop(0)
            self.priorities.pop(0)
            self.buffer.append(data)
            self.priorities.append(max_prio)

    def sample(self, batch_size, beta=0.4):

        priors = np.array(self.priorities)
        probs = priors ** self.prob_alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = [], [], [], [], []
        for i in indices:
            batch_states.append(self.buffer[i].state)
            batch_actions.append(self.buffer[i].action)
            batch_rewards.append(self.buffer[i].reward)
            batch_next_states.append(self.buffer[i].next_state)
            batch_dones.append(self.buffer[i].done)

        return np.array(batch_states), np.array(batch_actions), np.array(batch_rewards), np.array(
            batch_next_states), np.array(batch_dones), indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio

    def __len__(self):
        return len(self.buffer)


if __name__ == '__main__':
    memory = PrioritizedMemory(capacity=1000, prob_alpha=0.6)
    ## memory initialization
    for i in range(1000):
        data = Transition(state=np.random.rand(2), action=np.random.randint(0, 2),
                          reward=np.random.randint(0, 10), next_state=np.random.rand(2), done=0)
        memory.push(data)
    batch_states, batch_action_idxs, batch_rewards, batch_next_states, batch_dones, batch_indices, batch_weights = memory.sample(64, beta=0.4)
    print('=> samples is retrieved!')