import random

import numpy as np
import torch
import collections


class ReplayBuffer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.buffer = collections.deque(maxlen=self.cfg.buffer_maxLen)

    def push(self, data):
        self.buffer.append(data)

    def sample(self):
        state_list = []
        action_list = []
        reward_list = []
        next_state_list = []
        observation_list = []
        batch = random.sample(self.buffer, self.cfg.batch_size)
        for experience in batch:
            s, a, r, n_s, o = experience
            state_list.append(s)
            action_list.append(a)
            reward_list.append(r)
            next_state_list.append(n_s)
            observation_list.append(o)

        return torch.FloatTensor(np.array(state_list)).to(self.cfg.device), \
               torch.FloatTensor(np.array(action_list)).to(self.cfg.device), \
               torch.FloatTensor(np.array(reward_list)).unsqueeze(-1).to(self.cfg.device), \
               torch.FloatTensor(np.array(next_state_list)).to(self.cfg.device), \
               torch.FloatTensor(np.array(observation_list)).to(self.cfg.device)

    def buffer_len(self):
        return len(self.buffer)
