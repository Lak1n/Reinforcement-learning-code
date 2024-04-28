import math
from DQNAlgorithm.qNet import MLP
import random
import torch
import numpy as np
import torch.nn.functional as f
import torch.optim as optim
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
        # done_list = []
        batch = random.sample(self.buffer, self.cfg.batch_size)
        for experience in batch:
            s, a, r, n_s = experience
            state_list.append(s)
            action_list.append(a)
            reward_list.append(r)
            next_state_list.append(n_s)
            # done_list.append(d)
        action = torch.tensor(np.array(action_list), dtype=torch.int64).to(self.cfg.device)

        return torch.FloatTensor(np.array(state_list)).to(self.cfg.device), \
               action.unsqueeze(1), \
               torch.FloatTensor(np.array(reward_list)).unsqueeze(-1).to(self.cfg.device), \
               torch.FloatTensor(np.array(next_state_list)).to(self.cfg.device)
        # torch.FloatTensor(np.array(done_list)).unsqueeze(-1).to(self.cfg.device)

    def buffer_len(self):
        return len(self.buffer)


class DQN:
    def __init__(self, cfg):
        self.cfg = cfg
        # e-greedy策略相关参数
        self.frame_idx = 0  # 用于epsilon的衰减计数
        self.epsilon = lambda frame_idx: self.cfg.epsilon_end + \
                                         (self.cfg.epsilon_start - self.cfg.epsilon_end) * \
                                         math.exp(-1. * frame_idx / self.cfg.epsilon_decay)
        self.lr = lambda frame_idx: cfg.lr_end + \
                                    (self.cfg.lr_start - self.cfg.lr_end) * \
                                    math.exp(-1. * frame_idx / self.cfg.lr_decay)
        self.batch_size = cfg.batch_size
        self.policy_net = MLP(self.cfg).to(self.cfg.device)
        self.target_net = MLP(self.cfg).to(self.cfg.device)
        for target_param, param in zip(self.target_net.parameters(),
                                       self.policy_net.parameters()):  # 复制参数到目标网路targe_net
            target_param.data.copy_(param.data)
        self.memory = ReplayBuffer(self.cfg)  # 经验回放

    def choose_action(self, state):
        self.frame_idx += 1
        q_values = np.random.random(self.cfg.DQN_action_dim)
        if random.random() > self.epsilon(self.frame_idx):
            with torch.no_grad():
                state = torch.FloatTensor(state).to(self.cfg.device).unsqueeze(0)
                q_values = self.policy_net(state).cpu()
                # print(q_values)
                action = q_values.max(1)[1].item()  # 选择Q值最大的动作
        else:
            action = random.randrange(self.cfg.DQN_action_dim)
        return action, q_values

    def update(self):
        if self.memory.buffer_len() < self.batch_size:  # 当memory中不满足一个批量时，不更新策略
            return
        # 从经验回放中(replay memory)中随机采样一个批量的转移(transition)
        state_batch, action_batch, reward_batch, next_state_batch = self.memory.sample()
        q_values = self.policy_net(state_batch).gather(dim=1, index=action_batch)  # 计算当前状态(s_t,a)对应的Q(s_t, a)
        # print(q_values)
        # #--------- DDQN ---------  收敛数度快了一倍
        q_values_next_state = self.policy_net(next_state_batch)
        next_action = q_values_next_state.max(1)[1].unsqueeze(1)
        next_q_values = self.target_net(next_state_batch).gather(dim=1, index=next_action)
        next_q_values = next_q_values.reshape(1, self.batch_size).squeeze(0)
        # #--------- DQN -----------
        # next_q_values = self.target_net(next_state_batch).max(1)[0].detach()  # 计算下一时刻的状态(s_t_,a)对应的Q值
        # 计算期望的Q值，对于终止状态，此时done_batch[0]=1, 对应的expected_q_value等于reward
        expected_q_values = reward_batch + self.cfg.gamma * next_q_values.unsqueeze(1)
        loss = f.mse_loss(q_values, expected_q_values)  # 计算均方根损失
        # 优化更新模型
        optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr(self.frame_idx))  # 优化器
        optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():  # clip防止梯度爆炸
            param.grad.data.clamp_(-1, 1)
        optimizer.step()

    def save(self, path):
        torch.save(self.target_net.state_dict(), path + 'dqn_checkpoint.pth')

    def load(self, path):
        self.target_net.load_state_dict(torch.load(path + 'dqn_checkpoint.pth'))
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            param.data.copy_(target_param.data)
