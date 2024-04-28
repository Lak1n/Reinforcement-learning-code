import math
import numpy as np

import torch
import torch.nn.functional as f
import torch.optim as optim
from DDPGAlgorithm import buffer
from DDPGAlgorithm import actorNet
from DDPGAlgorithm import criticNet


class DDPG:
    def __init__(self, cfg):
        self.cfg = cfg
        self.frame_idx = 0  # 用于epsilon的衰减计数
        self.epsilon = lambda frame_idx: self.cfg.epsilon_end + \
                                         (self.cfg.epsilon_start - self.cfg.epsilon_end) * \
                                         math.exp(-1. * frame_idx / self.cfg.epsilon_decay)
        self.lr = lambda frame_idx: 8e-5 + \
                                    (self.cfg.value_lr - 8e-5) * \
                                    math.exp(-1. * frame_idx / self.cfg.lr_decay)

        # initialize networks
        self.critic_net = criticNet.CriticNet(self.cfg).to(self.cfg.device)
        self.target_critic_net = criticNet.CriticNet(self.cfg).to(self.cfg.device)
        self.actor_net = actorNet.ActorNet(self.cfg).to(self.cfg.device)
        self.target_actor_net = actorNet.ActorNet(self.cfg).to(self.cfg.device)

        # Load the target network parameters
        self.target_critic_net.load_state_dict(self.critic_net.state_dict())
        self.target_actor_net.load_state_dict(self.actor_net.state_dict())

        # Initialize the optimizer
        self.critic_optimizer = optim.Adam(self.critic_net.parameters(), lr=self.lr(self.frame_idx))
        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), lr=self.lr(self.frame_idx))

        # Initialize the buffer
        self.buffer = buffer.ReplayBuffer(self.cfg)

    def get_action(self, observation):
        self.frame_idx += 1

        if self.frame_idx < 1500 or np.random.random() > self.epsilon(self.frame_idx):
            observation = torch.FloatTensor(observation).to(self.cfg.device)
            action = self.actor_net(observation).detach().cpu().numpy()
        else:
            action = np.random.random(self.cfg.SAC_action_dim) * 2 -1
        return action

    def update(self):
        if self.buffer.buffer_len() < self.cfg.batch_size:
            return
        state, action, reward, next_state, observation, next_observation = self.buffer.sample()

        # Initialize the optimizer
        self.critic_optimizer = optim.Adam(self.critic_net.parameters(), lr=self.lr(self.frame_idx))
        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), lr=self.lr(self.frame_idx))

        # Compute the target Q value
        target_Q = self.target_critic_net(next_state, self.target_actor_net(next_observation))
        target_Q = reward + (self.cfg.gamma * target_Q).detach()

        # Get current Q estimate
        current_Q = self.critic_net(state, action)

        # Compute critic loss
        critic_loss = f.mse_loss(current_Q, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Compute actor loss
        actor_loss = -self.critic_net(state, self.actor_net(observation)).mean()

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update the frozen target models
        for param, target_param in zip(self.critic_net.parameters(), self.target_critic_net.parameters()):
            target_param.data.copy_(self.cfg.tau * param.data + (1 - self.cfg.tau) * target_param.data)

        for param, target_param in zip(self.actor_net.parameters(), self.target_actor_net.parameters()):
            target_param.data.copy_(self.cfg.tau * param.data + (1 - self.cfg.tau) * target_param.data)

    def save(self, path):
        torch.save(self.actor_net.state_dict(), path + 'DDPG_policy.pth')

    def load(self, path):
        self.actor_net.load_state_dict(torch.load(path + 'DDPG_policy.pth'))
