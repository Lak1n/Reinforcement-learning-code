import torch.nn as nn
import torch
import torch.nn.functional as f


class ActorNet(nn.Module):
    def __init__(self, cfg):
        super(ActorNet, self).__init__()
        self.cfg = cfg

        self.linear1 = nn.Linear(self.cfg.SAC_observation_dim, self.cfg.hide_dim)
        self.linear2 = nn.Linear(self.cfg.hide_dim, self.cfg.hide_dim)
        self.linear3 = nn.Linear(self.cfg.hide_dim, self.cfg.hide_dim)
        self.linear4 = nn.Linear(self.cfg.hide_dim, self.cfg.hide_dim)

        self.mean_linear = nn.Linear(self.cfg.hide_dim, self.cfg.SAC_action_dim)
        self.mean_linear.weight.data.uniform_(-self.cfg.edge, self.cfg.edge)
        self.mean_linear.bias.data.uniform_(-self.cfg.edge, self.cfg.edge)

    def forward(self, observation):
        x = f.relu(self.linear1(observation))
        x = f.relu(self.linear2(x))
        x = f.relu(self.linear3(x))
        x = f.relu(self.linear4(x))
        mean = self.mean_linear(x)
        mean = torch.tanh(mean)

        return mean
