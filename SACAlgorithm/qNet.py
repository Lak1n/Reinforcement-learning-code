import torch.nn as nn
import torch
import torch.nn.functional as f


class QNet(nn.Module):
    def __init__(self, cfg):
        super(QNet, self).__init__()
        self.cfg = cfg
        self.linear1 = nn.Linear(self.cfg.SAC_state_dim + self.cfg.SAC_action_dim, self.cfg.hide_dim)
        self.linear2 = nn.Linear(self.cfg.hide_dim, self.cfg.hide_dim)
        self.linear3 = nn.Linear(self.cfg.hide_dim, self.cfg.hide_dim)
        self.linear4 = nn.Linear(self.cfg.hide_dim, self.cfg.hide_dim)
        self.linear5 = nn.Linear(self.cfg.hide_dim, 1)

        self.linear5.weight.data.uniform_(-self.cfg.edge, self.cfg.edge)
        self.linear5.bias.data.uniform_(-self.cfg.edge, self.cfg.edge)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = f.relu(self.linear1(x))
        x = f.relu(self.linear2(x))
        x = f.relu(self.linear3(x))
        x = f.relu(self.linear4(x))
        x = self.linear5(x)

        return x
