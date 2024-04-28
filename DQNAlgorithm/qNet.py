import torch.nn as nn
import torch.nn.functional as f


class MLP(nn.Module):
    def __init__(self, cfg):
        super(MLP, self).__init__()
        self.cfg = cfg
        self.linear1 = nn.Linear(self.cfg.DQN_state_dim, self.cfg.hide_dim)
        self.linear2 = nn.Linear(self.cfg.hide_dim, self.cfg.hide_dim)
        self.linear3 = nn.Linear(self.cfg.hide_dim, self.cfg.hide_dim)
        self.linear4 = nn.Linear(self.cfg.hide_dim, self.cfg.DQN_action_dim)

    def forward(self, state):
        x = f.relu(self.linear1(state))
        x = f.relu(self.linear2(x))
        x = f.relu(self.linear3(x))
        x = self.linear4(x)

        return x
