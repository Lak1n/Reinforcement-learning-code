import torch.nn as nn
import torch
import torch.nn.functional as f
from torch.distributions import Normal


class PolicyNet(nn.Module):
    def __init__(self, cfg):
        super(PolicyNet, self).__init__()
        self.cfg = cfg

        self.linear1 = nn.Linear(self.cfg.SAC_observation_dim, self.cfg.hide_dim)
        self.linear2 = nn.Linear(self.cfg.hide_dim, self.cfg.hide_dim)
        self.linear3 = nn.Linear(self.cfg.hide_dim, self.cfg.hide_dim)
        self.linear4 = nn.Linear(self.cfg.hide_dim, self.cfg.hide_dim)

        self.mean_linear = nn.Linear(self.cfg.hide_dim, self.cfg.SAC_action_dim)
        self.mean_linear.weight.data.uniform_(-self.cfg.edge, self.cfg.edge)
        self.mean_linear.bias.data.uniform_(-self.cfg.edge, self.cfg.edge)

        self.log_std_linear = nn.Linear(self.cfg.hide_dim, self.cfg.SAC_action_dim)
        self.log_std_linear.weight.data.uniform_(-self.cfg.edge, self.cfg.edge)
        self.log_std_linear.bias.data.uniform_(-self.cfg.edge, self.cfg.edge)

    def forward(self, observation):
        x = f.relu(self.linear1(observation))
        x = f.relu(self.linear2(x))
        x = f.relu(self.linear3(x))
        x = f.relu(self.linear4(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.cfg.log_std_min, self.cfg.log_std_max)

        return mean, log_std

    def action(self, observation):
        observation = torch.FloatTensor(observation).to(self.cfg.device)
        mean, log_std = self.forward(observation)
        std = log_std.exp()
        normal = Normal(mean, std)

        z = normal.sample()
        action = torch.tanh(z).detach().cpu().numpy()

        return action

    # Use re-parameterization tick
    def evaluate(self, observation, epsilon=1e-6):
        mean, log_std = self.forward(observation)
        std = log_std.exp()
        normal = Normal(mean, std)
        noise = Normal(0, 1)

        z = noise.sample()
        action = torch.tanh(mean + std * z.to(self.cfg.device))
        log_prob = normal.log_prob(mean + std * z.to(self.cfg.device)) - torch.log(1 - action.pow(2) + epsilon)
        # log_prob = log_prob.detach().cpu().numpy()
        # log_prob = np.mean(np.array(log_prob, dtype=np.float32), 1).reshape(self.cfg.batch_size, 1)
        return action, log_prob
