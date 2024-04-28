import torch
import torch.nn.functional as f
import torch.optim as optim
from SACAlgorithm import buffer
from SACAlgorithm import policyNet
from SACAlgorithm import qNet
from SACAlgorithm import vNet


class SAC:
    def __init__(self, cfg):
        self.cfg = cfg

        # Temperature
        self.log_alpha = torch.zeros(1).to(self.cfg.device)
        self.alpha = self.log_alpha.exp().to(self.cfg.device)
        self.target_entropy = [-self.cfg.SAC_observation_dim]  # -|S|
        self.target_entropy = torch.FloatTensor(self.target_entropy).to(self.cfg.device)
        # initialize networks
        self.value_net = vNet.ValueNet(self.cfg).to(self.cfg.device)
        self.target_value_net = vNet.ValueNet(self.cfg).to(self.cfg.device)
        self.q1_net = qNet.QNet(self.cfg).to(self.cfg.device)
        self.q2_net = qNet.QNet(self.cfg).to(self.cfg.device)
        self.policy_net = policyNet.PolicyNet(self.cfg).to(self.cfg.device)

        # Load the target value network parameters
        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(self.cfg.tau * param + (1 - self.cfg.tau) * target_param)

        # Initialize the optimizer
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=self.cfg.value_lr)
        self.q1_optimizer = optim.Adam(self.q1_net.parameters(), lr=self.cfg.q_lr)
        self.q2_optimizer = optim.Adam(self.q2_net.parameters(), lr=self.cfg.q_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=self.cfg.policy_lr)
        self.log_alpha.requires_grad = True
        self.alpha.requires_grad = True
        self.alpha_optim = optim.Adam([self.log_alpha], lr=self.cfg.alpha_lr)
        # Initialize the buffer
        self.buffer = buffer.ReplayBuffer(self.cfg)

    def get_action(self, observation):
        action = self.policy_net.action(observation)
        return action

    def update(self):
        if self.buffer.buffer_len() < self.cfg.batch_size:
            return
        state, action, reward, next_state, observation = self.buffer.sample()
        new_action, log_prob = self.policy_net.evaluate(observation)

        # V value loss
        value = self.value_net(state)
        new_q1_value = self.q1_net(state, new_action)
        new_q2_value = self.q2_net(state, new_action)
        # print(new_q2_value.is_cuda, new_q1_value.is_cuda, log_prob.is_cuda, self.alpha.is_cuda)
        next_value = torch.min(new_q1_value, new_q2_value) - log_prob * self.alpha
        value_loss = f.mse_loss(value, next_value.detach())

        # Soft q  loss
        q1_value = self.q1_net(state, action)
        q2_value = self.q2_net(state, action)
        target_value = self.target_value_net(next_state)
        target_q_value = reward + self.cfg.gamma * target_value
        q1_value_loss = f.mse_loss(q1_value, target_q_value.detach())
        q2_value_loss = f.mse_loss(q2_value, target_q_value.detach())

        # Policy loss
        # policy_loss = (log_prob * (log_prob - log_prob_target).detach()).mean()
        policy_loss = (self.alpha * log_prob - torch.min(new_q1_value, new_q2_value)).mean()

        # alpha loss
        alpha_loss = -torch.mean(self.log_alpha * (self.target_entropy + log_prob).detach())

        # Update Policy
        # if np.random.random() <= 0.7:
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # Update v
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        # Update Soft q
        self.q1_optimizer.zero_grad()
        self.q2_optimizer.zero_grad()
        q1_value_loss.backward()
        q2_value_loss.backward()
        self.q1_optimizer.step()
        self.q2_optimizer.step()

        # Update temperature alpha
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        # alpha_loss.requires_grad_(True)
        self.alpha_optim.step()
        self.alpha = self.log_alpha.exp()

        # Update target networks
        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(self.cfg.tau * param + (1 - self.cfg.tau) * target_param)

    def save(self, path):
        torch.save(self.policy_net.state_dict(), path + 'SAC_policy.pth')

    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path + 'SAC_policy.pth'))
