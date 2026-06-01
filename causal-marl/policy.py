import torch
import torch.nn as nn
from torch.distributions import Normal


class Policy(nn.Module):
    """
    Shared LSTM + multi-agent + H-step forecasting head
    """

    def __init__(self, input_size, num_agents, hidden=64, horizon=1):
        super().__init__()

        self.num_agents = num_agents
        self.hidden = hidden
        self.H = horizon

        self.lstm = nn.LSTM(input_size, hidden, batch_first=True)

        # output per agent per horizon
        self.mu = nn.Linear(hidden, num_agents * self.H)
        self.value = nn.Linear(hidden, num_agents * self.H)

        self.log_std = nn.Parameter(torch.zeros(num_agents, self.H))

    def forward(self, x):
        """
        x: (1, L, F)
        """

        h, _ = self.lstm(x)
        h = h[:, -1, :]  # (1, hidden)

        mu = self.mu(h).view(1, self.num_agents, self.H)
        value = self.value(h).view(1, self.num_agents, self.H)

        std = torch.exp(self.log_std).unsqueeze(0)  # (1, N, H)

        return mu, std, value

    def sample(self, x):
        mu, std, value = self.forward(x)

        dist = Normal(mu, std)

        action = dist.rsample()   # (1, N, H)
        logp = dist.log_prob(action).sum()

        return action, logp, value