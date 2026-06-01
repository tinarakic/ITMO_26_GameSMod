import torch
import torch.nn as nn
from torch.distributions import Normal


class Policy(nn.Module):
    """
    Fully vectorized multi-agent policy:
    - 1 shared LSTM
    - N agent-specific heads
    - SCM-consistent observation handling
    """

    def __init__(self, input_size, num_agents, hidden=64):
        super().__init__()

        self.num_agents = num_agents
        self.hidden = hidden

        # shared temporal encoder
        self.lstm = nn.LSTM(input_size, hidden, batch_first=True)

        # agent-specific heads
        self.mu = nn.Linear(hidden, num_agents)
        self.value = nn.Linear(hidden, num_agents)

        # per-agent volatility
        self.log_std = nn.Parameter(torch.zeros(num_agents))

    def forward(self, x):
        """
        x: (1, L, N * F_max)
        """

        h, _ = self.lstm(x)
        h = h[:, -1, :]  # (1, hidden)

        mu = self.mu(h)        # (1, N)
        value = self.value(h)  # (1, N)

        std = torch.exp(self.log_std).unsqueeze(0)  # (1, N)

        return mu, std, value

    def sample(self, x):
        mu, std, value = self.forward(x)

        dist = Normal(mu, std)

        action = dist.rsample()      # (1, N)
        logp = dist.log_prob(action).sum()

        return action, logp, value