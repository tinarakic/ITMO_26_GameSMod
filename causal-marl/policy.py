import torch
import torch.nn as nn
from torch.distributions import Normal


class Policy(nn.Module):
    """
    Shared LSTM encoder across all variables.
    Each variable has its own heads (mu + value).
    """

    def __init__(self, input_size, hidden=64):
        super().__init__()

        self.input_size = input_size
        self.hidden = hidden

        # -------- SHARED ENCODER --------
        self.lstm = nn.LSTM(input_size, hidden, batch_first=True)

        # -------- VARIABLE-SPECIFIC HEADS --------
        # NOTE: These are now per-instance (still one Policy per variable,
        # but they share structure via same LSTM weights across all vars)
        self.mu = nn.Linear(hidden, 1)
        self.value = nn.Linear(hidden, 1)

        self.log_std = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """
        x: (batch=1, time, features)
        """

        h, _ = self.lstm(x)
        h = h[:, -1]  # last timestep

        mu = self.mu(h)
        value = self.value(h)

        std = torch.exp(self.log_std)

        return mu, std, value

    def sample(self, x):
        mu, std, value = self.forward(x)

        dist = Normal(mu, std)

        action = dist.rsample()
        logp = dist.log_prob(action).sum()

        return action, logp, value
    def sample(self, x):
        mu, std, value = self.forward(x)

        dist = Normal(mu, std)
        action = dist.rsample()
        logp = dist.log_prob(action).sum()

        return action, logp, value