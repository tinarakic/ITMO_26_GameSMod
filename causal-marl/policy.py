import torch
import torch.nn as nn
from torch.distributions import Normal


class Policy(nn.Module):
    def __init__(
        self,
        input_size,
        num_agents,
        hidden=64,
        horizon=1,
    ):
        super().__init__()

        self.num_agents = num_agents
        self.hidden = hidden
        self.H = horizon

        self.lstm = nn.LSTM(
            input_size,
            hidden,
            batch_first=True,
        )

        self.mu = nn.Linear(
            hidden,
            num_agents * horizon,
        )

        # scalar critic
        self.value = nn.Linear(
            hidden,
            1,
        )

        self.log_std = nn.Parameter(
            torch.zeros(num_agents, horizon)
        )

    def forward(self, x):

        h, _ = self.lstm(x)

        h = h[:, -1, :]

        mu = self.mu(h).view(
            -1,
            self.num_agents,
            self.H,
        )

        std = torch.exp(
            self.log_std
        ).unsqueeze(0)

        value = self.value(h).squeeze(-1)

        return mu, std, value

    def sample(self, x):

        mu, std, value = self.forward(x)

        dist = Normal(mu, std)

        action = dist.rsample()

        return action, value

    def evaluate_actions(
        self,
        x,
        actions,
    ):

        mu, std, value = self.forward(x)

        dist = Normal(mu, std)

        logp = dist.log_prob(actions).sum(
            dim=(-1, -2)
        )

        entropy = dist.entropy().sum(
            dim=(-1, -2)
        )

        return logp, entropy, value