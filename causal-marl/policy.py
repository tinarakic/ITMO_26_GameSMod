import torch
import torch.nn as nn

class Policy(nn.Module):

    def __init__(self, input_size, hidden=64):
        super().__init__()

        self.input_size = input_size

        self.lstm = nn.LSTM(input_size, hidden, batch_first=True)
        self.mu = nn.Linear(hidden, 1)
        self.log_std = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        h, _ = self.lstm(x)
        h = h[:, -1]

        mu = self.mu(h)
        std = torch.exp(self.log_std)

        return mu, std

    def sample(self, x):
        mu, std = self.forward(x)

        action = mu + std * torch.randn_like(mu)
        logp = -((action - mu) ** 2 / (2 * std ** 2 + 1e-8)).sum()

        return action, logp