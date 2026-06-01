import torch
import torch.nn as nn

class Policy(nn.Module):
    def __init__(self, input_size, hidden=64):
        super().__init__()
        self.input_size = input_size
        self.hidden = hidden

        self.lstm = nn.LSTM(input_size, hidden, batch_first=True)
        self.mu = nn.Linear(hidden, 1)
        self.log_std = nn.Parameter(torch.zeros(1))

        # keep hidden state across steps
        self.hidden_state = None

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        # if batch is 1, seq_len=1 for step-by-step
        h, self.hidden_state = self.lstm(
            x, self.hidden_state
        )
        h = h[:, -1]
        mu = self.mu(h)
        std = torch.exp(self.log_std)
        return mu, std

    def sample(self, x):
        # detach hidden to prevent graph growth
        if self.hidden_state is not None:
            self.hidden_state = (
                self.hidden_state[0].detach(),
                self.hidden_state[1].detach(),
            )

        mu, std = self.forward(x)
        action = mu + std * torch.randn_like(mu)
        logp = -((action - mu) ** 2 / (2 * std ** 2 + 1e-8)).sum()
        return action, logp