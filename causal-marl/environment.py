import numpy as np
import torch
from data_utils import normalize_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CausalEnv:

    def __init__(self, data, graph, lookback=10, horizon=1):
        self.raw_data = data.copy()
        self.data, self.scalers = normalize_data(
            data.select_dtypes(include=[np.number]).copy()
        )

        self.graph = graph
        self.lookback = lookback
        self.H = horizon
        self.vars = list(graph.nodes())
        self.t = lookback

    def reset(self):
        self.t = self.lookback
        return self._obs()

    def _obs(self):
        obs = {}

        for v in self.vars:
            if v not in self.data.columns:
                continue

            parents = list(self.graph.predecessors(v))
            feats = []

            for p in parents:
                if p not in self.data.columns:
                    continue

                lag = self.graph.edges[p, v]["lag"]
                start = self.t - self.lookback - lag
                end = self.t - lag

                x = self.data[p].values[start:end]

                if len(x) < self.lookback:
                    x = np.pad(x, (self.lookback - len(x), 0))

                feats.append(x)

            # Include self-history
            self_series = self.data[v].values[self.t - self.lookback:self.t]
            feats.append(self_series)

            obs[v] = torch.tensor(
                np.stack(feats, axis=-1),
                dtype=torch.float32
            ).unsqueeze(0)

        return obs

    def step(self, actions):
        rewards = {}

        # ==================================================
        # compute per-variable rewards as negative MAE
        # ==================================================
        for v in self.vars:

            if v not in self.data.columns:
                continue

            if v not in actions:
                continue

            pred = actions[v].detach().squeeze()  # (H,)
            true = self.data[v].values[self.t:self.t + self.H]

            if len(true) < self.H:
                continue

            # Convert to torch tensor (normalized values)
            true = torch.tensor(true, dtype=torch.float32)
            pred = pred.float()

            # MAE reward on normalized values
            mae = torch.mean(torch.abs(true - pred))

            # reward per agent = -MAE
            rewards[v] = -float(mae)

        self.t += 1
        done = self.t >= len(self.data) - self.H - 1

        return self._obs(), rewards, done