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

    # ======================================================
    # RESET
    # ======================================================
    def reset(self):
        self.t = self.lookback
        return self._obs()

    # ======================================================
    # OBSERVATION
    # ======================================================
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

            # include self-history
            self_series = self.data[v].values[self.t - self.lookback:self.t]
            feats.append(self_series)

            obs[v] = torch.tensor(
                np.stack(feats, axis=-1),
                dtype=torch.float32
            ).unsqueeze(0)

        return obs

    # ======================================================
    # STEP (FIXED: PER-AGENT REWARDS)
    # ======================================================
    def step(self, actions):

        eps = 1e-8

        rewards = {}

        # ==================================================
        # compute per-variable rewards
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

            true = torch.tensor(true, dtype=torch.float32)

            # normalized error
            mape = torch.mean(
                torch.abs((true - pred.cpu()) / (true + eps))
            )

            # reward per agent (negative error)
            rewards[v] = -mape.item()

        # ==================================================
        # IMPORTANT: no averaging anymore
        # ==================================================

        self.t += 1
        done = self.t >= len(self.data) - self.H - 1

        return self._obs(), rewards, done