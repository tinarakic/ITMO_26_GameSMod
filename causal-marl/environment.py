import numpy as np
import torch
from data_utils import normalize_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CausalEnv:

    def __init__(self, data, graph, lookback=10):

        self.raw_data = data.copy()

        # -------------------------
        # normalize + convert once
        # -------------------------
        numeric = data.select_dtypes(include=[np.number]).copy()

        self.data, self.scalers = normalize_data(numeric)

        self.data = torch.tensor(
            self.data.values,
            dtype=torch.float32,
            device=device
        )

        self.graph = graph
        self.lookback = lookback

        self.vars = list(graph.nodes())

        # only valid variables
        self.valid_vars = [
            v for v in self.vars
            if v in numeric.columns
        ]

        # map variable → column index
        self.var_to_idx = {
            v: i for i, v in enumerate(numeric.columns)
        }

        # precompute parents
        self.parents = {
            v: [
                p for p in graph.predecessors(v)
                if p in numeric.columns
            ]
            for v in self.valid_vars
        }

        self.t = lookback
        self.T = self.data.shape[0]

    # --------------------------------------------------
    def reset(self):
        self.t = self.lookback
        return self._obs()

    # --------------------------------------------------
    def _safe_window(self, start, end, idx, L):
        """
        Safe tensor slice with left padding (correct for time series)
        """

        start = max(0, start)
        end = max(0, end)

        x = self.data[start:end, idx]

        # ensure shape [L]
        if x.shape[0] < L:
            pad = torch.zeros(
                (L - x.shape[0],),
                device=self.data.device
            )
            x = torch.cat([pad, x], dim=0)

        return x

    # --------------------------------------------------
    def _obs(self):

        obs = {}

        t = self.t
        L = self.lookback

        for v in self.valid_vars:

            feats = []

            # ---------------- parents ----------------
            for p in self.parents[v]:

                lag = self.graph.edges[p, v]["lag"]

                p_idx = self.var_to_idx[p]

                start = t - L - lag
                end = t - lag

                x = self._safe_window(start, end, p_idx, L)
                feats.append(x)

            # ---------------- self ----------------
            v_idx = self.var_to_idx[v]

            self_series = self._safe_window(
                t - L,
                t,
                v_idx,
                L
            )

            feats.append(self_series)

            # shape: (L, num_features)
            obs[v] = torch.stack(feats, dim=-1).unsqueeze(0)

        return obs

    # --------------------------------------------------
    def step(self, actions):

        reward = 0.0
        t = self.t

        # ---------------- reward ----------------
        for v in self.valid_vars:

            v_idx = self.var_to_idx[v]

            true = self.data[t, v_idx]
            pred = actions[v].detach().flatten()[0]

            reward += -torch.abs(true - pred)

        reward = reward / len(self.valid_vars)

        # advance time
        self.t += 1
        done = self.t >= self.T - 1

        return self._obs(), reward, done