import torch
from collections import deque
import time
import copy
import numpy as np

from environment import CausalEnv
from policy import Policy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------
# CAUSAL IMPORTANCE
# ---------------------------
def compute_causal_importance(env, policies):
    """
    Estimate importance of edges through parent features effect on policy loss proxy
    """
    importance = {}

    for v in env.vars:
        if v not in env.graph.nodes:
            continue

        parents = list(env.graph.predecessors(v))
        if len(parents) == 0:
            continue

        for p in parents:
            if (p, v) not in importance:
                importance[(p, v)] = 0.0

            lag = env.graph.edges[p, v]["lag"]
            signal_strength = abs(lag) / (1.0 + len(parents))
            importance[(p, v)] += signal_strength

    return importance


# ---------------------------
# TRAIN FUNCTION
# ---------------------------
def train(data, graph, episodes=10, gamma=0.99, lookback=10, log_every=50, checkpoint_path="best_epoch.pth"):

    env = CausalEnv(data, graph, lookback=lookback)
    obs = env.reset()

    # ---------------------------
    # POLICIES AND OPTIMIZERS
    # ---------------------------
    policies, optimizers = {}, {}
    for v in env.vars:
        policies[v] = Policy(obs[v].shape[-1]).to(device)
        optimizers[v] = torch.optim.Adam(policies[v].parameters(), lr=1e-3)

    global_step = 0
    reward_window = deque(maxlen=100)
    start_time = time.time()
    total_steps = len(data) * episodes

    # ---------------------------
    # TRACKING METRICS
    # ---------------------------
    reward_per_epoch = []
    reward_per_agent = {v: [] for v in env.vars}
    loss_per_epoch = []
    loss_per_agent = {v: [] for v in env.vars}
    mse_per_epoch = []
    causal_importance_per_epoch = []
    forecast_metrics_per_epoch = {v: {"mse": [], "mape": []} for v in env.vars}

    best_epoch_reward = -np.inf
    best_epoch_idx = -1
    best_policies = None

    # ===========================
    # EPISODES LOOP
    # ===========================
    for ep in range(episodes):
        obs = env.reset()

        step_counter = 0
        ep_reward_sum = 0.0
        ep_agent_rewards = {v: 0.0 for v in env.vars}
        ep_losses = []
        ep_agent_losses = {v: [] for v in env.vars}

        ep_var_mse = {v: [] for v in env.vars}
        ep_var_mape = {v: [] for v in env.vars}

        done = False
        ep_start_time = time.time()

        # ---------------------------
        # STEP LOOP
        # ---------------------------
        while not done:
            step_counter += 1
            global_step += 1

            actions = {}
            step_logp = {}

            # ---------------------------
            # SAMPLE ACTIONS
            # ---------------------------
            for v in env.vars:
                if v not in obs:
                    continue

                obs[v] = obs[v].to(device)
                a, lp = policies[v].sample(obs[v])
                actions[v] = a
                step_logp[v] = lp  # only for immediate loss

            # ---------------------------
            # STEP ENV
            # ---------------------------
            obs, r, done = env.step(actions)

            # ---------------------------
            # REWARD TRACKING (CPU)
            # ---------------------------
            reward_window.append(float(r))
            ep_reward_sum += r
            for v in env.vars:
                ep_agent_rewards[v] += r / len(env.vars)

            # ---------------------------
            # METRICS
            # ---------------------------
            for v in env.vars:
                if v not in env.data.columns:
                    continue

                true = float(env.data[v].values[env.t])
                pred = actions[v].detach().cpu().numpy().flatten()[0]
                mse = (true - pred) ** 2
                mape = (abs(true - pred) / max(abs(true), 1e-8)) * 100.0
                ep_var_mse[v].append(mse)
                ep_var_mape[v].append(mape)

            # ---------------------------
            # STEP LOSS
            # ---------------------------
            step_loss = 0.0
            for v, lp in step_logp.items():
                loss_v = (-lp).detach().float()
                ep_agent_losses[v].append(loss_v.item())
                step_loss += loss_v.item()
            ep_losses.append(step_loss)

            # ---------------------------
            # LOGGING
            # ---------------------------
            if step_counter % log_every == 0:
                avg_100 = float(np.mean(reward_window))
                elapsed = time.time() - start_time
                steps_done = global_step
                steps_left = total_steps - steps_done + 1e-8
                eta = elapsed / steps_done * steps_left

                print(
                    f"[Episode {ep+1}/{episodes}] Step {step_counter} | "
                    f"Reward: {r:.4f} | Avg(100): {avg_100:.4f} | "
                    f"Loss: {step_loss:.4f} | ETA: {eta/60:.1f} min"
                )

            # ---------------------------
            # FREE MEMORY
            # ---------------------------
            del step_logp, actions
            torch.cuda.empty_cache()

        # ===========================
        # EPISODE SUMMARY
        # ===========================
        ep_avg_reward = ep_reward_sum / max(len(reward_window), 1)
        reward_per_epoch.append(ep_avg_reward)

        for v in env.vars:
            reward_per_agent[v].append(ep_agent_rewards[v] / max(len(reward_window), 1))
            loss_per_agent[v].append(np.mean(ep_agent_losses[v]) if len(ep_agent_losses[v]) else 0.0)

        loss_per_epoch.append(np.mean(ep_losses))

        # Forecast metrics
        for v in env.vars:
            forecast_metrics_per_epoch[v]["mse"].append(float(np.mean(ep_var_mse[v])))
            forecast_metrics_per_epoch[v]["mape"].append(float(np.mean(ep_var_mape[v])))

        mse_per_epoch.append(float(np.mean([abs(r) for r in reward_window])))
        causal_importance_per_epoch.append(compute_causal_importance(env, policies))

        # ---------------------------
        # CHECKPOINT
        # ---------------------------
        if ep_avg_reward > best_epoch_reward:
            best_epoch_reward = ep_avg_reward
            best_epoch_idx = ep

            # Save only state_dicts
            best_policies = {v: policies[v].state_dict() for v in env.vars}

            torch.save(
                {
                    "forecast_metrics_per_epoch": forecast_metrics_per_epoch,
                    "policies": best_policies,
                    "graph": env.graph,
                    "graph_edges": list(env.graph.edges),
                    "graph_nodes": list(env.graph.nodes),
                    "data": env.raw_data,
                    "reward_per_epoch": reward_per_epoch,
                    "loss_per_epoch": loss_per_epoch,
                    "mse_per_epoch": mse_per_epoch,
                    "causal_importance_per_epoch": causal_importance_per_epoch,
                    "best_epoch": best_epoch_idx,
                    "input_sizes": {v: policies[v].lstm.input_size for v in env.vars},
                },
                checkpoint_path,
            )
            print(f"Checkpoint saved (epoch {ep+1}) reward={ep_avg_reward:.4f}\n")

        # ---------------------------
        # POLICY UPDATE
        # ---------------------------
        G = 0
        returns = []
        for r in reversed(reward_window):
            G = r + gamma * G
            returns.insert(0, G)

        returns = torch.tensor(returns, dtype=torch.float32, device=device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        loss = 0.0
        for t, r_t in enumerate(returns):
            loss += ep_losses[t] * r_t  # already CPU floats

        for v in env.vars:
            optimizers[v].zero_grad()

        loss.backward()

        for v in env.vars:
            optimizers[v].step()

    # ===========================
    # RETURN
    # ===========================
    return (
        best_policies,
        reward_per_epoch,
        reward_per_agent,
        loss_per_epoch,
        loss_per_agent,
        best_epoch_idx,
        mse_per_epoch,
        causal_importance_per_epoch,
        forecast_metrics_per_epoch,
        env,
    )