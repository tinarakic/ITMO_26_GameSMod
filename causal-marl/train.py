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
    importance = {}

    for v in env.vars:
        if v not in env.graph.nodes:
            continue

        parents = list(env.graph.predecessors(v))
        if len(parents) == 0:
            continue

        for p in parents:
            importance.setdefault((p, v), 0.0)
            lag = env.graph.edges[p, v]["lag"]
            importance[(p, v)] += abs(lag) / (1.0 + len(parents))

    return importance


# ---------------------------
# TRAIN
# ---------------------------
def train(
    data,
    graph,
    episodes=10,
    gamma=0.99,
    lookback=10,
    log_every=50,
    checkpoint_path="best_epoch.pth",
):

    env = CausalEnv(data, graph, lookback=lookback)
    obs = env.reset()

    policies = {}
    optimizers = {}

    for v in env.vars:
        policies[v] = Policy(obs[v].shape[-1]).to(device)
        optimizers[v] = torch.optim.Adam(policies[v].parameters(), lr=1e-3)

    global_step = 0
    reward_window = deque(maxlen=100)
    start_time = time.time()
    total_steps = len(data) * episodes

    reward_per_epoch = []
    reward_per_agent = {v: [] for v in env.vars}
    loss_per_epoch = []
    loss_per_agent = {v: [] for v in env.vars}

    mse_per_epoch = []
    causal_importance_per_epoch = []
    forecast_metrics_per_epoch = {
        v: {"mse": [], "mape": []} for v in env.vars
    }

    best_epoch_reward = -np.inf
    best_epoch_idx = -1
    best_policies = None

    # =========================
    # EPISODES LOOP
    # =========================
    for ep in range(episodes):
        obs = env.reset()
        log_probs = []
        rewards = []

        step_counter = 0
        ep_reward_sum = 0.0
        ep_agent_rewards = {v: 0.0 for v in env.vars}
        ep_losses = []
        ep_agent_losses = {v: [] for v in env.vars}

        ep_var_mse = {v: [] for v in env.vars}
        ep_var_mape = {v: [] for v in env.vars}

        done = False
        ep_start_time = time.time()

        # =========================
        # STEP LOOP
        # =========================
        while not done:
            step_counter += 1
            global_step += 1

            actions = {}
            step_logp = {}

            for v in env.vars:
                if v not in obs:
                    continue
                obs[v] = obs[v].unsqueeze(0).unsqueeze(0).to(device)
                a, lp = policies[v].sample(obs[v])
                actions[v] = a
                step_logp[v] = lp  # keep tensor for gradients

            obs, r, done = env.step(actions)

            r_float = float(r)  # safe for CPU / NumPy
            rewards.append(torch.as_tensor(r_float, dtype=torch.float32, device=device))
            reward_window.append(r_float)

            # per agent reward split
            for v in env.vars:
                ep_agent_rewards[v] += r / len(env.vars)

            # track log probs for gradient
            log_probs.append({v: lp for v, lp in step_logp.items()})

            # step loss for logging (CPU)
            step_loss = 0.0
            for v, lp in step_logp.items():
                agent_loss = float((-lp).detach().cpu())
                ep_agent_losses[v].append(agent_loss)
                step_loss += agent_loss

            ep_losses.append(step_loss)

            # optional logging
            if step_counter % log_every == 0:
                avg_100 = np.mean(reward_window)

                # ----------------------------
                # Episode-level ETA
                # ----------------------------
                episode_elapsed = time.time() - ep_start_time
                episode_steps_done = step_counter
                episode_steps_left = len(data) - step_counter + 1e-8
                episode_eta = episode_elapsed / episode_steps_done * episode_steps_left

                # ----------------------------
                # Global training ETA
                # ----------------------------
                total_elapsed = time.time() - start_time
                global_steps_done = global_step
                global_steps_left = total_steps - global_steps_done + 1e-8
                global_eta = total_elapsed / global_steps_done * global_steps_left

                print(
                    f"[Episode {ep+1}/{episodes}] Step {step_counter} | "
                    f"Reward: {r:.4f} | Avg(100): {avg_100:.4f} | "
                    f"Loss: {step_loss:.4f} | "
                    f"Episode ETA: {episode_eta/60:.1f} min | "
                    f"Global ETA: {global_eta/60:.1f} min"
                )

            # memory cleanup per step
            del actions
            del step_logp
            torch.cuda.empty_cache()

        # =========================
        # EPISODE SUMMARY
        # =========================
        ep_avg_reward = ep_reward_sum / max(len(rewards), 1)
        reward_per_epoch.append(ep_avg_reward)
        for v in env.vars:
            reward_per_agent[v].append(ep_agent_rewards[v] / max(len(rewards), 1))
            if len(ep_agent_losses[v]):
                loss_per_agent[v].append(np.mean(ep_agent_losses[v]))
            else:
                loss_per_agent[v].append(0.0)
        loss_per_epoch.append(np.mean(ep_losses))

        # optional MSE / forecast metrics
        for v in env.vars:
            forecast_metrics_per_epoch[v]["mse"].append(
                float(np.mean(ep_var_mse[v])) if ep_var_mse[v] else 0.0
            )
            forecast_metrics_per_epoch[v]["mape"].append(
                float(np.mean(ep_var_mape[v])) if ep_var_mape[v] else 0.0
            )

        causal_importance_per_epoch.append(
            compute_causal_importance(env, policies)
        )

        print("\n" + "=" * 60)
        print(f"EPISODE {ep+1}/{episodes} SUMMARY")
        print(f"Total Reward: {ep_reward_sum:.4f}")
        print(f"Avg Reward:   {ep_avg_reward:.4f}")
        print(f"Avg Loss:     {np.mean(ep_losses):.4f}")
        print(f"Steps:        {len(rewards)}")
        print(f"Time:         {time.time() - ep_start_time:.2f} sec")
        print("=" * 60 + "\n")

        # =========================
        # CHECKPOINT
        # =========================
        if ep_avg_reward > best_epoch_reward:
            best_epoch_reward = ep_avg_reward
            best_epoch_idx = ep
            best_policies = copy.deepcopy(policies)

            torch.save(
                {
                    "forecast_metrics_per_epoch": forecast_metrics_per_epoch,
                    "policies": {v: best_policies[v].state_dict() for v in env.vars},
                    "graph": env.graph,
                    "graph_edges": list(env.graph.edges),
                    "graph_nodes": list(env.graph.nodes),
                    "data": env.raw_data,
                    "reward_per_epoch": reward_per_epoch,
                    "loss_per_epoch": loss_per_epoch,
                    "mse_per_epoch": mse_per_epoch,
                    "causal_importance_per_epoch": causal_importance_per_epoch,
                    "best_epoch": best_epoch_idx,
                    "input_sizes": {v: best_policies[v].lstm.input_size for v in env.vars},
                },
                checkpoint_path,
            )

            print(f"Checkpoint saved (epoch {ep+1}) reward={ep_avg_reward:.4f}\n")

        # =========================
        # POLICY UPDATE
        # =========================
        G = 0
        returns = []
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)

        returns = torch.tensor(returns, dtype=torch.float32, device=device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        loss = torch.tensor(0.0, device=device)
        for t in range(len(returns)):
            for v in env.vars:
                if v in log_probs[t]:
                    loss = loss + (-log_probs[t][v] * returns[t])
        loss = loss / max(len(returns), 1)

        for v in env.vars:
            optimizers[v].zero_grad()
        loss.backward()
        for v in env.vars:
            optimizers[v].step()

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