import torch
from collections import deque
import time
import copy
import numpy as np

from environment import CausalEnv
from policy import Policy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


# ----------------------------
# GAE helper
# ----------------------------
def compute_gae(rewards, values, gamma=0.99, lam=0.95):
    advantages = []
    gae = 0.0
    values = values + [0.0]

    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)

    returns = [a + v for a, v in zip(advantages, values[:-1])]
    return advantages, returns


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

    policies = {}
    optimizers = {}

    for v in env.vars:
        policies[v] = Policy(env.reset()[v].shape[-1], num_agents=1).to(device)
        optimizers[v] = torch.optim.Adam(policies[v].parameters(), lr=1e-3)

    reward_window = deque(maxlen=100)
    start_time = time.time()

    # ================================
    # METRIC TRACKERS
    # ================================
    reward_per_epoch = []
    reward_per_agent = {v: [] for v in env.vars}
    loss_per_epoch = []
    loss_per_agent = {v: [] for v in env.vars}
    mse_per_epoch = []
    causal_scores = []
    forecast_error = []

    best_epoch_reward = -np.inf
    best_epoch_idx = -1
    best_policies = None
    best_epoch = -1

    total_steps = len(data) * episodes
    global_step = 0

    # optional value normalization stability
    value_baseline = {v: 0.0 for v in env.vars}

    for ep in range(episodes):
        obs = env.reset()

        trajectories = {v: {"logps": [], "values": [], "rewards": []} for v in env.vars}

        ep_reward_sum = 0.0
        ep_start_time = time.time()
        step_counter = 0
        done = False

        while not done:
            step_counter += 1
            global_step += 1

            actions = {}
            logps = {}
            values = {}

            # -------- ACT --------
            for v in env.vars:
                if v not in obs:
                    continue
                x = obs[v].float().to(device)
                if x.ndim == 2:
                    x = x.unsqueeze(0)
                a, lp, val = policies[v].sample(x)
                actions[v] = a
                logps[v] = lp
                values[v] = val.squeeze()

            next_obs, reward, done = env.step(actions)
            reward = float(reward)

            # 🔥 stabilize reward
            reward = reward / (1.0 + abs(reward))
            reward_window.append(reward)
            ep_reward_sum += reward

            # -------- store trajectory --------
            for v in env.vars:
                if v not in logps:
                    continue
                trajectories[v]["logps"].append(logps[v])
                trajectories[v]["values"].append(values[v].detach().cpu())
                trajectories[v]["rewards"].append(reward / len(env.vars))

            obs = next_obs

            # -------- logging --------
            if step_counter % log_every == 0:
                avg100 = np.mean(reward_window)
                episode_elapsed = time.time() - ep_start_time
                episode_steps_done = step_counter
                episode_steps_left = len(data) - step_counter + 1e-8
                episode_eta = episode_elapsed / episode_steps_done * episode_steps_left
                total_elapsed = time.time() - start_time
                global_steps_done = global_step
                global_steps_left = total_steps - global_steps_done + 1e-8
                global_eta = total_elapsed / global_steps_done * global_steps_left

                print(
                    f"[Episode {ep+1}/{episodes}] Step {step_counter} | "
                    f"Reward: {reward:.4f} | Avg(100): {avg100:.4f} | "
                    f"Episode ETA: {episode_eta/60:.1f} min | "
                    f"Global ETA: {global_eta/60:.1f} min"
                )

        # ======================================================
        # EPISODE UPDATE (STABLE LEARNING)
        # ======================================================
        ep_losses = {v: [] for v in env.vars}

        for v in env.vars:
            logps_t = torch.stack(trajectories[v]["logps"])
            values_t = torch.tensor(trajectories[v]["values"], device=device)
            rewards_t = trajectories[v]["rewards"]

            advantages, returns = compute_gae(rewards_t, values_t.tolist(), gamma=gamma)
            advantages = torch.tensor(advantages, device=device, dtype=torch.float32)
            returns = torch.tensor(returns, device=device, dtype=torch.float32)

            # normalize advantage
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # ---------------- ACTOR ----------------
            actor_loss = -(logps_t * advantages.detach()).mean()
            optimizers[v].zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(policies[v].parameters(), 1.0)
            optimizers[v].step()

            # ---------------- CRITIC ----------------
            critic_values = values_t
            critic_loss = (critic_values - returns).pow(2).mean()
            optimizers[v].zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(policies[v].parameters(), 1.0)
            optimizers[v].step()

            ep_losses[v].append(float((actor_loss + critic_loss).detach().cpu()))

        # ======================================================
        # EPISODE SUMMARY
        # ======================================================
        ep_avg_reward = ep_reward_sum / max(step_counter, 1)
        reward_per_epoch.append(ep_avg_reward)
        causal_scores.append(compute_causal_importance(env, policies))
        best_epoch_reward = max(best_epoch_reward, ep_avg_reward)

        for v in env.vars:
            reward_per_agent[v].append(np.mean(trajectories[v]["rewards"]))
            loss_per_agent[v].append(np.mean(ep_losses[v]))

        loss_per_epoch.append(np.mean([np.mean(ep_losses[v]) for v in env.vars]))

        # placeholder forecast metrics
        forecast_metrics = {v: {"mape": ep_avg_reward, "mse": np.mean([r ** 2 for r in trajectories[v]["rewards"]])} for v in env.vars}
        forecast_error.append(forecast_metrics)
        mse_per_epoch.append(np.mean([forecast_metrics[v]["mse"] for v in env.vars]))

        if ep_avg_reward >= best_epoch_reward:
            best_policies = copy.deepcopy(policies)
            best_epoch_idx = ep
            best_epoch = ep
            torch.save(
                {
                    "policies": {v: best_policies[v].state_dict() for v in env.vars},
                    "graph": env.graph,
                    "best_epoch": best_epoch_idx,
                    "reward_per_epoch": reward_per_epoch,
                    "loss_per_epoch": loss_per_epoch,
                    "causal_scores": causal_scores,
                    "forecast_error": forecast_error,
                },
                checkpoint_path,
            )
            print(f"Checkpoint saved at Episode {ep+1} with reward {ep_avg_reward:.4f}")

        print("\n" + "=" * 60)
        print(f"EPISODE {ep+1}/{episodes}")
        print(f"Reward: {ep_reward_sum:.4f}")
        print(f"Avg:    {ep_avg_reward:.4f}")
        print(f"Steps:  {step_counter}")
        print("=" * 60 + "\n")

    # ======================================================
    # FINAL RETURN
    # ======================================================
    return (
        policies,
        reward_per_epoch,
        reward_per_agent,
        loss_per_epoch,
        loss_per_agent,
        best_epoch,
        mse_per_epoch,
        causal_scores,
        forecast_error,
        env,
    )