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

    num_agents = len(env.vars)
    input_size = sum(obs[v].shape[-1] for v in env.vars)

    policy = Policy(input_size=input_size, num_agents=num_agents).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

    reward_window = deque(maxlen=100)
    start_time = time.time()

    reward_per_epoch = []
    reward_per_agent = {v: [] for v in env.vars}
    loss_per_epoch = []
    loss_per_agent = {v: [] for v in env.vars}

    mse_per_epoch = []
    causal_importance_per_epoch = []
    forecast_metrics_per_epoch = {v: {"mse": [], "mape": []} for v in env.vars}

    best_epoch_reward = -np.inf
    best_epoch_idx = -1
    best_policy = None

    total_steps = len(data) * episodes
    global_step = 0

    for ep in range(episodes):

        obs = env.reset()

        ep_reward_sum = 0.0
        ep_agent_rewards = {v: 0.0 for v in env.vars}
        ep_agent_losses = {v: [] for v in env.vars}

        ep_var_mse = {v: [] for v in env.vars}
        ep_var_mape = {v: [] for v in env.vars}

        step_counter = 0
        done = False
        ep_start_time = time.time()

        while not done:

            step_counter += 1
            global_step += 1

            # -------------------------
            # VECTORIZED OBS
            # -------------------------
            obs_tensor = torch.cat(
                [obs[v] for v in env.vars],
                dim=-1
            ).unsqueeze(0).to(device)

            # -------------------------
            # FORWARD PASS
            # -------------------------
            action_tensor, logp_tensor, value_tensor = policy.sample(obs_tensor)

            actions = {}
            logps = {}
            values = {}

            for i, v in enumerate(env.vars):
                actions[v] = action_tensor[0, i]
                values[v] = value_tensor[0, i]

                # per-agent logprob
                dist = torch.distributions.Normal(
                    action_tensor[0, i],
                    torch.exp(policy.log_std[i])
                )
                logps[v] = dist.log_prob(action_tensor[0, i])

            # -------------------------
            # ENV STEP
            # -------------------------
            next_obs, reward, done = env.step(actions)
            reward_t = torch.tensor(float(reward), device=device)

            reward_window.append(float(reward))
            ep_reward_sum += float(reward)

            # -------------------------
            # BOOTSTRAP VALUE
            # -------------------------
            next_values = {}

            with torch.no_grad():
                if not done:
                    next_obs_tensor = torch.cat(
                        [next_obs[v] for v in env.vars],
                        dim=-1
                    ).unsqueeze(0).to(device)

                    _, _, next_value_tensor = policy.forward(next_obs_tensor)

                    for i, v in enumerate(env.vars):
                        next_values[v] = next_value_tensor[0, i]
                else:
                    for v in env.vars:
                        next_values[v] = torch.tensor(0.0, device=device)

            # -------------------------
            # UPDATE
            # -------------------------
            for v in env.vars:

                v_t = values[v]
                v_next = next_values[v]

                advantage = reward_t + gamma * v_next - v_t

                actor_loss = -logps[v] * advantage.detach()
                critic_loss = advantage.pow(2)

                loss = actor_loss + 0.5 * critic_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                ep_agent_losses[v].append(float(loss.detach().cpu()))
                ep_agent_rewards[v] += float(reward) / num_agents

            obs = next_obs

            # -------------------------
            # LOGGING
            # -------------------------
            if step_counter % log_every == 0:
                avg100 = np.mean(reward_window)
                avg_loss = np.mean([np.mean(ep_agent_losses[v]) for v in env.vars])

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
                    f"Reward: {float(reward):.4f} | Avg(100): {avg100:.4f} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"Episode ETA: {episode_eta/60:.1f} min | "
                    f"Global ETA: {global_eta/60:.1f} min"
                )

        # -------------------------
        # EPISODE SUMMARY
        # -------------------------
        ep_avg_reward = ep_reward_sum / max(step_counter, 1)
        reward_per_epoch.append(ep_avg_reward)

        for v in env.vars:
            reward_per_agent[v].append(ep_agent_rewards[v] / max(step_counter, 1))
            loss_per_agent[v].append(np.mean(ep_agent_losses[v]) if ep_agent_losses[v] else 0.0)

        loss_per_epoch.append(np.mean([np.mean(ep_agent_losses[v]) for v in env.vars]) if env.vars else 0.0)

        causal_importance_per_epoch.append(
            compute_causal_importance(env, {"policy": policy})
        )

        # forecast metrics (kept but not modified)
        for v in env.vars:
            forecast_metrics_per_epoch[v]["mse"].append(
                float(np.mean(ep_var_mse[v])) if ep_var_mse[v] else 0.0
            )
            forecast_metrics_per_epoch[v]["mape"].append(
                float(np.mean(ep_var_mape[v])) if ep_var_mape[v] else 0.0
            )

        mse_per_epoch.append(0.0)

        print("\n" + "=" * 60)
        print(f"EPISODE {ep+1}/{episodes}")
        print(f"Reward: {ep_reward_sum:.4f}")
        print(f"Avg:    {ep_avg_reward:.4f}")
        print(f"Steps:  {step_counter}")
        print("=" * 60 + "\n")

        # -------------------------
        # CHECKPOINT
        # -------------------------
        if ep_avg_reward > best_epoch_reward:
            best_epoch_reward = ep_avg_reward
            best_epoch_idx = ep
            best_policy = copy.deepcopy(policy)

            torch.save(
                {
                    "policy": best_policy.state_dict(),
                    "graph": env.graph,
                    "best_epoch": best_epoch_idx,
                    "reward_per_epoch": reward_per_epoch,
                    "loss_per_epoch": loss_per_epoch,
                    "mse_per_epoch": mse_per_epoch,
                    "causal_importance_per_epoch": causal_importance_per_epoch,
                    "forecast_metrics_per_epoch": forecast_metrics_per_epoch,
                },
                checkpoint_path,
            )

            print(f"Checkpoint saved: {ep_avg_reward:.4f}")

    return (
        best_policy,
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