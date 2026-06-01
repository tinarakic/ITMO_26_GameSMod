import torch
from collections import deque
import time
import copy
import numpy as np

from environment import CausalEnv
from policy import Policy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------
# causal importance (unchanged)
# ----------------------------
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
# GAE (unchanged)
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


# ----------------------------
# TRAIN
# ----------------------------
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
        policies[v] = Policy(
            input_size=env.reset()[v].shape[-1],
            num_agents=1,
            horizon=1,   # keep consistent with current env reward design
        ).to(device)

        optimizers[v] = torch.optim.Adam(
            policies[v].parameters(),
            lr=1e-3,
        )

    reward_window = deque(maxlen=100)
    start_time = time.time()

    # ----------------------------
    # METRICS
    # ----------------------------
    reward_per_epoch = []
    reward_per_agent = {v: [] for v in env.vars}
    loss_per_epoch = []
    loss_per_agent = {v: [] for v in env.vars}
    mse_per_epoch = []
    causal_scores = []
    forecast_error = []

    best_epoch_reward = -np.inf
    best_policies = None
    best_epoch = -1

    total_steps = len(data) * episodes
    global_step = 0

    # ======================================================
    # TRAIN LOOP
    # ======================================================
    for ep in range(episodes):

        obs = env.reset()

        trajectories = {
            v: {
                "obs": [],
                "actions": [],
                "rewards": [],
            }
            for v in env.vars
        }

        ep_reward_sum = 0.0
        ep_start_time = time.time()
        step_counter = 0
        done = False

        # ----------------------------
        # ROLLOUT
        # ----------------------------
        while not done:

            step_counter += 1
            global_step += 1

            actions = {}

            for v in env.vars:
                if v not in obs:
                    continue

                x = obs[v].float().to(device)

                if x.ndim == 2:
                    x = x.unsqueeze(0)

                with torch.no_grad():
                    a, _ = policies[v].sample(x)

                actions[v] = a

            next_obs, reward, done = env.step(actions)

            reward = float(reward)
            reward = reward / (1.0 + abs(reward))

            reward_window.append(reward)
            ep_reward_sum += reward

            # store rollout
            for v in env.vars:
                if v not in obs:
                    continue

                trajectories[v]["obs"].append(obs[v])
                trajectories[v]["actions"].append(actions[v].cpu())
                trajectories[v]["rewards"].append(
                    reward / len(env.vars)
                )

            obs = next_obs

            # logging
            if step_counter % log_every == 0:

                avg100 = np.mean(reward_window)

                elapsed = time.time() - ep_start_time
                total_elapsed = time.time() - start_time

                global_eta = (
                    total_elapsed
                    / max(global_step, 1)
                    * (total_steps - global_step)
                )

                print(
                    f"[Episode {ep+1}/{episodes}] "
                    f"Step {step_counter} | "
                    f"Reward: {reward:.4f} | "
                    f"Avg(100): {avg100:.4f} | "
                    f"Global ETA: {global_eta/60:.1f} min"
                )

        # ======================================================
        # UPDATE
        # ======================================================
        ep_losses = {v: [] for v in env.vars}

        for v in env.vars:

            if len(trajectories[v]["obs"]) == 0:
                continue

            obs_batch = torch.cat(
                [o.to(device) for o in trajectories[v]["obs"]],
                dim=0,
            )

            actions_batch = torch.cat(
                [a.to(device) for a in trajectories[v]["actions"]],
                dim=0,
            )

            rewards_t = trajectories[v]["rewards"]

            # forward pass (for value bootstrap)
            with torch.no_grad():
                _, _, values_pred = policies[v].evaluate_actions(
                    obs_batch,
                    actions_batch,
                )

            advantages, returns = compute_gae(
                rewards_t,
                values_pred.cpu().tolist(),
                gamma=gamma,
            )

            advantages = torch.tensor(
                advantages,
                device=device,
                dtype=torch.float32,
            )

            returns = torch.tensor(
                returns,
                device=device,
                dtype=torch.float32,
            )

            advantages = (
                advantages - advantages.mean()
            ) / (advantages.std() + 1e-8)

            # recompute log probs + value
            logp, entropy, value = policies[v].evaluate_actions(
                obs_batch,
                actions_batch,
            )

            actor_loss = -(logp * advantages.detach()).mean()

            critic_loss = (value - returns).pow(2).mean()

            entropy_loss = entropy.mean()

            loss = (
                actor_loss
                + 0.5 * critic_loss
                - 0.001 * entropy_loss
            )

            optimizers[v].zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                policies[v].parameters(),
                1.0,
            )

            optimizers[v].step()

            ep_losses[v].append(
                float(loss.detach().cpu())
            )

        # ======================================================
        # METRICS
        # ======================================================
        ep_avg_reward = ep_reward_sum / max(step_counter, 1)

        reward_per_epoch.append(ep_avg_reward)

        causal_scores.append(
            compute_causal_importance(env, policies)
        )

        for v in env.vars:
            reward_per_agent[v].append(
                np.mean(trajectories[v]["rewards"])
            )
            loss_per_agent[v].append(
                np.mean(ep_losses[v]) if ep_losses[v] else 0.0
            )

        loss_per_epoch.append(
            np.mean([np.mean(ep_losses[v]) for v in env.vars])
        )

        mse_per_epoch.append(ep_avg_reward**2)

        forecast_error.append(
            {
                v: {
                    "mse": ep_avg_reward**2,
                    "mape": ep_avg_reward,
                }
                for v in env.vars
            }
        )

        # checkpoint
        if ep_avg_reward > best_epoch_reward:
            best_epoch_reward = ep_avg_reward
            best_epoch = ep

            best_policies = copy.deepcopy(policies)

            torch.save(
                {
                    "policies": {
                        v: best_policies[v].state_dict()
                        for v in env.vars
                    },
                    "graph": env.graph,
                    "best_epoch": best_epoch,
                    "reward_per_epoch": reward_per_epoch,
                    "loss_per_epoch": loss_per_epoch,
                    "causal_scores": causal_scores,
                    "forecast_error": forecast_error,
                },
                checkpoint_path,
            )

            print(
                f"Checkpoint saved at epoch {ep+1} "
                f"reward={ep_avg_reward:.4f}"
            )

        print("\n" + "=" * 60)
        print(f"EPISODE {ep+1}/{episodes}")
        print(f"Reward: {ep_reward_sum:.4f}")
        print(f"Avg:    {ep_avg_reward:.4f}")
        print(f"Steps:  {step_counter}")
        print("=" * 60 + "\n")

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