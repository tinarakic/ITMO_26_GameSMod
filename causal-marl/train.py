import torch
from collections import deque
import time
import copy
import numpy as np

from environment import CausalEnv
from policy import Policy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------
# CAUSAL IMPORTANCE (restored)
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
# GAE
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
    lookback=72,
    log_every=100,
    checkpoint_path="best_epoch.pth",
):

    env = CausalEnv(data, graph, lookback=lookback)

    policies = {}
    optimizers = {}

    sample_obs = env.reset()

    for v in env.vars:
        if v not in sample_obs:
            continue

        policies[v] = Policy(
            input_size=sample_obs[v].shape[-1],
            num_agents=1,
            horizon=1,
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
    # Reward metrics
    reward_total_per_epoch = []
    reward_avg_per_epoch = []

    # Loss metrics
    loss_total_per_epoch = []
    loss_avg_per_epoch = []

    # Per-agent metrics
    reward_per_agent = {v: [] for v in env.vars}
    loss_per_agent = {v: [] for v in env.vars}

    mse_per_epoch = []
    causal_scores = []

    forecast_metrics_per_epoch = {
        v: {"mse": [], "mae": []}
        for v in env.vars
    }

    best_reward = -np.inf
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
            v: {"obs": [], "actions": [], "rewards": []}
            for v in env.vars
        }

        ep_reward_sum = 0.0
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

            next_obs, rewards, done = env.step(actions)

            ep_reward_sum += sum(rewards.values())  # instead of np.mean(list(rewards.values()))


            # store trajectory
            for v in env.vars:
                if v not in obs:
                    continue

                trajectories[v]["obs"].append(obs[v])
                trajectories[v]["actions"].append(actions[v].cpu())
                trajectories[v]["rewards"].append(rewards[v])

            obs = next_obs

            now = time.time()
    
            if step_counter % log_every == 0:
                now = time.time()
                
                # ----------------------------
                # GLOBAL ETA (TOTAL TRAINING)
                # ----------------------------
                elapsed_total = now - start_time
                global_frac_done = global_step / max(total_steps, 1)
                global_frac_done = min(max(global_frac_done, 1e-8), 1.0)  # clamp to avoid div0
                eta_total = elapsed_total * (1 - global_frac_done) / global_frac_done

                # ----------------------------
                # EPOCH ETA (CURRENT EPISODE)
                # ----------------------------
                # fraction of steps done in this epoch
                epoch_frac_done = step_counter / max(len(data), 1)
                epoch_frac_done = min(max(epoch_frac_done, 1e-8), 1.0)
                eta_epoch = elapsed_total * (1 - epoch_frac_done) / epoch_frac_done / episodes

                # ----------------------------
                # PRINT
                # ----------------------------
                print(
                    f"[EP {ep+1}/{episodes}] | "
                    f"STEP={step_counter}/{len(data)} | "
                    f"GLOBAL_STEP={global_step}/{total_steps} | "
                    f"REWARD={ep_reward_sum:.4f} | "
                    f"ETA_EPOCH={eta_epoch/60:.1f}min | "
                    f"ETA_TOTAL={eta_total/60:.1f}min"
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
                advantages, device=device, dtype=torch.float32
            )

            returns = torch.tensor(
                returns, device=device, dtype=torch.float32
            )

            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            logp, entropy, value = policies[v].evaluate_actions(
                obs_batch,
                actions_batch,
            )

            actor_loss = -(logp * advantages.detach()).mean()
            critic_loss = (value - returns).pow(2).mean()
            entropy_loss = entropy.mean()

            loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy_loss

            optimizers[v].zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(policies[v].parameters(), 1.0)

            optimizers[v].step()

            ep_losses[v].append(float(loss.detach().cpu()))

        # ======================================================
        # METRICS
        # ======================================================
        # ======================================================
        # REWARD METRICS
        # ======================================================

        # Total reward accumulated during the episode
        epoch_total_reward = float(ep_reward_sum)

        # Average reward per agent-decision
        epoch_avg_reward = (
            ep_reward_sum /
            max(step_counter * len(env.vars), 1)
        )

        reward_total_per_epoch.append(epoch_total_reward)
        reward_avg_per_epoch.append(epoch_avg_reward)

        # 🔥 RESTORED CAUSAL IMPORTANCE
        causal_scores.append(
            compute_causal_importance(env, policies)
        )

            
    # get the rewards (which are already -MAE per step in your env.step)
        for v in env.vars:
            
            # get the rewards (which are already -MAE per step in your env.step)
            r = trajectories[v]["rewards"]
            
            # Save the reward per agent (negative MAE)
            reward_per_agent[v].append(
                np.mean(r) if len(r) > 0 else 0.0
            )
            
            # Save metrics: MSE and MAE directly from the reward
            if len(r) > 0:
                arr = np.array(r, dtype=np.float32)
                
                # MSE in terms of reward
                mse = float(np.mean(arr ** 2))
                
                # MAE = -reward (because reward = -MAE)
                mae = -float(np.mean(arr))
            else:
                mse, mae = 0.0, 0.0

            forecast_metrics_per_epoch[v]["mse"].append(mse)
            forecast_metrics_per_epoch[v]["mae"].append(mae)

        # optional: aggregate MSE across variables
        mse_per_epoch.append(epoch_avg_reward ** 2)

        # ======================================================
        # EPOCH LOSS (CORRECT VERSION)
        # ======================================================

        valid_losses = []

        for v in env.vars:
            if len(ep_losses[v]) > 0:
                valid_losses.append(np.mean(ep_losses[v]))

        epoch_total_loss = (
            float(np.sum(valid_losses))
            if len(valid_losses) > 0
            else 0.0
        )

        epoch_avg_loss = (
            float(np.mean(valid_losses))
            if len(valid_losses) > 0
            else 0.0
        )

        loss_total_per_epoch.append(epoch_total_loss)
        loss_avg_per_epoch.append(epoch_avg_loss)

        # ======================================================
        # CHECKPOINT
        # ======================================================
        if epoch_avg_reward > best_reward:

            best_reward = epoch_avg_reward
            best_epoch = ep
            best_policies = copy.deepcopy(policies)

            torch.save(
                {
                    "policies": {
                        v: best_policies[v].state_dict()
                        for v in env.vars
                        if v in best_policies
                    },
                    "input_sizes": {
                        v: sample_obs[v].shape[-1]
                        for v in sample_obs.keys()
                    },
                    "best_epoch": best_epoch,
                    "reward_total_per_epoch": reward_total_per_epoch,
                    "reward_avg_per_epoch": reward_avg_per_epoch,

                    "reward_per_agent": reward_per_agent,

                    "loss_total_per_epoch": loss_total_per_epoch,
                    "loss_avg_per_epoch": loss_avg_per_epoch,

                    "loss_per_agent": loss_per_agent,
                    "causal_scores": causal_scores,
                    "forecast_metrics_per_epoch": forecast_metrics_per_epoch,
                },
                checkpoint_path,
            )

            print(f"Checkpoint saved epoch={ep+1} reward={epoch_avg_reward:.4f}")

        print("\n" + "=" * 60)
        print(f"EPOCH {ep+1}/{episodes}")
        print(f"Reward: {epoch_avg_reward:.4f}")
        print(f"Steps:  {step_counter}")
        print("=" * 60 + "\n")

    return (
        policies,

        reward_total_per_epoch,
        reward_avg_per_epoch,

        reward_per_agent,

        loss_total_per_epoch,
        loss_avg_per_epoch,

        loss_per_agent,

        best_epoch,
        mse_per_epoch,
        causal_scores,
        forecast_metrics_per_epoch,
        env,
)
    