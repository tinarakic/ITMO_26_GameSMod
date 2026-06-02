import torch
from collections import deque
import time
import copy
import numpy as np

from environment import CausalEnv
from policy import Policy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# CAUSAL IMPORTANCE 
def compute_causal_importance(env):
    """
    Оценивает причинную важность (causal importance) связей в графе SCM
    на основе структуры графа и временных лагов.

    Алгоритм:
    1. Проходит по всем переменным среды.
    2. Для каждой переменной находит ее родительские узлы в графе.
    3. Для каждой причинной связи (parent -> variable):
       - извлекает временной лаг (lag) из ребра графа
       - добавляет вклад в importance, нормированный по числу родителей
    4. Агрегирует вклад всех связей и возвращает словарь важности.

    Формула вклада:
        importance(p -> v) += |lag| / (1 + number_of_parents)

    Args:
        env: CausalEnv
            Среда, содержащая SCM-граф и список переменных.

    Returns:
        dict:
            {(parent, target): score}, где score отражает
            структурную причинную важность связи.
    """
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


# GAE
def compute_gae(rewards, values, gamma=0.99, lam=0.95):
    """
    Вычисляет обобщенную оценку преимущества (GAE) и целевые возвраты для PPO.

    GAE решает дилемму смещения и дисперсии в методах Policy Gradient. Он 
    минимизирует дисперсию градиента (разброс оценок) и сохраняет низкое 
    смещение (bias) за счет экспоненциального взвешивания временных TD-ошибок 
    с шагом назад по траектории.

    Математическая логика:
        1. Инициализируется терминальное состояние V(T) = 0.
        2. Векторизованный проход выполняется в обратном времени (от T-1 до 0).
        3. Мгновенная TD-ошибка: 
           delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
        4. Накопление GAE сглаживания: 
           A_t^{GAE} = delta_t + (gamma * lam) * A_{t+1}^{GAE}
        5. Целевой возврат (target return) для критика восстанавливается как:
           R_t = A_t^{GAE} + V(s_t)

    Args:
        rewards: list[float]
            Награды по временным шагам.
        values: list[float]
            Оценки value-функции для каждого шага.
        gamma: float, optional
            Коэффициент дисконтирования (по умолчанию 0.99).
        lam: float, optional
            Параметр сглаживания GAE (по умолчанию 0.95).

    Returns:
        tuple:
            advantages: list[float]
                Оценки преимуществ для каждого шага.
            returns: list[float]
                Оценки целевых значений (returns) для critic.
    """
    advantages = []
    gae = 0.0
    values = values + [0.0]

    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)

    returns = [a + v for a, v in zip(advantages, values[:-1])]
    return advantages, returns


# TRAIN AGENTS
def train(
    data,
    graph,
    episodes=10,
    gamma=0.99,
    lookback=72,
    log_every=100,
    checkpoint_path="best_epoch.pth",
):
    
    """
    Функция обучения агентов в SCM RL-среде, используя PPO-подобный подход с GAE и стохастической policy.

    Модель обучает отдельную policy для каждой переменной графа и оптимизирует
    их на основе локальных наград, полученных из среды CausalEnv.

    Функциональность:
    1. Создание среды CausalEnv на основе данных и SCM-графа.
    2. Инициализация отдельной Policy и optimizer для каждой переменной.
    3. Сбор траекторий взаимодействия агента со средой (rollout).
    4. Вычисление преимуществ (GAE) и возвратов (returns).
    5. Обновление политик с использованием actor-critic loss.
    6. Подсчет метрик качества (reward, loss, MSE/MAE, causal importance).
    7. Сохранение лучшей модели (checkpoint) по среднему reward.

    Пошаговый алгоритм:
    -------------------
    1. Создает среду CausalEnv с заданным lookback.
    2. Для каждой переменной инициализирует Policy и Adam optimizer.
    3. Для каждого эпизода:
    3.1. Сбрасывает среду и собирает rollout:
            - генерирует действия через policies[v].sample()
            - получает награды через env.step()
            - сохраняет наблюдения, действия и награды
    3.2. После завершения эпизода:
            - вычисляет value-функции через policy.evaluate_actions()
            - считает GAE (advantages и returns)
            - нормализует advantages
            - обновляет параметры политики (actor + critic + entropy)
    4. Логирует метрики обучения и ETA.
    5. Вычисляет:
            - reward_per_epoch
            - loss_per_epoch
            - per-agent metrics
            - forecast metrics (MSE/MAE)
            - causal importance
    6. Сохраняет лучший checkpoint по epoch_avg_reward.

    Args:
        data: pandas.DataFrame
            Временные ряды с числовыми признаками.
        graph: networkx.DiGraph
            SCM-граф причинных зависимостей с атрибутами lag.
        episodes: int
            Количество эпизодов обучения.
        gamma: float
            Discount factor для GAE и RL-обновлений.
        lookback: int
            Размер временного окна наблюдений.
        log_every: int
            Частота логирования шагов обучения.
        checkpoint_path: str
            Путь для сохранения лучшей модели.

    Returns:
        tuple:
            policies:
                Обученные политики для всех переменных.
            reward_total_per_epoch:
                Суммарная награда по эпохам.
            reward_avg_per_epoch:
                Средняя награда по эпохам.
            reward_per_agent:
                Награды по каждой переменной.
            loss_total_per_epoch:
                Суммарные потери по эпохам.
            loss_avg_per_epoch:
                Средние потери по эпохам.
            loss_per_agent:
                Потери по каждой переменной.
            best_epoch:
                Эпоха с лучшим результатом.
            mse_per_epoch:
                Среднеквадратичная ошибка по эпохам.
            causal_scores:
                Оценки причинной важности по эпохам.
            forecast_metrics_per_epoch:
                MSE/MAE метрики по переменным и эпохам.
            env:
                Финальное состояние среды.
    """

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

    # TRAIN LOOP
    for ep in range(episodes):

        obs = env.reset()

        trajectories = {
            v: {"obs": [], "actions": [], "rewards": []}
            for v in env.vars
        }

        ep_reward_sum = 0.0
        step_counter = 0
        done = False

        # ROLLOUT
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

            ep_reward_sum += sum(rewards.values())  

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
                
                # GLOBAL ETA (TOTAL TRAINING)
                elapsed_total = now - start_time
                global_frac_done = global_step / max(total_steps, 1)
                global_frac_done = min(max(global_frac_done, 1e-8), 1.0)  # clamp to avoid div0
                eta_total = elapsed_total * (1 - global_frac_done) / global_frac_done

                # LOG STEP
                print(
                    f"[EP {ep+1}/{episodes}] | "
                    f"STEP={step_counter}/{len(data)} | "
                    f"REWARD={ep_reward_sum:.4f} | "
                    f"ETA={eta_total/60:.1f}min"
                )

        # UPDATE
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


        # LOSS PER AGENT
        for v in env.vars:
            if len(ep_losses[v]) > 0:
                loss_per_agent[v].append(float(np.mean(ep_losses[v])))
            else:
                loss_per_agent[v].append(0.0)

        # REWARD METRICS
        # Total reward 
        epoch_total_reward = float(ep_reward_sum)

        # Average reward per agent-decision
        epoch_avg_reward = (
            ep_reward_sum /
            max(step_counter * len(env.vars), 1)
        )

        reward_total_per_epoch.append(epoch_total_reward)
        reward_avg_per_epoch.append(epoch_avg_reward)

        # CAUSAL IMPORTANCE
        causal_scores.append(
            compute_causal_importance(env, policies)
        )

            
        for v in env.vars:
            
            r = trajectories[v]["rewards"]
            reward_per_agent[v].append(
                np.mean(r) if len(r) > 0 else 0.0
            )
            
            if len(r) > 0:
                arr = np.array(r, dtype=np.float32)
                
                mse = float(np.mean(arr ** 2))
                
                # MAE = -reward 
                mae = -float(np.mean(arr))
            else:
                mse, mae = 0.0, 0.0

            forecast_metrics_per_epoch[v]["mse"].append(mse)
            forecast_metrics_per_epoch[v]["mae"].append(mae)

        mse_per_epoch.append(epoch_avg_reward ** 2)


        # EPOCH LOSS
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

        # BEST MODEL CHECKPOINT
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
    