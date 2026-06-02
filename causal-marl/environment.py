import numpy as np
import torch
from data_utils import normalize_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CausalEnv:
    """
    Среда для моделирования причинно-следственных зависимостей с использованием
    временных рядов и графа причинности (SCM).

    Класс предоставляет интерфейс оркужения для RL:
    - reset(): сброс состояния среды и получение начальных наблюдений
    - step(actions): выполнение действия (предсказаний) и получение награды

    Атрибуты:
    ----------
    raw_data : pandas.DataFrame
        Исходные данные без нормализации.
    data : pandas.DataFrame
        Нормализованные числовые данные для обучения/предсказания.
    scalers : dict
        Словарь StandardScaler для каждого столбца, используемый для денормализации.
    graph : networkx.DiGraph
        Направленный граф причинных зависимостей между переменными.
        Ребра должны иметь атрибут "lag" для временного лага.
    lookback : int
        Количество прошлых шагов, используемых для построения признаков.
    H : int
        Горизонт прогнозирования (количество шагов вперед).
    vars : list
        Список имен переменных в графе.
    t : int
        Текущий индекс времени в данных.

    Методы:
    --------
    reset():
        Сбрасывает текущее время до значения lookback и возвращает начальные наблюдения.

    _obs():
        Генерирует наблюдение для текущего времени t.
        Для каждой переменной собирает:
            - значения родительских переменных с учетом лагов
            - собственную историю переменной
        Возвращает словарь {variable: torch.Tensor(features)}.

    step(actions):
        Выполняет шаг среды:
        1. Принимает словарь actions {variable: predicted_values}.
        2. Вычисляет награды для каждой переменной как отрицательное MAE между
           предсказанными и истинными нормализованными значениями.
        3. Обновляет текущее время t.
        4. Возвращает кортеж (observations, rewards, done), где:
            - observations: словарь наблюдений после шага
            - rewards: словарь наград по переменным
            - done: булево значение окончания эпизода/эпохи
    """

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

            # Включение прошлих значений самой переменной
            self_series = self.data[v].values[self.t - self.lookback:self.t]
            feats.append(self_series)

            obs[v] = torch.tensor(
                np.stack(feats, axis=-1),
                dtype=torch.float32
            ).unsqueeze(0)

        return obs

    def step(self, actions):
        rewards = {}

        # Вознаграждение - отрицательная средная абсолютная ошибка (MAE)
        for v in self.vars:

            if v not in self.data.columns:
                continue

            if v not in actions:
                continue

            pred = actions[v].detach().squeeze()  # (H,)
            true = self.data[v].values[self.t:self.t + self.H]

            if len(true) < self.H:
                continue

            true = torch.tensor(true, dtype=torch.float32, device=pred.device)
            pred = pred.float().to(pred.device)

            # MAE для нормализованных значений
            mae = torch.mean(torch.abs(true - pred))

            # reward per agent = -MAE
            rewards[v] = -float(mae)

        self.t += 1
        done = self.t >= len(self.data) - self.H - 1

        return self._obs(), rewards, done