import torch
import torch.nn as nn
from torch.distributions import Normal


class Policy(nn.Module):
    """
    Нейронная сеть policy и критика для прогнозирования временных рядов
    в многoагентной причинно-следственной среде.

    Модель использует LSTM для извлечения временных зависимостей из входной
    последовательности и формирует:
    - параметры распределения действий (mu и std) для актора;
    - оценку состояния (value) для критика.

    Архитектура:
    ------------
    1. LSTM-слой для обработки временного окна наблюдений.
    2. Полносвязный слой mu для вычисления средних значений действий.
    3. Обучаемый параметр log_std для вычисления стандартного отклонения.
    4. Полносвязный слой value для оценки состояния среды.

    Атрибуты:
    ----------
    num_agents : int
        Количество агентов (или прогнозируемых выходов).
    hidden : int
        Размер скрытого состояния LSTM.
    H : int
        Горизонт прогнозирования.
    lstm : nn.LSTM
        Рекуррентный слой для обработки временных рядов.
    mu : nn.Linear
        Слой, вычисляющий средние значения действий.
    value : nn.Linear
        Слой критика для оценки состояния.
    log_std : nn.Parameter
        Обучаемый логарифм стандартного отклонения распределения действий.

    Методы:
    --------
    forward(x):
        Выполняет прямой проход сети и возвращает параметры распределения
        действий и оценку состояния.

    sample(x):
        Генерирует действие из нормального распределения с помощью
        репараметризации (rsample).

    evaluate_actions(x, actions):
        Вычисляет логарифм вероятности действий, энтропию распределения
        и значение критика для PPO-обновления.
    """

    def __init__(self, input_size, num_agents, hidden=64, horizon=1, ):
        """
        Инициализирует модель агента и критика.

        Args:
            input_size: int
                Размерность входных признаков.
            num_agents: int
                Количество агентов (или выходных переменных).
            hidden: int, optional
                Размер скрытого состояния LSTM (по умолчанию 64).
            horizon: int, optional
                Горизонт прогнозирования (по умолчанию 1).
        """
        super().__init__()

        self.num_agents = num_agents
        self.hidden = hidden
        self.H = horizon

        self.lstm = nn.LSTM(
            input_size,
            hidden,
            batch_first=True,
        )

        self.mu = nn.Linear(
            hidden,
            num_agents * horizon,
        )

        self.value = nn.Linear(
            hidden,
            1,
        )

        self.log_std = nn.Parameter(
            torch.zeros(num_agents, horizon)
        )

    def forward(self, x):
        """
        Выполняет прямой проход через сеть.

        Алгоритм:
        1. Передает входную последовательность через LSTM.
        2. Извлекает последнее скрытое состояние.
        3. Вычисляет средние значения действий (mu).
        4. Вычисляет стандартные отклонения (std).
        5. Вычисляет оценку состояния (value).
        6. Возвращает параметры распределения и значение критика.

        Args:
            x: torch.Tensor
                Тензор формы (batch_size, seq_len, input_size).

        Returns:
            tuple:
                mu: torch.Tensor
                    Средние значения действий.
                std: torch.Tensor
                    Стандартные отклонения действий.
                value: torch.Tensor
                    Оценка состояния среды.
        """

        h, _ = self.lstm(x)
        h = h[:, -1, :]
        mu = self.mu(h).view(-1, self.num_agents, self.H, )
        std = torch.exp(self.log_std).unsqueeze(0)
        value = self.value(h).squeeze(-1)

        return mu, std, value
    

    def sample(self, x):
        """
        Генерирует действие на основе текущей политики.

        Использует нормальное распределение с параметрами,
        полученными из сети, и метод репараметризации
        для обеспечения дифференцируемости.

        Алгоритм:
        1. Выполняет прямой проход через сеть.
        2. Создает распределение Normal(mu, std).
        3. Генерирует действие методом rsample().
        4. Возвращает действие и оценку состояния.

        Args:
            x: torch.Tensor
                Входное наблюдение.

        Returns:
            tuple:
                action: torch.Tensor
                    Сгенерированное действие.
                value: torch.Tensor
                    Оценка состояния среды.
        """

        mu, std, value = self.forward(x)
        dist = Normal(mu, std)
        action = dist.rsample()

        return action, value

    def evaluate_actions(self, x, actions, ):
        """
        Оценивает заданные действия относительно текущей политики.

        Используется при обновлении PPO для вычисления
        логарифмов вероятностей действий и энтропии распределения.

        Алгоритм:
        1. Выполняет прямой проход через сеть.
        2. Формирует распределение Normal(mu, std).
        3. Вычисляет log-probability переданных действий.
        4. Вычисляет энтропию распределения.
        5. Возвращает log-probability, энтропию и value.

        Args:
            x: torch.Tensor
                Входные наблюдения.
            actions: torch.Tensor
                Действия, которые необходимо оценить.

        Returns:
            tuple:
                logp: torch.Tensor
                    Логарифм вероятности действий.
                entropy: torch.Tensor
                    Энтропия распределения действий.
                value: torch.Tensor
                    Оценка состояния среды.
        """

        mu, std, value = self.forward(x)
        dist = Normal(mu, std)
        logp = dist.log_prob(actions).sum(dim=(-1, -2))
        entropy = dist.entropy().sum( dim=(-1, -2))

        return logp, entropy, value