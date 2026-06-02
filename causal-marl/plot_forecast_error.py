import torch
import matplotlib.pyplot as plt


CHECKPOINT = "best_epoch.pth"


def load_metrics():
    """
    Загружает ошибки прогноза из checkpoint-файла.

    Returns:
        dict: словарь метрик прогнозирования по эпохам для каждой переменной.
    """

    ckpt = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=False
    )

    return ckpt["forecast_metrics_per_epoch"]


def print_variables(metrics):
    """
    Выводит список доступных переменных для анализа.

    Args:
        metrics: dict
            Словарь метрик прогнозирования по переменным.

    Returns:
        None
    """

    print("\nAvailable variables:\n")

    for i, v in enumerate(metrics.keys()):
        print(f"{i + 1}. {v}")


def plot_metric(metrics, variable, metric):
    """
    Строит график выбранной метрики качества по эпохам обучения.

    Для указанной переменной отображает динамику изменения
    ошибки прогнозирования на протяжении обучения модели.

    Args:
        metrics: dict
            Словарь метрик прогнозирования.
        variable: str
            Название анализируемой переменной.
        metric: str
            Метрика для отображения ("mse" или "mae").

    Returns:
        None
    """

    values = metrics[variable][metric]

    plt.figure(figsize=(10, 5))

    plt.plot(
        range(1, len(values) + 1),
        values,
        marker="o",
        linewidth=2
    )

    plt.title(
        f"{metric.upper()} by Epoch — {variable}"
    )

    plt.xlabel("Epoch")

    if metric == "mse":
        plt.ylabel("MSE")
    else:
        plt.ylabel("MAE")

    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    """
    Запускает интерактивный режим анализа метрик прогнозирования.

    Алгоритм:
    1. Загружает сохраненные метрики.
    2. Отображает список доступных переменных.
    3. Запрашивает выбор переменной.
    4. Запрашивает выбор метрики.
    5. Строит соответствующий график.
    6. Повторяет процесс до выхода пользователя.

    Returns:
        None
    """

    metrics = load_metrics()

    while True:

        print_variables(metrics)

        variable = input(
            "\nSelect variable (or q): "
        ).strip()

        if variable.lower() == "q":
            break

        if variable not in metrics:

            print(
                "\nVariable not found.\n"
            )

            continue

        metric = input(
            "Metric [mse/mae]: "
        ).strip().lower()

        if metric not in ["mse", "mae"]:

            print(
                "\nChoose mse or mae.\n"
            )

            continue

        plot_metric(
            metrics,
            variable,
            metric
        )


if __name__ == "__main__":
    main()