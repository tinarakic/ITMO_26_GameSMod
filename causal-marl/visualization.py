import matplotlib.pyplot as plt
import os


def plot_training_metrics(
    reward_total_per_epoch,
    reward_avg_per_epoch,
    loss_total_per_epoch,
    loss_avg_per_epoch,
    save_dir="."
):
    
    """
    Визуализирует динамику обучения модели по эпохам и сохраняет графики
    награды (reward) и потерь (loss).

    Функция строит два отдельных графика:
    1. Reward per Epoch (суммарная и средняя награда)
    2. Loss per Epoch (суммарная и средняя потеря)

    Для каждого графика используется две оси Y:
    - основная ось: total metric
    - дополнительная ось: average metric

    Args:
        reward_total_per_epoch: list[float]
            Суммарная награда по каждой эпохе.
        reward_avg_per_epoch: list[float]
            Средняя награда по каждой эпохе.
        loss_total_per_epoch: list[float]
            Суммарная функция потерь по эпохам.
        loss_avg_per_epoch: list[float]
            Средняя функция потерь по эпохам.
        save_dir: str, optional
            Директория для сохранения графиков (по умолчанию текущая).

    Returns:
        tuple:
            reward_path: str
                Путь к сохраненному графику reward.
            loss_path: str
                Путь к сохраненному графику loss.
    """

    os.makedirs(save_dir, exist_ok=True)

    epochs = range(1, len(reward_total_per_epoch) + 1)

    # REWARD 
    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.plot(
        epochs,
        reward_total_per_epoch,
        "b-o",
        linewidth=2,
        label="Total Reward"
    )
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Total Reward", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(
        epochs,
        reward_avg_per_epoch,
        "g--s",
        linewidth=2,
        label="Avg Reward"
    )
    ax2.set_ylabel("Average Reward", color="green")
    ax2.tick_params(axis="y", labelcolor="green")

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")

    plt.title("Reward per Epoch")

    reward_path = os.path.join(save_dir, "reward_per_epoch.png")
    plt.tight_layout()
    plt.savefig(reward_path, dpi=300)
    plt.close()

    # LOSS
    fig, ax3 = plt.subplots(figsize=(12, 6))

    ax3.plot(
        epochs,
        loss_total_per_epoch,
        "r-o",
        linewidth=2,
        label="Total Loss"
    )
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Total Loss", color="red")
    ax3.tick_params(axis="y", labelcolor="red")
    ax3.grid(True, alpha=0.3)

    ax4 = ax3.twinx()
    ax4.plot(
        epochs,
        loss_avg_per_epoch,
        "m--s",
        linewidth=2,
        label="Avg Loss"
    )
    ax4.set_ylabel("Average Loss", color="magenta")
    ax4.tick_params(axis="y", labelcolor="magenta")

    lines_1, labels_1 = ax3.get_legend_handles_labels()
    lines_2, labels_2 = ax4.get_legend_handles_labels()
    ax3.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")

    plt.title("Loss per Epoch")

    loss_path = os.path.join(save_dir, "loss_per_epoch.png")
    plt.tight_layout()
    plt.savefig(loss_path, dpi=300)
    plt.close()

    return reward_path, loss_path