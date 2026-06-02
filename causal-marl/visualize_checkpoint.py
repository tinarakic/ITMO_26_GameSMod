import torch
import os
import numpy as np
import matplotlib.pyplot as plt


# CONFIG
CHECKPOINT_PATH = "best_epoch.pth"
SAVE_DIR = "plots"


# LOAD CHECKPOINT
def load_checkpoint(path):
    '''
    Загружает checkpoint-файл.
    '''
    return torch.load(path, map_location="cpu", weights_only=False)


def plot_dual_axis(x, y1, y2, title,
                   y1_label, y2_label,
                   y1_color, y2_color,
                   style1, style2,
                   save_path):
    """
    Создает и сохраняет график суммарной и средней награды и loss.
    Args:
        x (numpy.ndarray): Значения оси X (эпохи).
        y1 (list[float] | numpy.ndarray): Данные для левой оси Y.
        y2 (list[float] | numpy.ndarray): Данные для правой оси Y.
        title (str): Заголовок графика.
        y1_label (str): Подпись левой оси.
        y2_label (str): Подпись правой оси.
        y1_color (str): Цвет левого графика.
        y2_color (str): Цвет правого графика.
        style1 (str): Маркер стиля для левого графика (например, 'b-o').
        style2 (str): Маркер стиля для правого графика (например, 'g--s').
        save_path (str): Путь для сохранения PNG файла.
    """

    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.plot(x, y1, style1, linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel(y1_label, color=y1_color)
    ax1.tick_params(axis="y", labelcolor=y1_color)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(x, y2, style2, linewidth=2)
    ax2.set_ylabel(y2_label, color=y2_color)
    ax2.tick_params(axis="y", labelcolor=y2_color)

    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# MULTI-AGENT PLOT 
def plot_multi_agent(epochs, data_dict, title, ylabel, save_path):
    """
    Визуализирует индивидуальные кривые обучения для каждого агента на одном графике.

    Args:
        epochs (numpy.ndarray): Массив номеров эпох (ось X).
        data_dict (dict[str | int, list[float]]): Метрики агентов, где ключ — ID агента.
        title (str): Заголовок графика.
        ylabel (str): Подпись оси Y.
        save_path (str): Путь для сохранения PNG файла.

    Returns:
        None
    """

    plt.figure(figsize=(12, 6))

    valid_plots = 0

    for agent, values in data_dict.items():

        if values is None:
            continue
        if len(values) == 0:
            continue

        values = np.array(values)

        if len(values) != len(epochs):
            print(f"Skipping {agent}: len(values)={len(values)} != len(epochs)={len(epochs)}")
            continue

        plt.plot(epochs, values, linewidth=2, marker="o", label=str(agent))
        valid_plots += 1

    if valid_plots == 0:
        print(f"No valid data for {title}, skipping plot")
        return

    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)

    plt.legend(
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        borderaxespad=0
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


# MAIN
def main():
    """
    Основная функция для загрузки checkpoint и генерации графиков наград и loss.
    """
    os.makedirs(SAVE_DIR, exist_ok=True)

    ckpt = load_checkpoint(CHECKPOINT_PATH)

    reward_total = ckpt.get("reward_total_per_epoch", [])
    reward_avg = ckpt.get("reward_avg_per_epoch", [])
    loss_total = ckpt.get("loss_total_per_epoch", [])
    loss_avg = ckpt.get("loss_avg_per_epoch", [])

    reward_per_agent = ckpt.get("reward_per_agent", {})
    loss_per_agent = ckpt.get("loss_per_agent", {})

    epochs = np.arange(1, len(reward_total) + 1)

    # REWARD
    plot_dual_axis(
        epochs,
        reward_total,
        reward_avg,
        "Reward per Epoch",
        "Total Reward",
        "Average Reward",
        "blue",
        "green",
        "b-o",
        "g--s",
        os.path.join(SAVE_DIR, "reward_per_epoch.png")
    )

    # LOSS
    plot_dual_axis(
        epochs,
        loss_total,
        loss_avg,
        "Loss per Epoch",
        "Total Loss",
        "Average Loss",
        "red",
        "magenta",
        "r-o",
        "m--s",
        os.path.join(SAVE_DIR, "loss_per_epoch.png")
    )

    # REWARD PER AGENT 
    if reward_per_agent:
        plot_multi_agent(
            epochs,
            reward_per_agent,
            "Reward per Agent",
            "Reward",
            os.path.join(SAVE_DIR, "reward_per_agent.png")
        )

    # LOSS PER AGENT 
    if loss_per_agent:
        plot_multi_agent(
            epochs,
            loss_per_agent,
            "Loss per Agent",
            "Loss",
            os.path.join(SAVE_DIR, "loss_per_agent.png")
        )


# RUN
if __name__ == "__main__":
    main()