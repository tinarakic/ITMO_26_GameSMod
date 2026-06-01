import matplotlib.pyplot as plt
import os


def plot_training_metrics(
    reward_total_per_epoch,
    reward_avg_per_epoch,
    loss_total_per_epoch,
    loss_avg_per_epoch,
    save_dir="."
):

    os.makedirs(save_dir, exist_ok=True)

    epochs = range(1, len(reward_total_per_epoch) + 1)

    # ==================================================
    # REWARD (dual axis)
    # ==================================================
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Total reward (left axis)
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

    # Average reward (right axis)
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

    # combined legend
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")

    plt.title("Reward per Epoch")

    reward_path = os.path.join(save_dir, "reward_per_epoch.png")
    plt.tight_layout()
    plt.savefig(reward_path, dpi=300)
    plt.close()

    # ==================================================
    # LOSS (dual axis)
    # ==================================================
    fig, ax3 = plt.subplots(figsize=(12, 6))

    # Total loss (left axis)
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

    # Average loss (right axis)
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

    # combined legend
    lines_1, labels_1 = ax3.get_legend_handles_labels()
    lines_2, labels_2 = ax4.get_legend_handles_labels()
    ax3.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")

    plt.title("Loss per Epoch")

    loss_path = os.path.join(save_dir, "loss_per_epoch.png")
    plt.tight_layout()
    plt.savefig(loss_path, dpi=300)
    plt.close()

    return reward_path, loss_path