import matplotlib.pyplot as plt
import os
import numpy as np


def plot_training_metrics(
    reward_per_epoch,
    reward_per_agent=None,
    loss_per_epoch=None,
    loss_per_agent=None,
    save_dir="."
):
    """
    Correct visualization for:
    - global reward (environment-level)
    - per-agent loss (policy-level)
    - safe aggregation without assuming alignment
    """

    epochs = np.arange(1, len(reward_per_epoch) + 1)

    # ======================================================
    # REWARD (GLOBAL — NOT PER AGENT)
    # ======================================================
    fig1, ax1 = plt.subplots(figsize=(10, 5))

    ax1.plot(
        epochs,
        reward_per_epoch,
        'o-',
        label='Total Reward (Env-level)',
        color='blue'
    )

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Reward", color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.grid(True, linestyle='--', alpha=0.5)

    # ---- show per-agent spread (NOT averaged line) ----
    if reward_per_agent:

        for i, (agent, values) in enumerate(reward_per_agent.items()):

            # ensure alignment safety
            values = values[:len(epochs)]

            ax1.plot(
                range(1, len(values) + 1),
                values,
                linestyle="--",
                alpha=0.4,
                label=f"{agent} (agent)"
            )

    ax1.legend(loc='upper left')
    plt.title("Reward per Epoch (Global + Per-Agent View)")

    reward_path = os.path.join(save_dir, "reward_per_epoch.png")
    plt.savefig(reward_path, bbox_inches='tight')
    plt.show()

    # ======================================================
    # LOSS (TRUE PER-AGENT MEANINGFUL SIGNAL)
    # ======================================================
    if loss_per_epoch:

        fig2, ax2 = plt.subplots(figsize=(10, 5))

        ax2.plot(
            epochs,
            loss_per_epoch,
            'r^-',
            label='Total Loss (mean across agents)'
        )

        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss", color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        ax2.grid(True, linestyle='--', alpha=0.5)

        # ---- per-agent loss curves (correct) ----
        if loss_per_agent:

            for agent, values in loss_per_agent.items():

                values = values[:len(epochs)]

                ax2.plot(
                    range(1, len(values) + 1),
                    values,
                    linestyle="--",
                    alpha=0.5,
                    label=f"{agent}"
                )

        ax2.legend(loc='upper right')
        plt.title("Loss per Epoch (Total + Per-Agent)")

        loss_path = os.path.join(save_dir, "loss_per_epoch.png")
        plt.savefig(loss_path, bbox_inches='tight')
        plt.show()

    else:
        loss_path = None

    return reward_path, loss_path