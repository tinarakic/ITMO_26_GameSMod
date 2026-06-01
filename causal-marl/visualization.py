import matplotlib.pyplot as plt
import os

def plot_training_metrics(
    reward_per_epoch,
    reward_per_agent=None,
    loss_per_epoch=None,
    loss_per_agent=None,  
    save_dir="."
):
    """
    Создаёт два отдельных графика:
    1. Reward (общая + средняя на агента по эпохе)
    2. Loss (общий + средний на агента по эпохе)

    reward_per_epoch: список общей награды за каждую эпоху
    reward_per_agent: словарь агент -> список наград по эпохам
    loss_per_epoch: список общих потерь за каждую эпоху
    loss_per_agent: словарь агент -> список потерь по эпохам 
    save_dir: директория для сохранения изображений
    """
    
    epochs = range(1, len(reward_per_epoch)+1)


    # REWARD
    fig1, ax1 = plt.subplots(figsize=(10,5))
    ax1.plot(epochs, reward_per_epoch, 'o-', label='Total Reward', color='blue')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Total Reward", color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # Compute avg reward per agent
    if reward_per_agent:
        avg_rewards = [
            sum(r[i] for r in reward_per_agent.values()) / len(reward_per_agent)
            for i in range(len(reward_per_epoch))
        ]
        ax2 = ax1.twinx()
        ax2.plot(epochs, avg_rewards, 's--', color='green', label='Avg Reward per Agent')
        ax2.set_ylabel("Avg Reward per Agent", color='green')
        ax2.tick_params(axis='y', labelcolor='green')
        ax2.legend(loc='upper right')

    ax1.legend(loc='upper left')
    plt.title("Reward per Epoch")
    reward_path = os.path.join(save_dir, "reward_per_epoch.png")
    plt.savefig(reward_path)
    plt.show()


    # LOSS
    if loss_per_epoch:
        fig2, ax3 = plt.subplots(figsize=(10,5))
        ax3.plot(epochs, loss_per_epoch, 'r^-', label='Total Loss')
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("Total Loss", color='red')
        ax3.tick_params(axis='y', labelcolor='red')
        ax3.grid(True, linestyle='--', alpha=0.5)

        # Compute avg loss per agent
        if loss_per_agent:
            avg_losses = [
                sum(loss[i] for loss in loss_per_agent.values()) / len(loss_per_agent)
                for i in range(len(loss_per_epoch))
            ]
        else:
            avg_losses = [
                sum(loss_per_epoch[:i+1]) / (i+1)
                for i in range(len(loss_per_epoch))
            ]

        ax4 = ax3.twinx()
        ax4.plot(epochs, avg_losses, 'm--', label='Avg Loss per Agent')
        ax4.set_ylabel("Avg Loss per Agent", color='magenta')
        ax4.tick_params(axis='y', labelcolor='magenta')
        ax4.legend(loc='upper right')

        ax3.legend(loc='upper left')
        plt.title("Loss per Epoch")
        loss_path = os.path.join(save_dir, "loss_per_epoch.png")
        plt.savefig(loss_path)
        plt.show()
    else:
        loss_path = None

    return reward_path, loss_path