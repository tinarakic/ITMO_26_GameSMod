import torch
import matplotlib.pyplot as plt


CHECKPOINT = "best_epoch.pth"


def load_metrics():

    ckpt = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=False
    )

    return ckpt["forecast_metrics_per_epoch"]


def print_variables(metrics):

    print("\nAvailable variables:\n")

    for i, v in enumerate(metrics.keys()):
        print(f"{i + 1}. {v}")


def plot_metric(metrics, variable, metric):

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