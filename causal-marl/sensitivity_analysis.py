import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scm_graph import load_scm_graph
from train import train


# =========================================================
# CONFIG
# =========================================================
DATA_PATH = "data.csv"
SCM_PATH = "scm.csv"

MODE = "gamma"   # "gamma" or "lookback"

GAMMAS = [0.90, 0.95, 0.99]
LOOKBACKS = [5, 10, 20]

EPISODES = 3


# =========================================================
# LOAD DATA
# =========================================================
data = pd.read_csv(DATA_PATH).select_dtypes(include=[np.number]).tail(100)
valid_nodes = list(data.columns)

graph = load_scm_graph(SCM_PATH, valid_nodes)
graph = graph.subgraph(valid_nodes).copy()


# =========================================================
# STORAGE
# =========================================================
results = {}


# =========================================================
# RUN SWEEP
# =========================================================
param_grid = GAMMAS if MODE == "gamma" else LOOKBACKS

for param in param_grid:

    print(f"\nRunning {MODE} = {param}")

    if MODE == "gamma":
        gamma = param
        lookback = 10
    else:
        gamma = 0.99
        lookback = param

    (
        policies,
        reward_per_epoch,
        reward_per_agent,
        _,
        loss_per_epoch,
        loss_per_agent,
        _,
        best_epoch,
        mse_per_epoch,
        causal_scores,
        forecast_metrics_per_epoch,
        env
    ) = train(
        data=data,
        graph=graph,
        episodes=EPISODES,
        gamma=gamma,
        lookback=lookback,
        log_every=50
    )

    results[param] = {
        "reward": reward_per_epoch,
        "forecast_metrics": forecast_metrics_per_epoch
    }


# =========================================================
# SAVE RAW RESULTS
# =========================================================
np.save("sensitivity_results.npy", results)
print("\nSaved sensitivity_results.npy")


# =========================================================
# PLOTTING HELPERS
# =========================================================
def get_param_list(results):
    return list(results.keys())


def get_variables(results, param):
    return list(results[param]["forecast_metrics"].keys())


def plot_reward_comparison(results):
    plt.figure(figsize=(10, 5))

    for param in results.keys():
        values = results[param]["reward"]
        plt.plot(values, marker="o", label=str(param))

    plt.title(f"Reward Comparison ({MODE})")
    plt.xlabel("Epoch")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.legend()

    path = f"sensitivity_reward_comparison_{MODE}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")

    plt.show()


def plot_metric_comparison(results, variable, metric):

    plt.figure(figsize=(10, 5))

    for param in results.keys():

        if variable not in results[param]["forecast_metrics"]:
            continue

        values = results[param]["forecast_metrics"][variable][metric]

        plt.plot(values, marker="o", label=f"{param}")

    plt.title(f"{metric.upper()} Comparison | {variable}")
    plt.xlabel("Epoch")
    plt.ylabel(metric.upper())
    plt.grid(True)
    plt.legend()

    path = f"sensitivity_{metric}_comparison_{variable}_{MODE}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")

    plt.show()


# =========================================================
# INTERACTIVE VIEWER
# =========================================================
print("\n=== SENSITIVITY ANALYSIS VIEWER ===")

while True:

    print("\nAvailable parameters:")
    print(list(results.keys()))

    print("\nOptions:")
    print("1 - Compare Reward across parameters")
    print("2 - Compare MSE across parameters")
    print("3 - Compare MAE across parameters")
    print("q - quit")

    choice = input("Select: ").strip().lower()

    if choice == "q":
        break

    # -----------------------------
    # REWARD COMPARISON
    # -----------------------------
    if choice == "1":
        plot_reward_comparison(results)

    # -----------------------------
    # MSE / MAE COMPARISON
    # -----------------------------
    elif choice in ["2", "3"]:

        metric = "mse" if choice == "2" else "mae"

        sample_param = list(results.keys())[0]
        vars_list = list(results[sample_param]["forecast_metrics"].keys())

        print("\nVariables:")
        for i, v in enumerate(vars_list):
            print(i, v)

        v = input("Select variable: ").strip()

        if v not in results[sample_param]["forecast_metrics"]:
            print("Invalid variable")
            continue

        plot_metric_comparison(results, v, metric)

    else:
        print("Unknown option")