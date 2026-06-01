import pandas as pd
import numpy as np

from scm_graph import load_scm_graph
from train import train
from visualization import plot_training_metrics
# from evaluation import predict
from causal import plot_causal

# LOAD DATA
data = pd.read_csv("data.csv").select_dtypes(include=[np.number])
valid_nodes = list(data.columns)
graph = load_scm_graph("scm.csv", valid_nodes)
graph = graph.subgraph(valid_nodes).copy()


# TRAIN
(
    policies,
    reward_per_epoch,
    reward_per_agent,
    loss_per_epoch,
    loss_per_agent,
    best_epoch,
    mse_per_epoch,
    causal_scores,
    forecast_error,
    env
) = train(
    data = data,
    graph = graph,
    episodes = 10,
    lookback = 72,
    log_every = 50
)

# VISUALIZATION: TRAINING METRICS
plot_training_metrics(
    reward_per_epoch=reward_per_epoch,
    reward_per_agent=reward_per_agent,
    loss_per_epoch=loss_per_epoch,
    loss_per_agent=loss_per_agent,
)

# CAUSAL IMPORTANCE 
final_scores = causal_scores[-1]
plot_causal(graph, final_scores, top_n=15)


# PREDICTION
# preds = predict(env, policies)

# print("\nPredictions:")
# for k, v in preds.items():
#     print(k, ":", v)