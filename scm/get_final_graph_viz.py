import os

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from adjustText import adjust_text

# INPUT / OUTPUT
CSV_FILE = "full_dataset_pcmci.csv"
OUTPUT_FILE = "causal_graph.png"

# LOAD CSV
df = pd.read_csv(CSV_FILE)


# BUILD GRAPH
G = nx.DiGraph()

for _, row in df.iterrows():

    link_type = str(row["Link type i --- j"]).strip()

    if link_type != "-->":
        continue

    var_i = row["Variable i"]
    var_j = row["Variable j"]
    lag = int(row["Time lag of i"])

    source = f"{var_i}_t{lag}"
    target = f"{var_j}_t0"

    G.add_edge(source, target)

if G.number_of_edges() == 0:
    raise ValueError("No directed edges found in CSV.")

# SPRING LAYOUT
n_nodes = G.number_of_nodes()

k = max(5.0, np.sqrt(n_nodes) * 1.5)

pos = nx.spring_layout(
    G,
    seed=42,
    k=k,
    iterations=3000,
    weight=None
)

# FIGURE SIZE
fig_width = max(25, n_nodes * 0.9)
fig_height = max(18, n_nodes * 0.8)

fig, ax = plt.subplots(
    figsize=(fig_width, fig_height)
)

# NODE COLORS
degrees = dict(G.degree())

node_colors = [
    degrees[node]
    for node in G.nodes()
]

# EDGES
nx.draw_networkx_edges(
    G,
    pos,
    ax=ax,
    arrows=True,
    arrowstyle="-|>",
    arrowsize=30,
    width=1.2,
    edge_color="black",
    alpha=0.8,
    connectionstyle="arc3,rad=0.12",
    min_source_margin=20,
    min_target_margin=20
)


# NODES
nx.draw_networkx_nodes(
    G,
    pos,
    ax=ax,
    node_size=1200,      
    node_color=node_colors,
    cmap=plt.cm.viridis,
    edgecolors="white",
    linewidths=1
)

# LABELS
texts = []

for node, (x, y) in pos.items():

    txt = ax.text(
        x,
        y,
        node,
        fontsize=12,      
        ha="center",
        va="center",
        bbox=dict(
            facecolor="white",
            edgecolor="none",
            alpha=0.75,
            pad=0.2
        )
    )

    texts.append(txt)

adjust_text(
    texts,
    ax=ax,
    expand_text=(3.0, 3.0),
    expand_points=(4.0, 4.0),
    force_text=4.0,
    force_points=3.0,
    force_pull=0.05,
    lim=500,
    arrowprops=dict(
        arrowstyle="-",
        color="gray",
        lw=0.4,
        alpha=0.5
    )
)

# EXTRA MARGINS
xs = [p[0] for p in pos.values()]
ys = [p[1] for p in pos.values()]

xpad = (max(xs) - min(xs)) * 0.35
ypad = (max(ys) - min(ys)) * 0.35

ax.set_xlim(min(xs) - xpad, max(xs) + xpad)
ax.set_ylim(min(ys) - ypad, max(ys) + ypad)

ax.set_axis_off()

plt.tight_layout()

# SAVE + SHOW
plt.savefig(
    OUTPUT_FILE,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(f"Saved graph to: {OUTPUT_FILE}")