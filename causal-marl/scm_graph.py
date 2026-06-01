import pandas as pd
import networkx as nx

def load_scm_graph(path, valid_nodes):
    df = pd.read_csv(path)
    df = df[df["Link type i --- j"] == "-->"]

    G = nx.DiGraph()

    for _, row in df.iterrows():
        src, tgt = row["Variable i"], row["Variable j"]

        if src in valid_nodes and tgt in valid_nodes and src != tgt:
            G.add_edge(src, tgt, lag=int(row["Time lag of i"]))

    return G