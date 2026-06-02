import pandas as pd
import networkx as nx

def load_scm_graph(path, valid_nodes):
    """
    Загружает граф причинно-следственных связей (SCM) из CSV-файла
    и преобразует его в ориентированный граф NetworkX.

    Используются только направленные связи типа "-->" и только узлы,
    присутствующие в списке valid_nodes.

    Args:
        path: str
            Путь к CSV-файлу с описанием SCM-графа.
        valid_nodes: list
            Список допустимых узлов (переменных), которые есть в данных.

    Returns:
        networkx.DiGraph:
            Ориентированный граф причинно-следственных связей.
    """
    df = pd.read_csv(path)
    df = df[df["Link type i --- j"] == "-->"]

    G = nx.DiGraph()

    for _, row in df.iterrows():
        src, tgt = row["Variable i"], row["Variable j"]

        if src in valid_nodes and tgt in valid_nodes and src != tgt:
            G.add_edge(src, tgt, lag=int(row["Time lag of i"]))

    return G