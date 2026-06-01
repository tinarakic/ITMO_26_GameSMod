import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os
from adjustText import adjust_text  

# CAUSAL IMPORTANCE SCORING
def causal_importance(env, policies):
    """
    Вычисляет оценку причинной важности (causal importance) связей между переменными в среде.

    Для каждой пары "родитель -> целевая переменная" измеряется, насколько
    сильно изменение (обнуление) значения родительской переменной влияет
    на предсказание политики для целевой переменной.

    Алгоритм:
    1. Получает исходное наблюдение из среды.
    2. Сохраняет базовые предсказания политик для всех переменных.
    3. Для каждой целевой переменной:
       - находит еe родительские переменные в графе среды
       - поочерeдно обнуляет каждую родительскую переменную
       - пересчитывает предсказание политики для целевой переменной
       - измеряет среднее абсолютное отклонение от базового предсказания
    4. Возвращает словарь оценок важности причинных связей.

    Args:
        env: среда с графовой структурой зависимостей (env.graph)
             и списком переменных (env.vars)
        policies: словарь политик (policy) для каждой переменной

    Returns:
        dict: {(parent, target): score}, где score отражает
              влияние родительской переменной на целевую
    """
    scores = {}
    vars = env.vars
    obs_orig = env.reset()

    with torch.no_grad():
        baseline_preds = {}
        for v in vars:
            if v in obs_orig:
                action, _ = policies[v].sample(obs_orig[v])
                baseline_preds[v] = action.squeeze().cpu().numpy()

        for tgt in vars:
            parents = list(env.graph.predecessors(tgt))
            for p in parents:
                obs_pert = {k: v.clone() for k,v in obs_orig.items()}
                if p in obs_pert:
                    obs_pert[p] = torch.zeros_like(obs_pert[p])
                pred_pert, _ = policies[tgt].sample(obs_pert[tgt])
                pred_pert = pred_pert.squeeze().cpu().numpy()
                delta = abs(baseline_preds[tgt] - pred_pert).mean()
                scores[(p, tgt)] = delta

    return scores


# PLOT CAUSAL GRAPH
def plot_causal(graph, scores, top_n=None, save_path="causal_importance.png", figsize=(12,8)):
    """
    Визуализирует граф SCM с весами по causal importance и сохраняет изображение.
    """
    G = graph.copy()
    
    for u, v in G.edges():
        G.edges[u, v]['weight'] = scores.get((u, v), 0.0)

    # оставляем top-N
    if top_n:
        sorted_edges = sorted(scores.items(), key=lambda x: -x[1])
        top_edges = {k:v for k,v in sorted_edges[:top_n]}
        for u, v in G.edges():
            if (u,v) not in top_edges:
                G.edges[u, v]['weight'] = 0.0


    n_nodes = G.number_of_nodes()
    k = max(0.5, np.sqrt(n_nodes) * 0.5) 
    pos = nx.spring_layout(G, seed=42, k=k, iterations=300)


    degrees = dict(G.degree())
    node_colors = [degrees[node] for node in G.nodes()]

    fig, ax = plt.subplots(figsize=(max(10, n_nodes*0.5), max(8, n_nodes*0.4)))

    edge_weights = [G.edges[u,v]['weight'] for u,v in G.edges()]
    nx.draw_networkx_edges(
        G,
        pos,
        ax=ax,
        arrows=True,
        arrowstyle='-|>',
        arrowsize=20,
        width=[max(w*5, 0.5) for w in edge_weights],
        edge_color='black',
        alpha=0.8,
        connectionstyle="arc3,rad=0.12"
    )

    nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        node_size=1000,
        node_color=node_colors,
        cmap=plt.cm.viridis,
        edgecolors='white',
        linewidths=1
    )

    texts = []
    for node, (x, y) in pos.items():
        txt = ax.text(
            x, y, node,
            fontsize=10,
            ha='center', va='center',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=0.2)
        )
        texts.append(txt)

    adjust_text(
        texts, ax=ax,
        expand_text=(1.5, 1.5),
        expand_points=(3.0, 3.0),
        force_text=2.0,
        force_points=2.0,
        force_pull=0.05,
        lim=200,
        arrowprops=dict(arrowstyle='-', color='gray', lw=0.4, alpha=0.5)
    )

    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    xpad = (max(xs) - min(xs)) * 0.3
    ypad = (max(ys) - min(ys)) * 0.3
    ax.set_xlim(min(xs)-xpad, max(xs)+xpad)
    ax.set_ylim(min(ys)-ypad, max(ys)+ypad)

    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()