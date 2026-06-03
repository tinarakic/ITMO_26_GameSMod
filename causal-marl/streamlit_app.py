import streamlit as st
import torch
import numpy as np
import pandas as pd
from pyvis.network import Network
import streamlit.components.v1 as components
import plotly.graph_objects as go

from scm_graph import load_scm_graph
from policy import Policy
from environment import CausalEnv
from data_utils import denormalize, normalize_data
from causal import causal_importance

# CONFIG
CHECKPOINT_PATH = "best_epoch.pth"
DATA_PATH = "data.csv"
SCM_PATH = "scm.csv"

# PAGE CONFIG
st.set_page_config(
    page_title="Crypto Forecast",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# THEME
st.markdown("""
<style>
header {
    visibility: hidden;
}
            
:root {
    --bg: #0a0f1a;           
    --bg-2: #1a1128;         
    --card: rgba(20, 15, 40, 0.85);
    --card-2: rgba(30, 20, 50, 0.90);
    --line: rgba(255, 122, 122, 0.25);  
    --text: #fefefe;    
    --muted: #f0c0c0;        
    --accent: #ff7a7a;    
    --accent-2: #f6a5c0;
    --accent-3: #fbc2a3;
    --success: #22c55e;
    --warning: #f59e0b;
    --danger: #ef4444;
    --shadow: 0 20px 50px rgba(0, 0, 0, 0.45);
}

html, body, [class*="css"] {
    background: radial-gradient(circle at top, #0f1d35 0%, var(--bg) 50%, #120918 100%);
    color: var(--text);
    font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

.stApp {
    background: radial-gradient(circle at top, #0f1d35 0%, var(--bg) 50%, #120918 100%);
    color: var(--text);
}

.block-container {
    padding: 1.6rem 2rem 2rem !important;
}

h1 {
    font-weight: 900;
    letter-spacing: -0.04em;

    /* gradient text */
    background: linear-gradient(135deg, #ff7a7a, #f6a5c0, #fbc2a3);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;

    /* fallback */
    color: #ff7a7a;

    text-shadow: 0 10px 30px rgba(255, 122, 122, 0.15);
}
                       
h2, h3, h4, h5, h6, p, span, label, div {
    color: var(--text);
}

h1 {
    font-weight: 800;
    letter-spacing: -0.03em;
}

h2 {
    font-weight: 700;
    letter-spacing: -0.02em;
}

.stButton > button {
    background: linear-gradient(135deg, var(--accent), var(--accent-3));
    color: white;
    border: 1px solid rgba(255,122,122,0.4);
    border-radius: 14px;
    padding: 0.65rem 1.1rem;
    font-weight: 700;
    box-shadow: 0 10px 24px rgba(255,122,122,0.32);
}

.stButton > button:hover {
    background: linear-gradient(135deg, var(--accent-2), var(--accent-3));
    border-color: rgba(246,165,192,0.6);
}

/* Remove NumberInput borders completely */
[data-testid="stNumberInput"] {
    border: none !important;
    box-shadow: none !important;
    background: transparent !important;
}

/* Remove BaseWeb input borders */
[data-testid="stNumberInput"] > div,
[data-testid="stNumberInput"] div[data-baseweb="input"] {
    border: none !important;
    box-shadow: none !important;
    background: rgba(38, 39, 48, 0.95) !important;
}

[data-baseweb="tag"] {
    background: rgba(255,122,122,0.18) !important;
    color: #fefefe !important;
    border-radius: 999px !important;
}

[data-testid="stDataFrame"] {
    border-radius: 16px;
    overflow: hidden;
    border: 1px solid rgba(255,122,122,0.15);
}

[data-testid="stDataFrame"] [role="grid"] {
    background: rgba(30, 20, 50, 0.95);
    color: var(--text);
}

[data-testid="stMetric"] {
    background: linear-gradient(180deg, rgba(30,20,50,0.95), rgba(20,15,40,0.92));
    border: 1px solid rgba(255,122,122,0.15);
    border-radius: 16px;
    padding: 1rem;
    box-shadow: var(--shadow);
}

[data-testid="stMetricLabel"], [data-testid="stMetricValue"] {
    color: var(--text) !important;
}

hr {
    border-color: rgba(255,122,122,0.18);
}

/* Plotly container */
div.plotly-container {
    background: transparent !important;
    color: var(--text) !important;
}
</style>
""", unsafe_allow_html=True)


# LOAD DATA + NORMALIZATION
raw_data = pd.read_csv(DATA_PATH)

numeric_data = raw_data.select_dtypes(include=[np.number])
data_norm, scalers = normalize_data(numeric_data)
valid_nodes = list(data_norm.columns)

# LOAD SCM GRAPH 
graph = load_scm_graph(SCM_PATH, valid_nodes)
graph = graph.subgraph(valid_nodes).copy()


scm_df = pd.read_csv(SCM_PATH)
directed_edges = scm_df[scm_df["Link type i --- j"] == "-->"].copy()

# LOAD MODEL
@st.cache_resource
def load_model():
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    policies = {}
    input_sizes = ckpt["input_sizes"]
    for v, state in ckpt["policies"].items():
        model = Policy(
            input_size=input_sizes[v],
            num_agents=1,
            hidden=64,
            horizon=1
        )
        model.load_state_dict(state)
        model.eval()
        policies[v] = model
    return policies, input_sizes


policies, input_sizes = load_model()

# ENVIRONMENT
env = CausalEnv(data_norm, graph)

# HEADER
st.markdown("""
<div style="display:flex; align-items:center; gap:10px;">
    <div style="font-size:38px;">🔮</div>
    <div style="
        font-size:38px;
        font-weight:800;
        background: linear-gradient(135deg, #ff7a7a, #f6a5c0, #fbc2a3);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.03em;
    ">
        Causal-based MARL Cryptocurrency Forecasting Dashboard
    </div>
</div>
""", unsafe_allow_html=True)
st.caption("Interactive dashboard for SCM-based forecasting, displaying structural relationships, predicted trajectories, and causal contribution flows between variables.")


# SCM VISUALIZATION
st.subheader("Structura Causal Model")
net = Network(
    height="550px",
    width="100%",
    bgcolor= "#050b14",
    font_color="#e5eefc",
    directed=True
)
net.barnes_hut()


for n in graph.nodes:
    if "whale" in n:
        color, size = "#facc15", 28
    elif "sentiment" in n:
        color, size = "#22c55e", 22
    elif "exchange" in n:
        color, size = "#60a5fa", 18
    else:
        color, size = "#a78bfa", 14
    net.add_node(n, label=n, title=n, color=color, size=size, font={"size": 18, "color": "#e5eefc"})


for u, v in graph.edges:
    net.add_edge(u, v, color="rgba(96,165,250,0.55)")

net.set_options("""
{
  "layout": {
    "improvedLayout": true
  },
  "interaction": {
    "hover": true,
    "navigationButtons": false
  },
  "physics": {
    "enabled": true
  },
  "edges": {
    "color": {
      "inherit": false
    }
  }
}
""")

net.save_graph("scm_graph.html")

with open("scm_graph.html", "r", encoding="utf-8") as f:
    html = f.read()

html = html.replace("background-color: #ffffff;", "background-color: #050b14 !important;")
html = html.replace("background-color:ffffff;", "background-color: #050b14 !important;")
html = html.replace("body {", "body { background-color: #050b14 !important; color: #050b14 !important; border: none !important; ")

css_patch = """
<style>
html, body {
    margin: 0 !important;
    padding: 0 !important;
    background: #050b14 !important;
    overflow: hidden !important;
}

/* ONLY border lives here */
div.vis-network {
    background: #050b14 !important;
    border: 0px solid #3b82f6 !important;
    border-radius: 0px !important;
    box-shadow: none !important;
}

/* Kill any inner visual framing */
div.vis-network canvas {
    background: #050b14 !important;
    border: none !important;
    outline: none !important;
}

/* Remove extra Pyvis wrapper layers */
.vis-network-outer,
.vis-network canvas,
#mynetwork {
    border: none !important;
    box-shadow: none !important;
    background: #050b14 !important;
    
}
</style>
"""
html = html.replace("</head>", f"{css_patch}</head>")

st.markdown("""
<style>
.graph-divider {
    border-bottom: 1px solid rgba(255,255,255,1);
    margin-top: -12px;
    margin-bottom: 26px;
}
</style>
""", unsafe_allow_html=True)

components.html(html, height=550, scrolling=False)

st.markdown(
    '<div class="graph-divider"></div>',
    unsafe_allow_html=True
)

# FORECAST SETTINGS
st.subheader("Forecast Settings")
steps_ahead = st.number_input("Steps ahead", min_value=1, max_value=20, value=1)
selected_vars = st.multiselect("Choose variables to forecast", list(graph.nodes), default=[list(graph.nodes)[0]])

# FORECAST HELPERS
def to_tensor(obs_v, input_size):
    '''
    Преобразует вектор наблюдений в тензор PyTorch фиксированного размера.

    Args:
        obs_v (list[float] | numpy.ndarray): Входной вектор истории переменной.
        input_size (int): Требуемый фиксированный размер контекста для модели.

    Returns:
        torch.Tensor: Трехмерный тензор типа float32.
    '''
    x = np.array(obs_v, dtype=np.float32).flatten()
    if x.size < input_size:
        x = np.pad(x, (input_size - x.size, 0), "constant")
    elif x.size > input_size:
        x = x[-input_size:]
    x = x.reshape(1, 1, input_size)
    return torch.tensor(x, dtype=torch.float32)


def predict_multi_step(env, policies, input_sizes, steps):
    '''
    Выполняет многошаговый авторегрессионный прогноз для всех переменных среды.

    Алгоритм:
    1. Сбрасывает состояние среды для получения начального контекста.
    2. Итерирует по шагам прогнозирования вперед.
    3. Формирует тензоры истории и прогоняет их через соответствующие политики.
    4. Записывает предсказания в историю и обновляет ими текущий контекст среды.

    Args:
        env (object): Объект среды авторегрессионного симулирования.
        policies (dict[str, torch.nn.Module]): Словарь нейросетевых моделей для каждой переменной.
        input_sizes (dict[str, int]): Словарь размеров входных окон для каждой переменной.
        steps (int): Количество шагов прогноза в будущее.

    Returns:
        pandas.DataFrame: Таблица с хронологией предсказанных значений по всем переменным.
    '''
    obs = env.reset()
    history = []
    for _ in range(steps):
        preds = {}
        for v in env.vars:
            if v not in obs:
                continue
            x = to_tensor(obs[v], input_size=input_sizes[v])
            mu, _, _ = policies[v].forward(x)
            preds[v] = mu.detach().cpu().reshape(-1)[0].item()
        history.append(preds)
        obs.update(preds)
    return pd.DataFrame(history)


# CAUSAL SANKEY
def get_var_color(var):
    '''
    Возвращает HEX-код цвета в зависимости от категории переменной.

    Args:
        var (str): Имя переменной (криптовалюта, сентимент, биржа и т.д.).

    Returns:
        str: Строка с HEX-кодом цвета.
    '''
    v = var.lower()
    if "whale" in v:
        return "#facc15"
    elif "sentiment" in v:
        return "#22c55e"
    elif "exchange" in v:
        return "#60a5fa"
    elif "bitcoin" in v or "btc" in v:
        return "#f97316"
    elif "ethereum" in v or "eth" in v:
        return "#a78bfa"
    elif "chainlink" in v:
        return "#3b82f6"
    elif "tether" in v or "usdt" in v:
        return "#22c55e"
    return "#8b5cf6"


def hex_to_rgba(hex_color, alpha=0.75):
    '''
    Конвертирует HEX-цвет в строковый формат RGBA с заданной прозрачностью.

    Args:
        hex_color (str): Строка HEX-кода (например, '#ffffff').
        alpha (float, optional): Коэффициент прозрачности (0.0–1.0). Defaults to 0.75.

    Returns:
        str: Строка в формате 'rgba(r,g,b,alpha)'.
    '''
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def pretty_label(name):
    '''
    Форматирует техническое имя переменной в читаемую подпись для графика.

    Args:
        name (str): Исходное системное имя переменной.

    Returns:
        str: Отформатированная подпись (валидный HTML/Text для Plotly).
    '''
    text = name.replace("_", " ")
    words = text.split()
    if len(words) > 4:
        middle = len(words) // 2
        return " ".join(words[:middle]) + "<br>" + " ".join(words[middle:])
    return " ".join(w.capitalize() for w in words)


def build_causal_sankey(selected_vars, scm_edges):
    '''
    Строит интерактивную диаграмму Санкей для визуализации весов причинно-следственных связей.

    Алгоритм:
    1. Динамически индексирует уникальные переменные-узлы.
    2. Фильтрует связи и рассчитывает процентный вклад каждого родителя (% от общего влияния).
    3. Формирует массивы источников, целей, весов и кастомных текстов подсказок (hover).
    4. Раскрашивает узлы по категориям, выделяя выбранные целевые переменные базовым цветом.
    5. Настраивает темный стиль графика (layout) для рендеринга.

    Args:
        selected_vars (list[str]): Список целевых зависимых переменных.
        scm_edges (pandas.DataFrame): Датафрейм ребер причинно-следственной модели (SCM) 
            с колонками ['Variable i', 'Variable j', 'Link value'].

    Returns:
        plotly.graph_objects.Figure: Интерактивный объект диаграммы Plotly.
    '''
    labels = []
    node_map = {}

    def get_idx(name):
        if name not in node_map:
            node_map[name] = len(labels)
            labels.append(name)
        return node_map[name]

    source = []
    target = []
    values = []
    link_colors = []
    hover_text = []

    for child in selected_vars:
        child_edges = scm_edges[scm_edges["Variable j"] == child].copy()
        child_edges = child_edges[child_edges["Variable i"] != child_edges["Variable j"]]

        if len(child_edges) == 0:
            continue

        child_edges = child_edges.sort_values("Link value", key=np.abs, ascending=False)
        total_weight = child_edges["Link value"].abs().sum()

        if total_weight == 0:
            continue

        for _, row in child_edges.iterrows():
            parent = row["Variable i"]
            weight = abs(float(row["Link value"]))
            pct = (weight / total_weight) * 100

            source.append(get_idx(parent))
            target.append(get_idx(child))
            values.append(pct)
            link_colors.append(hex_to_rgba(get_var_color(parent), 0.60))
            hover_text.append(
                f"<b>{parent}</b><br> → <b>{child}</b><br><br>Contribution: {pct:.2f}%"
            )

    selected_set = set(selected_vars)
    node_colors = []
    for label in labels:
        if label in selected_set:
            node_colors.append("#3281dc")
        else:
            node_colors.append(get_var_color(label))

    fig = go.Figure(
        go.Sankey(
            arrangement="perpendicular",
            node=dict(
                pad=25,
                thickness=24,
                line=dict(color="rgba(0,0,0,0.15)", width=1),
                label=[pretty_label(x) for x in labels],
                color=node_colors,
                hovertemplate="%{label}"
            ),
            link=dict(
                source=source,
                target=target,
                value=values,
                color=link_colors,
                customdata=hover_text,
                hovertemplate="%{customdata}<extra></extra>"
            )
        )
    )

    fig.update_layout(
        paper_bgcolor="#050b14",
        plot_bgcolor="#050b14",
        font=dict(family="Arial", size=12, color="#e5eefc"),
        height=750,
        margin=dict(l=20, r=20, t=70, b=20)
    )

    return fig

# OUTGOING INFLUENCE SANKEY
def build_downstream_sankey(selected_vars, scm_edges):
    '''
    Строит диаграмму Sankey для визуализации силы исходящего влияния (downstream) переменных.

    Алгоритм:
    1. Итерирует по списку выбранных родительских переменных.
    2. Фильтрует и сортирует по убыванию все дочерние связи для каждого родителя.
    3. Рассчитывает долю влияния каждого ребра в процентах от суммарного исходящего веса.
    4. Формирует массивы индексов, процентных весов, цветов и HTML-подсказок (hover).
    5. Генерирует интерактивную фигуру Plotly с фиксированной темной стилизацией холста.

    Args:
        selected_vars (list[str]): Список исходных (влияющих) переменных для анализа.
        scm_edges (pandas.DataFrame): Датафрейм ребер структурной причинно-следственной модели 
            с колонками ['Variable i', 'Variable j', 'Link value'].

    Returns:
        plotly.graph_objects.Figure: Интерактивный объект диаграммы Plotly.
    '''
    labels = []
    node_map = {}

    def get_idx(name):
        if name not in node_map:
            node_map[name] = len(labels)
            labels.append(name)
        return node_map[name]

    source = []
    target = []
    values = []
    link_colors = []
    hover_text = []

    for parent in selected_vars:

        child_edges = scm_edges[
            scm_edges["Variable i"] == parent
        ].copy()

        child_edges = child_edges[
            child_edges["Variable i"] != child_edges["Variable j"]
        ]

        if len(child_edges) == 0:
            continue

        child_edges = child_edges.sort_values(
            "Link value",
            key=np.abs,
            ascending=False
        )

        total_weight = child_edges["Link value"].abs().sum()

        if total_weight == 0:
            continue

        for _, row in child_edges.iterrows():

            child = row["Variable j"]

            weight = abs(float(row["Link value"]))
            pct = (weight / total_weight) * 100

            source.append(get_idx(parent))
            target.append(get_idx(child))
            values.append(pct)

            link_colors.append(
                hex_to_rgba(get_var_color(parent), 0.60)
            )

            hover_text.append(
                f"<b>{parent}</b><br> → <b>{child}</b><br><br>"
                f"Influence: {pct:.2f}%"
            )

    selected_set = set(selected_vars)

    node_colors = [
        get_var_color(label) for label in labels
    ]

    fig = go.Figure(
        go.Sankey(
            arrangement="perpendicular",
            node=dict(
                pad=25,
                thickness=24,
                line=dict(
                    color="rgba(0,0,0,0.15)",
                    width=1
                ),
                label=[pretty_label(x) for x in labels],
                color=node_colors,
                hovertemplate="%{label}"
            ),
            link=dict(
                source=source,
                target=target,
                value=values,
                color=link_colors,
                customdata=hover_text,
                hovertemplate="%{customdata}<extra></extra>"
            )
        )
    )

    fig.update_layout(
        paper_bgcolor="#050b14",
        plot_bgcolor="#050b14",
        font=dict(
            family="Arial",
            size=12,
            color="#e5eefc"
        ),
        height=750,
        margin=dict(
            l=20,
            r=20,
            t=70,
            b=20
        )
    )

    return fig

# RUN FORECAST
if st.button("Run Forecast"):
    df_norm_forecast = predict_multi_step(env, policies, input_sizes, steps_ahead)

    df = pd.DataFrame()
    for col in df_norm_forecast.columns:
        df[col] = denormalize(torch.tensor(df_norm_forecast[col].values), scalers[col])

    last_date = pd.to_datetime(raw_data['datetime'].max())
    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(hours=1), periods=steps_ahead, freq='h')
    df.index = forecast_dates

    st.subheader("Forecast Output")
    st.dataframe(df, use_container_width=True)

    if selected_vars:
        valid_cols = [c for c in selected_vars if c in df.columns]
        if valid_cols:
            fig = go.Figure()
            colors = ["#60a5fa", "#38bdf8", "#22c55e", "#fbbf24", "#f97316", "#a78bfa"]
            for i, col in enumerate(valid_cols):
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df[col],
                    mode="lines+markers",
                    name=col,
                    line=dict(width=3, color=colors[i % len(colors)]),
                    marker=dict(size=7),
                    hovertemplate="%{y:.4f}<extra></extra>"
                ))
            fig.update_layout(
                title=dict(
                    text="Forecasted Variable Trends",
                    x=0.5,
                    xanchor="center",
                    font=dict(size=18, color="#e5eefc")
                ),
                paper_bgcolor="#050b14",
                plot_bgcolor="#050b14",
                font=dict(color="#e5eefc"),
                margin=dict(l=10, r=10, t=50, b=10),
                legend=dict(orientation="h", yanchor="top", y=-0.25, xanchor="center", x=0.5),
                xaxis=dict(
                    title="Date",
                    tickformat="%Y-%m-%d %H:%M",
                    gridcolor="rgba(96,165,250,0.15)",
                    zeroline=False
                ),
                yaxis=dict(gridcolor="rgba(96,165,250,0.15)", zeroline=False)
            )
            st.plotly_chart(fig, use_container_width=True)

            # INCOMING CAUSAL CONTRIBUTIONS
            st.subheader("Causal Contribution Flow")

            sankey_fig = build_causal_sankey(
                selected_vars,
                directed_edges
            )

            st.plotly_chart(
                sankey_fig,
                use_container_width=True
            )

            # OUTGOING INFLUENCE FLOW
            st.subheader("Causal Impact Flow")

            downstream_fig = build_downstream_sankey(
                selected_vars,
                directed_edges
            )

            st.plotly_chart(
                downstream_fig,
                use_container_width=True
            )

            # IMPACT TABLE
            st.subheader("Causal Impact Table")

            outgoing = directed_edges[
                directed_edges["Variable i"].isin(selected_vars)
            ].copy()

            outgoing = outgoing[
                outgoing["Variable i"] != outgoing["Variable j"]
]

            if len(outgoing):

                impact_table = (
                    outgoing[
                        [
                            "Variable i",
                            "Variable j",
                            "Link value"
                        ]
                    ]
                    .rename(
                        columns={
                            "Variable i": "Source Variable",
                            "Variable j": "Affected Variable",
                            "Link value": "Impact Strength"
                        }
                    )
                    .sort_values(
                        "Impact Strength",
                        key=np.abs,
                        ascending=False
                    )
                )

                st.dataframe(
                    impact_table,
                    use_container_width=True,
                    hide_index=True
                )

            else:
                st.info(
                    "No downstream causal effects found for the selected variables."
                )

# CAUSAL IMPORTANCE GRAPH (NETWORK VISUALIZATION)
def build_causal_importance_graph(scores):
    '''
    Строит интерактивный сетевой граф причинно-следственной важности и возвращает его HTML-код.

    Алгоритм:
        1. Инициализирует направленный граф PyVis с темной темой и физикой Barnes-Hut.
        2. Собирает уникальные узлы из ключей словаря `scores` и регистрирует их в графе.
        3. Нормализует веса связей и добавляет ребра, масштабируя их толщину (width) по силе влияния.
        4. Задает параметры сглаживания ребер и интерактивного hover через JSON-опции.
        5. Сохраняет граф во временный файл, внедряет CSS-патч для бесшовного темного фона и возвращает HTML.

    Args:
        scores (dict[tuple[str, str], float | torch.Tensor]): Словарь, где ключ — кортеж 
            (родитель, потомок), а значение — численная оценка важности связи.

    Returns:
        str: Полный HTML-код страницы с интерактивной визуализацией графа.
    '''
    net = Network(
        height="600px",
        width="100%",
        bgcolor="#050b14",
        font_color="#e5eefc",
        directed=True
    )

    net.barnes_hut()

    nodes = set()

    for (p, t), score in scores.items():
        nodes.add(p)
        nodes.add(t)

    for n in nodes:
        net.add_node(
            n,
            label=n,
            color=get_var_color(n),
            size=18,
            font={"size": 14, "color": "#e5eefc"}
        )

    max_score = max(scores.values()) if len(scores) > 0 else 1.0

    for (p, t), score in scores.items():

        score_val = score.item() if hasattr(score, "item") else float(score)

        width = 1 + 8 * (score_val / max_score if max_score > 0 else 0)

        net.add_edge(
            p,
            t,
            value=float(score_val),
            width=float(width),
            color=hex_to_rgba(get_var_color(p), 0.7),
            title=f"{p} → {t}<br>importance: {score_val:.4f}"
        )

    net.set_options("""
    {
      "physics": {
        "enabled": true
      },
      "interaction": {
        "hover": true
      },
      "edges": {
        "smooth": true
      }
    }
    """)

    net.save_graph("causal_importance.html")

    with open("causal_importance.html", "r", encoding="utf-8") as f:
        html = f.read()

    css_patch = """
    <style>
    html, body {
        margin: 0 !important;
        padding: 0 !important;
        background: #050b14 !important;
    }
    div.vis-network {
        background: #050b14 !important;
        border: none !important;
    }
    </style>
    """

    html = html.replace("</head>", css_patch + "</head>")

    return html

# CAUSAL IMPORTANCE GRAPH VISUALIZATION
st.subheader("Causal Importance Graph")

if st.button("Run Causal Importance Graph"):

    scores = causal_importance(env, policies)

    graph_html = build_causal_importance_graph(scores)

    st.markdown("""
    <style>
    .graph-divider {
        border-bottom: 1px solid rgba(255,255,255,1);
        margin-top: -16px;
        margin-bottom: 26px;
    }
    </style>
    """, unsafe_allow_html=True)

    components.html(graph_html, height=550, scrolling=False)

    st.markdown(
        '<div class="graph-divider"></div>',
        unsafe_allow_html=True
    )