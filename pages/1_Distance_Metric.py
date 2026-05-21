import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# =========================
# THEME TOGGLE
# =========================

light_mode = st.sidebar.toggle("Light Mode")

if light_mode:
    graph_bg   = "#FFF8F0"
    graph_text = "#6F4E37"
    graph_grid = "#D9C9B5"
    line1      = "#8B5E3C"
    line2      = "#C68E5B"
    line3      = "#A0785A"
    bar_edge   = "#6F4E37"
else:
    graph_bg   = "#0E1117"
    graph_text = "#FFFFFF"
    graph_grid = "#2E3347"
    line1      = "#1f77b4"
    line2      = "#ff7f0e"
    line3      = "#2ca02c"
    bar_edge   = "#FFFFFF"


# =========================
# LOAD CSS
# =========================

def load_css():
    with open("style.css") as f:
        css = f.read()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

def load_dark_css():
    with open("style_dark.css") as f:
        css = f.read()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

if light_mode:
    load_css()
else:
    load_dark_css()


# =========================
# HELPER — apply theme to axes
# =========================

def style_ax(fig, ax):
    fig.patch.set_facecolor(graph_bg)
    ax.set_facecolor(graph_bg)
    ax.tick_params(colors=graph_text)
    ax.xaxis.label.set_color(graph_text)
    ax.yaxis.label.set_color(graph_text)
    ax.title.set_color(graph_text)
    for spine in ax.spines.values():
        spine.set_color(graph_grid)
    ax.grid(True, color=graph_grid, linewidth=0.5)


def style_legend(legend):
    frame = legend.get_frame()
    frame.set_facecolor(graph_bg)
    frame.set_edgecolor(graph_grid)
    for text in legend.get_texts():
        text.set_color(graph_text)


# =========================
# SIDEBAR CONTROLS
# =========================

st.sidebar.header("Metric Settings")

distance_metric = st.sidebar.selectbox(
    "Choose Distance Metric",
    options=["euclidean", "manhattan", "minkowski"],
    index=0
)

if distance_metric == "minkowski":
    p_value = st.sidebar.slider("Minkowski p-value", min_value=1, max_value=5, value=3)
else:
    p_value = 2

k = st.sidebar.slider(
    "Choose K Value",
    min_value=1,
    max_value=25,
    step=2,
    value=5
)

st.sidebar.markdown("---")


# =========================
# LOAD DATASET
# =========================

@st.cache_data
def load_data():
    return pd.read_csv("heart.csv")

df = load_data()

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)


# =========================
# TRAIN CURRENT MODEL
# =========================

model = KNeighborsClassifier(n_neighbors=k, metric=distance_metric, p=p_value)
model.fit(X_train_s, y_train)

train_accuracy = accuracy_score(y_train, model.predict(X_train_s))
test_accuracy  = accuracy_score(y_test,  model.predict(X_test_s))

metric_display = distance_metric.capitalize()
if distance_metric == "minkowski":
    metric_display = f"Minkowski (p={p_value})"


# =========================
# PAGE TITLE
# =========================

st.title("Distance Metric Explorer")
st.write(
    "Compare how Euclidean, Manhattan, and Minkowski distance metrics "
    "affect KNN performance on the heart disease dataset."
)


# =========================
# 1. STAT BOXES
# =========================

stat_html = (
    '<div class="stat-boxes-row">'
    '<div class="stat-box"><div class="stat-label">metric</div><div class="stat-value">' + metric_display + '</div></div>'
    '<div class="stat-box"><div class="stat-label">K value</div><div class="stat-value">' + str(k) + '</div></div>'
    '<div class="stat-box"><div class="stat-label">train accuracy</div><div class="stat-value">' + f"{train_accuracy * 100:.1f}%" + '</div></div>'
    '<div class="stat-box"><div class="stat-label">test accuracy</div><div class="stat-value">' + f"{test_accuracy * 100:.1f}%" + '</div></div>'
    '</div>'
)
st.markdown(stat_html, unsafe_allow_html=True)


# =========================
# OVERFITTING WARNING
# =========================

gap = train_accuracy - test_accuracy

if gap > 0.15:
    st.warning(f"Large gap ({gap:.2%}) between train and test — possible overfitting.")
elif k <= 3:
    st.warning("K is very small — model may be overfitting.")
elif k >= 15:
    st.warning("K is large — model may be underfitting.")
else:
    st.success("Model looks reasonably balanced.")


# =========================
# 2. METRIC COMPARISON GRAPH
#    All 3 metrics across K values,
#    current K highlighted
# =========================

st.subheader("Test Accuracy — All Metrics Across K Values")

k_values = [1, 3, 5, 7, 11, 15, 21]
metrics  = ["euclidean", "manhattan", "minkowski"]
colors   = [line1, line2, line3]

fig, ax = plt.subplots(figsize=(8, 5))
style_ax(fig, ax)

for metric, color in zip(metrics, colors):
    acc_list = []
    for kv in k_values:
        m = KNeighborsClassifier(n_neighbors=kv, metric=metric, p=p_value)
        m.fit(X_train_s, y_train)
        acc_list.append(accuracy_score(y_test, m.predict(X_test_s)))
    label = f"Minkowski (p={p_value})" if metric == "minkowski" else metric.capitalize()
    lw    = 2.5 if metric == distance_metric else 1.2
    alpha = 1.0 if metric == distance_metric else 0.5
    ax.plot(k_values, acc_list, marker="o", label=label, color=color,
            linewidth=lw, alpha=alpha)

ax.axvline(x=k, color=graph_text, linestyle="--", linewidth=1, alpha=0.4, label=f"Current K = {k}")

ax.set_xlabel("K Value")
ax.set_ylabel("Test Accuracy")
ax.set_title("Test Accuracy by Distance Metric")
style_legend(ax.legend())
st.pyplot(fig)
plt.close(fig)


# =========================
# 3. LIVE PERFORMANCE — Bias-variance gap chart
#    for the selected metric (replaces bar chart)
# =========================

st.subheader(f"Live Performance — {metric_display}, K = {k}")

k_range  = list(range(1, 26, 2))
tr_curve = []
te_curve = []

for kv in k_range:
    m = KNeighborsClassifier(n_neighbors=kv, metric=distance_metric, p=p_value)
    m.fit(X_train_s, y_train)
    tr_curve.append(accuracy_score(y_train, m.predict(X_train_s)))
    te_curve.append(accuracy_score(y_test,  m.predict(X_test_s)))

fig2, ax2 = plt.subplots(figsize=(8, 4))
style_ax(fig2, ax2)

ax2.plot(k_range, tr_curve, color=line1, linewidth=2, label="Train accuracy")
ax2.plot(k_range, te_curve, color=line2, linewidth=2, label="Test accuracy")
ax2.fill_between(k_range, tr_curve, te_curve, alpha=0.15, color=line3, label="Bias-variance gap")

# Mark current K
cur_tr = accuracy_score(y_train, model.predict(X_train_s))
cur_te = accuracy_score(y_test,  model.predict(X_test_s))

ax2.axvline(x=k, color=line3, linestyle="--", linewidth=1.5)
ax2.scatter([k], [cur_tr], color=line1, s=80, zorder=5)
ax2.scatter([k], [cur_te], color=line2, s=80, zorder=5)

ax2.annotate(f"Train {cur_tr:.2f}", xy=(k, cur_tr),
             xytext=(k + 0.5, cur_tr + 0.01), color=line1, fontsize=9)
ax2.annotate(f"Test {cur_te:.2f}",  xy=(k, cur_te),
             xytext=(k + 0.5, cur_te - 0.025), color=line2, fontsize=9)

ax2.set_xlabel("K Value")
ax2.set_ylabel("Accuracy")
ax2.set_title(f"Bias-Variance Gap — {metric_display}")
ax2.set_ylim(0.5, 1.05)
style_legend(ax2.legend())
st.pyplot(fig2)
plt.close(fig2)


# =========================
# 4. METRIC COMPARISON TABLE
# =========================

st.subheader("Metric Comparison Table")

rows_html = ""

for metric in metrics:
    best_k, best_acc = None, 0
    for kv in k_values:
        m = KNeighborsClassifier(n_neighbors=kv, metric=metric, p=p_value)
        m.fit(X_train_s, y_train)
        acc = accuracy_score(y_test, m.predict(X_test_s))
        if acc > best_acc:
            best_acc, best_k = acc, kv

    m_best = KNeighborsClassifier(n_neighbors=best_k, metric=metric, p=p_value)
    m_best.fit(X_train_s, y_train)
    tr = accuracy_score(y_train, m_best.predict(X_train_s))
    te = accuracy_score(y_test,  m_best.predict(X_test_s))

    label     = f"Minkowski (p={p_value})" if metric == "minkowski" else metric.capitalize()
    is_active = "metric-active" if metric == distance_metric else ""

    rows_html += (
        "<tr class='" + is_active + "'>"
        "<td class='acc-k'>" + label + "</td>"
        "<td>K = " + str(best_k) + "</td>"
        "<td class='acc-train'>" + f"{tr * 100:.1f}%" + "</td>"
        "<td class='acc-test'>"  + f"{te * 100:.1f}%" + "</td>"
        "<td class='acc-gap'>"   + f"{(tr - te) * 100:.1f}%" + "</td>"
        "</tr>"
    )

table_html = (
    '<div class="acc-table-wrapper"><table class="acc-table">'
    "<thead><tr><th>Metric</th><th>Best K</th>"
    "<th class='acc-train-header'>Train acc</th>"
    "<th class='acc-test-header'>Test acc</th>"
    "<th>Gap</th></tr></thead>"
    "<tbody>" + rows_html + "</tbody>"
    "</table></div>"
)
st.markdown(table_html, unsafe_allow_html=True)


# =========================
# 5. METRIC INFO CARDS
# =========================

st.subheader("About the Metrics")

info_html = (
    '<div class="metric-cards-row">'

    '<div class="metric-card">'
    '<div class="metric-card-title">Euclidean</div>'
    '<div class="metric-card-formula">√ Σ (xᵢ - yᵢ)²</div>'
    '<div class="metric-card-desc">Straight-line distance. Default KNN metric. Works well when features are on similar scales — which yours are after StandardScaler.</div>'
    '</div>'

    '<div class="metric-card">'
    '<div class="metric-card-title">Manhattan</div>'
    '<div class="metric-card-formula">Σ |xᵢ - yᵢ|</div>'
    '<div class="metric-card-desc">Sum of absolute differences. More robust to outliers than Euclidean. Often performs better on medical datasets with skewed features.</div>'
    '</div>'

    '<div class="metric-card">'
    '<div class="metric-card-title">Minkowski</div>'
    '<div class="metric-card-formula">(Σ |xᵢ - yᵢ|ᵖ)^(1/p)</div>'
    '<div class="metric-card-desc">Generalisation of both. p=1 → Manhattan, p=2 → Euclidean. Use the p-value slider to explore values in between.</div>'
    '</div>'

    '</div>'
)
st.markdown(info_html, unsafe_allow_html=True)
