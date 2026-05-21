import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

<<<<<<< HEAD
# PAGE CONFIG
st.set_page_config(
    page_title="Dashboard",
    layout="wide"
)
=======
>>>>>>> 96e8b7b0c81e576564dbc2968b5a2ba0d9278dde

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
# SIDEBAR — Model Settings
# =========================

st.sidebar.header("Model Settings")

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
# TRAIN MODEL FOR CURRENT K
# =========================

model = KNeighborsClassifier(n_neighbors=k)
model.fit(X_train_s, y_train)

train_accuracy = accuracy_score(y_train, model.predict(X_train_s))
test_accuracy  = accuracy_score(y_test,  model.predict(X_test_s))


# =========================
# PAGE TITLE
# =========================

st.title("KNN Bias-Variance Tradeoff Dashboard")
st.write("Explore how different K values affect model performance on the Heart Disease dataset.")


# =========================
# 1. DATASET SUMMARY
# =========================

st.subheader("Dataset Summary")

n_samples  = len(df)
n_features = X.shape[1]

stat_html = (
    '<div class="stat-boxes-row">'
    '<div class="stat-box"><div class="stat-label">dataset</div><div class="stat-value">Heart Disease</div></div>'
    '<div class="stat-box"><div class="stat-label">samples</div><div class="stat-value">' + str(n_samples) + '</div></div>'
    '<div class="stat-box"><div class="stat-label">train split</div><div class="stat-value">80%</div></div>'
    '<div class="stat-box"><div class="stat-label">features</div><div class="stat-value">' + str(n_features) + '</div></div>'
    '<div class="stat-box"><div class="stat-label">current K</div><div class="stat-value">' + str(k) + '</div></div>'
    '</div>'
)
st.markdown(stat_html, unsafe_allow_html=True)


# =========================
# 2. DATASET PREVIEW
# =========================

st.subheader("Dataset Preview")

if light_mode:
    st.table(df.head())
else:
    st.dataframe(df.head())


# =========================
# 3. CURRENT K RESULTS
# =========================

st.subheader("Results")

st.write(f"Training Accuracy: {train_accuracy:.4f}")
st.write(f"Testing Accuracy:  {test_accuracy:.4f}")

if k <= 3:
    st.warning("Model may be overfitting — K is too small.")
elif k >= 15:
    st.warning("Model may be underfitting — K is too large.")
else:
    st.success("Model is reasonably balanced.")


# =========================
# 4. ACCURACY COMPARISON GRAPH
#    (static — all K values)
# =========================

k_values  = [1, 3, 5, 7, 11, 15, 21]
train_acc = []
test_acc  = []

for kv in k_values:
    m = KNeighborsClassifier(n_neighbors=kv)
    m.fit(X_train_s, y_train)
    train_acc.append(accuracy_score(y_train, m.predict(X_train_s)))
    test_acc.append(accuracy_score(y_test,  m.predict(X_test_s)))

st.subheader("Accuracy Comparison Graph")

fig, ax = plt.subplots(figsize=(8, 5))
style_ax(fig, ax)

ax.plot(k_values, train_acc, marker="o", label="Training Accuracy", color=line1, linewidth=2)
ax.plot(k_values, test_acc,  marker="o", label="Testing Accuracy",  color=line2, linewidth=2)

# Highlight current K
ax.axvline(x=k, color=line3, linestyle="--", linewidth=1.5, label=f"Current K = {k}")

ax.set_xlabel("K Value")
ax.set_ylabel("Accuracy")
ax.set_title("Effect of K on Bias-Variance Tradeoff")

style_legend(ax.legend())
st.pyplot(fig)
plt.close(fig)


# =========================
# 5. ACCURACY TABLE
# =========================

st.subheader("Accuracy Table")

k_table_values = [1, 3, 5, 11, 21]

def build_row(kv, tr, te, gap_val, status_label, status_cls, is_current):
    highlight = " row-current" if is_current else ""
    return (
        "<tr class='" + highlight + "'>"
        "<td class='acc-k'>K = " + str(kv) + ("  ◀" if is_current else "") + "</td>"
        "<td class='acc-train'>" + str(round(tr * 100, 1)) + "%</td>"
        "<td class='acc-test'>"  + str(round(te * 100, 1)) + "%</td>"
        "<td class='acc-gap'>"   + str(round(gap_val * 100, 1)) + "%</td>"
        "<td class='" + status_cls + "'>" + status_label + "</td>"
        "</tr>"
    )

rows_html = ""

for kv in k_table_values:
    tm = KNeighborsClassifier(n_neighbors=kv)
    tm.fit(X_train_s, y_train)
    tr   = accuracy_score(y_train, tm.predict(X_train_s))
    te   = accuracy_score(y_test,  tm.predict(X_test_s))
    gap  = tr - te

    if kv == 1:
        status_label, status_cls = "Overfitting",    "status-over"
    elif kv == 3:
        status_label, status_cls = "Mild overfit",   "status-mild-over"
    elif kv == 5:
        status_label, status_cls = "&#10003; Best",  "status-best"
    elif kv == 11:
        status_label, status_cls = "Mild underfit",  "status-mild-under"
    else:
        status_label, status_cls = "Underfitting",   "status-under"

    rows_html += build_row(kv, tr, te, gap, status_label, status_cls, kv == k)

table_html = (
    '<div class="acc-table-wrapper"><table class="acc-table">'
    "<thead><tr>"
    "<th>K value</th>"
    "<th class='acc-train-header'>Train acc</th>"
    "<th class='acc-test-header'>Test acc</th>"
    "<th>Gap</th><th>Status</th>"
    "</tr></thead>"
    "<tbody>" + rows_html + "</tbody>"
    "</table></div>"
)
st.markdown(table_html, unsafe_allow_html=True)


# =========================
# 6. LIVE PERFORMANCE — Needle/Gauge style
#    Replaced bar chart with a bias-variance
#    gap line chart focused on current K
# =========================

st.subheader(f"Live Performance — K = {k}")

# Build fine-grained K range for smooth curve
k_range     = list(range(1, 26, 2))
tr_curve    = []
te_curve    = []

for kv in k_range:
    m = KNeighborsClassifier(n_neighbors=kv)
    m.fit(X_train_s, y_train)
    tr_curve.append(accuracy_score(y_train, m.predict(X_train_s)))
    te_curve.append(accuracy_score(y_test,  m.predict(X_test_s)))

fig3, ax3 = plt.subplots(figsize=(8, 4))
style_ax(fig3, ax3)

ax3.plot(k_range, tr_curve, color=line1, linewidth=2, label="Train accuracy")
ax3.plot(k_range, te_curve, color=line2, linewidth=2, label="Test accuracy")

# Shade the gap between curves
ax3.fill_between(k_range, tr_curve, te_curve, alpha=0.15, color=line3, label="Bias-variance gap")

# Current K marker
current_tr = accuracy_score(y_train, model.predict(X_train_s))
current_te = accuracy_score(y_test,  model.predict(X_test_s))

ax3.axvline(x=k, color=line3, linestyle="--", linewidth=1.5)
ax3.scatter([k], [current_tr], color=line1, s=80, zorder=5)
ax3.scatter([k], [current_te], color=line2, s=80, zorder=5)

ax3.annotate(
    f"Train {current_tr:.2f}",
    xy=(k, current_tr),
    xytext=(k + 0.6, current_tr + 0.01),
    color=line1,
    fontsize=9
)
ax3.annotate(
    f"Test {current_te:.2f}",
    xy=(k, current_te),
    xytext=(k + 0.6, current_te - 0.02),
    color=line2,
    fontsize=9
)

ax3.set_xlabel("K Value")
ax3.set_ylabel("Accuracy")
ax3.set_title(f"Bias-Variance Gap — Current K = {k}")
ax3.set_ylim(0.5, 1.05)

style_legend(ax3.legend())
st.pyplot(fig3)
plt.close(fig3)


# =========================
# 7. PROJECT CONCLUSION
# =========================

st.subheader("Project Conclusion")
st.write("""
- Small K values lead to overfitting (high variance, low bias).
- Large K values lead to underfitting (low variance, high bias).
- Moderate K values provide balanced generalization.
- KNN performance strongly depends on choosing the correct K value.
""")
