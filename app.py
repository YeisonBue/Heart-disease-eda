import os
import json
import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from flask import Flask, render_template, request, jsonify

# ── ML ────────────────────────────────────────────────────────────────────────
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report)
from sklearn.pipeline import Pipeline

app = Flask(__name__)

# ── Data ─────────────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "Heart_disease_cleveland_new.csv")
df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
df = df.dropna()
for col in ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal", "target"]:
    df[col] = df[col].astype(int)

# ── Labels ───────────────────────────────────────────────────────────────────
CP_LABELS      = {0: "Angina Típica", 1: "Angina Atípica", 2: "Dolor No Anginoso", 3: "Asintomático"}
THAL_LABELS    = {1: "Flujo Normal", 2: "Defecto Fijo", 3: "Def. Reversible"}
SLOPE_LABELS   = {0: "Ascendente", 1: "Plano", 2: "Descendente"}
RESTECG_LABELS = {0: "Normal", 1: "Anomalía ST-T", 2: "HVI (Estes)"}
SEX_LABELS     = {0: "Femenino", 1: "Masculino"}
TARGET_LABELS  = {0: "Sin Enfermedad", 1: "Enfermedad Cardíaca"}
FBS_LABELS     = {0: "Normal (≤120)", 1: "Alto (>120)"}
EXANG_LABELS   = {0: "No", 1: "Sí"}
CA_LABELS      = {0: "0 vasos", 1: "1 vaso", 2: "2 vasos", 3: "3 vasos"}

# Human-readable variable names for chart builder
VAR_LABELS = {
    "age":      "Edad (años)",
    "sex":      "Sexo",
    "cp":       "Tipo de Dolor Torácico",
    "trestbps": "Presión Arterial Reposo (mmHg)",
    "chol":     "Colesterol (mg/dl)",
    "fbs":      "Glucosa en Ayunas",
    "restecg":  "ECG en Reposo",
    "thalach":  "FC Máxima (bpm)",
    "exang":    "Angina por Ejercicio",
    "oldpeak":  "Depresión ST (Oldpeak)",
    "slope":    "Pendiente Segmento ST",
    "ca":       "Vasos Coloreados",
    "thal":     "Talasemia",
    "target":   "Diagnóstico",
}

CATEGORICAL_VARS = {"sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal", "target"}
NUMERICAL_VARS   = {"age", "trestbps", "chol", "thalach", "oldpeak"}

# Maps for categorical vars (used in chart builder)
CAT_MAPS = {
    "sex": SEX_LABELS, "cp": CP_LABELS, "fbs": FBS_LABELS,
    "restecg": RESTECG_LABELS, "exang": EXANG_LABELS, "slope": SLOPE_LABELS,
    "ca": CA_LABELS, "thal": THAL_LABELS, "target": TARGET_LABELS,
}

FEATURE_COLS = ["age", "sex", "cp", "trestbps", "chol", "fbs",
                "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]

# ── Colours ──────────────────────────────────────────────────────────────────
C_DISEASE    = "#e74c3c"
C_NO_DISEASE = "#3498db"
LAYOUT_BASE  = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", size=12, color="#2c3e50"),
    hoverlabel=dict(bgcolor="white", font_size=12, bordercolor="#dee2e6"),
    margin=dict(l=55, r=35, t=58, b=50),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    autosize=False,
)


def j(fig):
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


def _apply(fig, height=380):
    fig.update_layout(**LAYOUT_BASE, height=height)
    fig.update_xaxes(gridcolor="#f0f3f7", showgrid=True, zeroline=False)
    fig.update_yaxes(gridcolor="#f0f3f7", showgrid=True, zeroline=False)
    return fig


# ── Machine Learning Model ───────────────────────────────────────────────────
X_all = df[FEATURE_COLS]
y_all = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
)

# Evaluate multiple algorithms with 5-fold cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

_candidates = {
    "Random Forest":      RandomForestClassifier(n_estimators=200, max_depth=7,
                                                  min_samples_split=4, random_state=42),
    "Gradient Boosting":  GradientBoostingClassifier(n_estimators=200, learning_rate=0.08,
                                                      max_depth=4, random_state=42),
    "Logistic Regression": Pipeline([
                              ("scaler", StandardScaler()),
                              ("clf",    LogisticRegression(max_iter=1000, C=0.8, random_state=42))
                           ]),
    "SVM":                Pipeline([
                              ("scaler", StandardScaler()),
                              ("clf",    SVC(kernel="rbf", C=1.5, probability=True, random_state=42))
                           ]),
}

_cv_scores = {}
for name, est in _candidates.items():
    scores = cross_val_score(est, X_all, y_all, cv=cv, scoring="accuracy")
    _cv_scores[name] = {"mean": round(scores.mean() * 100, 2),
                        "std":  round(scores.std()  * 100, 2)}

# Best model = highest mean CV accuracy
best_name = max(_cv_scores, key=lambda k: _cv_scores[k]["mean"])
model = _candidates[best_name]
model.fit(X_train, y_train)

# Test-set metrics
y_pred      = model.predict(X_test)
y_prob      = model.predict_proba(X_test)[:, 1]
MODEL_STATS = {
    "name":      best_name,
    "accuracy":  round(accuracy_score(y_test, y_pred)          * 100, 2),
    "precision": round(precision_score(y_test, y_pred)         * 100, 2),
    "recall":    round(recall_score(y_test, y_pred)            * 100, 2),
    "f1":        round(f1_score(y_test, y_pred)                * 100, 2),
    "auc_roc":   round(roc_auc_score(y_test, y_prob)           * 100, 2),
    "cv_scores": _cv_scores,
    "conf_matrix": confusion_matrix(y_test, y_pred).tolist(),
}

# Feature importances (only for tree-based models)
if hasattr(model, "feature_importances_"):
    imp = model.feature_importances_
elif hasattr(model, "named_steps"):
    clf = model.named_steps.get("clf")
    imp = clf.coef_[0] if hasattr(clf, "coef_") else None
else:
    imp = None

FEATURE_IMPORTANCE = (
    sorted(zip(FEATURE_COLS, imp.tolist()), key=lambda x: x[1], reverse=True)
    if imp is not None else []
)


# ── Stats ────────────────────────────────────────────────────────────────────
def get_stats():
    return dict(
        n_rows=len(df),
        n_cols=len(df.columns),
        missing=int(df.isnull().sum().sum()),
        duplicates=int(df.duplicated().sum()),
        disease=int(df["target"].sum()),
        no_disease=int((df["target"] == 0).sum()),
        disease_pct=round(df["target"].mean() * 100, 1),
        male=int((df["sex"] == 1).sum()),
        female=int((df["sex"] == 0).sum()),
        mean_age=round(df["age"].mean(), 1),
        mean_chol=round(df["chol"].mean(), 1),
        mean_thalach=round(df["thalach"].mean(), 1),
    )


# ── Charts ───────────────────────────────────────────────────────────────────
def chart_target_donut():
    d_count = int(df["target"].sum())
    nd_count = len(df) - d_count
    fig = go.Figure(data=[go.Pie(
        labels=["Enfermedad Cardíaca", "Sin Enfermedad"],
        values=[d_count, nd_count],
        hole=0.55,
        marker=dict(colors=[C_DISEASE, C_NO_DISEASE], line=dict(color="white", width=3)),
        textinfo="label+percent",
        hovertemplate="%{label}<br><b>%{value} pacientes</b><br>%{percent}<extra></extra>",
        pull=[0.03, 0],
    )])
    fig.update_layout(
        **LAYOUT_BASE, height=380,
        title=dict(text="Distribución de la Variable Objetivo", font=dict(size=15)),
        annotations=[dict(text=f"<b>{len(df)}</b><br>pacientes", x=0.5, y=0.5,
                          font=dict(size=14), showarrow=False)],
    )
    return j(fig)


def chart_age_distribution():
    fig = go.Figure()
    for t, color, name in [(0, C_NO_DISEASE, "Sin Enfermedad"), (1, C_DISEASE, "Enfermedad Cardíaca")]:
        fig.add_trace(go.Histogram(
            x=df[df["target"] == t]["age"], name=name,
            marker_color=color, opacity=0.72, nbinsx=20,
            hovertemplate="Edad: %{x}<br>Pacientes: %{y}<extra></extra>",
        ))
    fig.update_layout(**LAYOUT_BASE, barmode="overlay",
                      title="Distribución de Edad por Diagnóstico",
                      xaxis_title="Edad (años)", yaxis_title="Número de Pacientes", height=380)
    fig.update_xaxes(gridcolor="#f0f3f7"); fig.update_yaxes(gridcolor="#f0f3f7")
    return j(fig)


def chart_sex_target():
    df2 = df.copy()
    df2["Sexo"] = df2["sex"].map(SEX_LABELS)
    df2["Diagnóstico"] = df2["target"].map(TARGET_LABELS)
    ct = df2.groupby(["Sexo", "Diagnóstico"]).size().reset_index(name="Pacientes")
    fig = px.bar(ct, x="Sexo", y="Pacientes", color="Diagnóstico",
                 color_discrete_map={"Sin Enfermedad": C_NO_DISEASE, "Enfermedad Cardíaca": C_DISEASE},
                 barmode="group", text="Pacientes",
                 title="Distribución por Sexo y Diagnóstico")
    fig.update_traces(textposition="outside")
    _apply(fig)
    return j(fig)


def chart_numerical_distributions():
    features = [
        ("trestbps", "Presión Arterial en Reposo (mmHg)", 1, 1),
        ("chol",     "Colesterol Sérico (mg/dl)",          1, 2),
        ("thalach",  "Frecuencia Cardíaca Máxima (bpm)",   2, 1),
        ("oldpeak",  "Depresión ST — Oldpeak",             2, 2),
    ]
    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=[f[1] for f in features],
                        vertical_spacing=0.16, horizontal_spacing=0.09)
    for feat, _, row, col in features:
        for t, color, name in [(0, C_NO_DISEASE, "Sin Enfermedad"), (1, C_DISEASE, "Enfermedad Cardíaca")]:
            fig.add_trace(
                go.Histogram(x=df[df["target"] == t][feat], name=name,
                             marker_color=color, opacity=0.70, nbinsx=22,
                             showlegend=(row == 1 and col == 1),
                             hovertemplate="%{x}: %{y}<extra></extra>"),
                row=row, col=col)
    fig.update_layout(**LAYOUT_BASE, barmode="overlay",
                      title="Distribuciones de Variables Clínicas Numéricas", height=580)
    fig.update_xaxes(gridcolor="#f0f3f7"); fig.update_yaxes(gridcolor="#f0f3f7")
    return j(fig)


def chart_box_plots():
    cols_info = [
        ("age",      "Edad",         1),
        ("trestbps", "Presión Art.", 2),
        ("chol",     "Colesterol",   3),
        ("thalach",  "FC Máxima",    4),
        ("oldpeak",  "Oldpeak",      5),
    ]
    fig = make_subplots(rows=1, cols=5,
                        subplot_titles=[c[1] for c in cols_info],
                        horizontal_spacing=0.05)
    for feat, _, col in cols_info:
        for t, color, name in [(0, C_NO_DISEASE, "Sin Enfermedad"), (1, C_DISEASE, "Enfermedad Cardíaca")]:
            fig.add_trace(
                go.Box(y=df[df["target"] == t][feat], name=name,
                       marker_color=color, boxmean=True, showlegend=(col == 1)),
                row=1, col=col)
    fig.update_layout(**LAYOUT_BASE, boxmode="group",
                      title="Box Plots de Variables Numéricas por Diagnóstico", height=420)
    fig.update_xaxes(showticklabels=False, gridcolor="#f0f3f7")
    fig.update_yaxes(gridcolor="#f0f3f7")
    return j(fig)


def chart_violin_thalach():
    fig = go.Figure()
    for t, color, name in [(0, C_NO_DISEASE, "Sin Enfermedad"), (1, C_DISEASE, "Enfermedad Cardíaca")]:
        fig.add_trace(go.Violin(
            x=[name] * len(df[df["target"] == t]),
            y=df[df["target"] == t]["thalach"],
            name=name, fillcolor=color,
            box_visible=True, meanline_visible=True,
            line_color="rgba(0,0,0,0.35)", opacity=0.75, points="outliers",
        ))
    fig.update_layout(**LAYOUT_BASE,
                      title="Frecuencia Cardíaca Máxima — Violin Plot",
                      yaxis_title="FC Máxima (bpm)", height=400)
    fig.update_xaxes(gridcolor="#f0f3f7"); fig.update_yaxes(gridcolor="#f0f3f7")
    return j(fig)


def _stacked_bar(column, labels_dict, xlabel, title):
    df2 = df.copy()
    df2["col_label"]    = df2[column].map(labels_dict)
    df2["Diagnóstico"]  = df2["target"].map(TARGET_LABELS)
    ct = df2.groupby(["col_label", "Diagnóstico"]).size().reset_index(name="Pacientes")
    fig = px.bar(ct, x="col_label", y="Pacientes", color="Diagnóstico",
                 color_discrete_map={"Sin Enfermedad": C_NO_DISEASE, "Enfermedad Cardíaca": C_DISEASE},
                 barmode="stack", text="Pacientes",
                 labels={"col_label": xlabel}, title=title)
    fig.update_traces(textposition="inside", insidetextanchor="middle")
    _apply(fig)
    return j(fig)


def chart_cp_target():
    return _stacked_bar("cp", CP_LABELS, "Tipo de Dolor Torácico",
                        "Tipo de Dolor Torácico vs Diagnóstico")

def chart_thal_target():
    return _stacked_bar("thal", THAL_LABELS, "Talasemia", "Talasemia vs Diagnóstico")

def chart_slope_target():
    return _stacked_bar("slope", SLOPE_LABELS, "Pendiente ST",
                        "Pendiente del Segmento ST vs Diagnóstico")

def chart_ca_target():
    return _stacked_bar("ca", {0: "0 vasos", 1: "1 vaso", 2: "2 vasos", 3: "3 vasos"},
                        "Vasos Principales", "Número de Vasos Coloreados vs Diagnóstico")

def chart_restecg_target():
    return _stacked_bar("restecg", RESTECG_LABELS, "Resultado ECG",
                        "ECG en Reposo vs Diagnóstico")

def chart_exang_target():
    return _stacked_bar("exang", {0: "No", 1: "Sí"}, "Angina por Ejercicio",
                        "Angina Inducida por Ejercicio vs Diagnóstico")

def chart_fbs_target():
    return _stacked_bar("fbs", {0: "Normal (≤120)", 1: "Alto (>120)"}, "Glucosa en Ayunas",
                        "Glucosa en Ayunas vs Diagnóstico")


def chart_correlation_heatmap():
    corr = df.corr(numeric_only=True)
    fig = go.Figure(data=go.Heatmap(
        z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
        colorscale="RdBu_r", zmid=0,
        text=np.round(corr.values, 2), texttemplate="%{text}",
        textfont=dict(size=10),
        hovertemplate="%{y} ↔ %{x}<br>r = %{z:.3f}<extra></extra>",
        colorbar=dict(title="r", thickness=14),
    ))
    fig.update_layout(**LAYOUT_BASE, title="Mapa de Calor — Correlación entre Variables", height=560)
    return j(fig)


def chart_correlation_target():
    corr_t = df.corr(numeric_only=True)["target"].drop("target").sort_values()
    colors = [C_DISEASE if c > 0 else C_NO_DISEASE for c in corr_t.values]
    fig = go.Figure(data=[go.Bar(
        x=corr_t.values, y=corr_t.index, orientation="h", marker_color=colors,
        text=[f"{v:+.3f}" for v in corr_t.values], textposition="outside",
        hovertemplate="%{y}: %{x:.3f}<extra></extra>",
    )])
    fig.update_layout(**LAYOUT_BASE,
                      title="Correlación de Variables con la Variable Objetivo",
                      xaxis_title="Coeficiente de Pearson", height=430)
    fig.update_xaxes(range=[-0.65, 0.65], gridcolor="#f0f3f7")
    fig.update_yaxes(gridcolor="#f0f3f7")
    return j(fig)


def chart_scatter_age_thalach():
    df2 = df.copy()
    df2["Diagnóstico"] = df2["target"].map(TARGET_LABELS)
    fig = px.scatter(
        df2, x="age", y="thalach", color="Diagnóstico",
        color_discrete_map={"Sin Enfermedad": C_NO_DISEASE, "Enfermedad Cardíaca": C_DISEASE},
        opacity=0.70,
        labels={"age": "Edad (años)", "thalach": "FC Máxima (bpm)"},
        title="Edad vs Frecuencia Cardíaca Máxima",
        hover_data=["trestbps", "chol"],
        trendline="ols",
    )
    _apply(fig, height=420)
    return j(fig)


def chart_scatter_chol_age():
    df2 = df.copy()
    df2["Diagnóstico"]  = df2["target"].map(TARGET_LABELS)
    df2["oldpeak_size"] = df2["oldpeak"] + 0.8
    fig = px.scatter(
        df2, x="age", y="chol", color="Diagnóstico",
        color_discrete_map={"Sin Enfermedad": C_NO_DISEASE, "Enfermedad Cardíaca": C_DISEASE},
        size="oldpeak_size", size_max=16, opacity=0.65,
        labels={"age": "Edad (años)", "chol": "Colesterol (mg/dl)", "oldpeak_size": "Oldpeak"},
        title="Edad vs Colesterol  (tamaño = Oldpeak)",
        hover_data=["thalach", "trestbps"],
    )
    _apply(fig, height=420)
    return j(fig)


def chart_disease_rate_age():
    df2 = df.copy()
    bins   = [20, 40, 50, 60, 70, 85]
    labels = ["20–40", "40–50", "50–60", "60–70", "70+"]
    df2["Grupo de Edad"] = pd.cut(df2["age"], bins=bins, labels=labels, right=True)
    rate = (df2.groupby("Grupo de Edad", observed=True)["target"]
              .agg(["mean", "count"]).reset_index())
    rate["Tasa (%)"] = (rate["mean"] * 100).round(1)
    colors = [C_DISEASE if r > 50 else C_NO_DISEASE for r in rate["Tasa (%)"]]
    fig = go.Figure(data=[go.Bar(
        x=rate["Grupo de Edad"].astype(str), y=rate["Tasa (%)"],
        marker_color=colors,
        text=[f"{r}%<br>(n={n})" for r, n in zip(rate["Tasa (%)"], rate["count"])],
        textposition="outside",
        hovertemplate="Grupo: %{x}<br>Tasa: %{y:.1f}%<extra></extra>",
    )])
    fig.update_layout(**LAYOUT_BASE,
                      title="Tasa de Enfermedad Cardíaca por Grupo de Edad",
                      xaxis_title="Grupo de Edad", yaxis_title="Tasa (%)",
                      yaxis=dict(range=[0, 100]), height=400)
    fig.update_xaxes(gridcolor="#f0f3f7"); fig.update_yaxes(gridcolor="#f0f3f7")
    return j(fig)


def chart_disease_rate_sex():
    df2 = df.copy()
    df2["Sexo"] = df2["sex"].map(SEX_LABELS)
    rate = df2.groupby("Sexo")["target"].agg(["mean", "count"]).reset_index()
    rate["Tasa (%)"] = (rate["mean"] * 100).round(1)
    colors = [C_DISEASE if r > 50 else C_NO_DISEASE for r in rate["Tasa (%)"]]
    fig = go.Figure(data=[go.Bar(
        x=rate["Sexo"], y=rate["Tasa (%)"],
        marker_color=colors,
        text=[f"{r}%<br>(n={n})" for r, n in zip(rate["Tasa (%)"], rate["count"])],
        textposition="outside",
        hovertemplate="%{x}: %{y:.1f}%<extra></extra>",
    )])
    fig.update_layout(**LAYOUT_BASE,
                      title="Tasa de Enfermedad Cardíaca por Sexo",
                      yaxis_title="Tasa (%)", yaxis=dict(range=[0, 100]), height=380)
    fig.update_xaxes(gridcolor="#f0f3f7"); fig.update_yaxes(gridcolor="#f0f3f7")
    return j(fig)


def chart_disease_rate_cp():
    df2 = df.copy()
    df2["Tipo de Dolor"] = df2["cp"].map(CP_LABELS)
    rate = df2.groupby("Tipo de Dolor")["target"].agg(["mean", "count"]).reset_index()
    rate["Tasa (%)"] = (rate["mean"] * 100).round(1)
    rate = rate.sort_values("Tasa (%)")
    colors = [C_DISEASE if r > 50 else C_NO_DISEASE for r in rate["Tasa (%)"]]
    fig = go.Figure(data=[go.Bar(
        x=rate["Tasa (%)"], y=rate["Tipo de Dolor"], orientation="h",
        marker_color=colors,
        text=[f"{r}%  (n={n})" for r, n in zip(rate["Tasa (%)"], rate["count"])],
        textposition="outside",
        hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
    )])
    fig.update_layout(**LAYOUT_BASE,
                      title="Tasa de Enfermedad por Tipo de Dolor Torácico",
                      xaxis_title="Tasa (%)", xaxis=dict(range=[0, 110]), height=340)
    fig.update_xaxes(gridcolor="#f0f3f7"); fig.update_yaxes(gridcolor="#f0f3f7")
    return j(fig)


def chart_parallel_coords():
    df2 = df.copy()
    fig = px.parallel_coordinates(
        df2, color="target",
        dimensions=["age", "trestbps", "chol", "thalach", "oldpeak", "ca"],
        color_continuous_scale=["#3498db", "#e74c3c"],
        labels={
            "age": "Edad", "trestbps": "Presión", "chol": "Colesterol",
            "thalach": "FC Máx", "oldpeak": "Oldpeak", "ca": "Vasos", "target": "Enfermedad",
        },
        title="Coordenadas Paralelas — Variables Clínicas Clave",
    )
    fig.update_layout(**LAYOUT_BASE, height=430, coloraxis_colorbar=dict(title="Target"))
    return j(fig)


def get_descriptive_stats():
    numerical = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    stats = df[numerical].describe().round(2)
    stats.index = ["Conteo", "Media", "Desv. Est.", "Mínimo", "Q1 (25%)",
                   "Mediana (50%)", "Q3 (75%)", "Máximo"]
    stats.columns = ["Edad", "Presión Art.", "Colesterol", "FC Máxima", "Oldpeak"]
    return stats.to_html(classes="table table-sm table-striped table-hover mb-0",
                         border=0, table_id="stats-table")


def chart_feature_importance():
    if not FEATURE_IMPORTANCE:
        return j(go.Figure())
    names  = [VAR_LABELS.get(f, f) for f, _ in FEATURE_IMPORTANCE]
    values = [abs(v) for _, v in FEATURE_IMPORTANCE]  # abs for LR coefficients
    # Normalize to [0,1] for consistent display
    max_v  = max(values) if values else 1
    norm_v = [v / max_v for v in values]
    colors = [C_DISEASE if i < 3 else ("#f39c12" if i < 6 else C_NO_DISEASE)
              for i in range(len(norm_v))]
    fig = go.Figure(go.Bar(
        x=norm_v[::-1], y=names[::-1], orientation="h",
        marker_color=colors[::-1],
        text=[f"{v*100:.1f}%" for v in norm_v[::-1]],
        textposition="outside",
        hovertemplate="%{y}: %{x:.3f}<extra></extra>",
    ))
    layout = {k: v for k, v in LAYOUT_BASE.items() if k != "margin"}
    fig.update_layout(**layout, height=420,
                      title=f"Importancia de Variables — {MODEL_STATS['name']}",
                      xaxis_title="Importancia Relativa (normalizada)",
                      margin=dict(l=180, r=60, t=58, b=50))
    fig.update_xaxes(gridcolor="#f0f3f7", range=[0, 1.25])
    fig.update_yaxes(gridcolor="#f0f3f7")
    return j(fig)


def chart_confusion_matrix():
    cm = MODEL_STATS["conf_matrix"]
    labels = ["Sin Enfermedad", "Enfermedad Cardíaca"]
    z_text = [[f"<b>{v}</b>" for v in row] for row in cm]
    fig = go.Figure(go.Heatmap(
        z=cm, x=[f"Pred: {l}" for l in labels], y=[f"Real: {l}" for l in labels],
        text=z_text, texttemplate="%{text}", textfont=dict(size=18),
        colorscale=[[0, "#dbeafe"], [1, "#1d4ed8"]],
        showscale=False,
        hovertemplate="Real: %{y}<br>Pred: %{x}<br>N = %{z}<extra></extra>",
    ))
    layout = {k: v for k, v in LAYOUT_BASE.items() if k != "margin"}
    fig.update_layout(**layout,
                      title="Matriz de Confusión (conjunto de prueba)",
                      height=340, margin=dict(l=160, r=50, t=58, b=80))
    return j(fig)


# ── API: Chart Builder ────────────────────────────────────────────────────────
@app.route("/api/chart-builder")
def api_chart_builder():
    x_var      = request.args.get("x", "age")
    y_var      = request.args.get("y", "chol")
    chart_type = request.args.get("chart_type", "scatter")
    color_by   = request.args.get("color_by", "target")

    df2 = df.copy()
    # Apply readable labels to categorical columns
    for col, mapping in CAT_MAPS.items():
        if col in df2.columns:
            df2[col + "_label"] = df2[col].map(mapping)

    x_col = x_var + "_label" if x_var in CAT_MAPS else x_var
    y_col = y_var + "_label" if y_var in CAT_MAPS else y_var
    c_col = color_by + "_label" if color_by in CAT_MAPS else color_by

    x_label = VAR_LABELS.get(x_var, x_var)
    y_label = VAR_LABELS.get(y_var, y_var)
    c_label = VAR_LABELS.get(color_by, color_by)

    color_map = None
    if color_by == "target":
        color_map = {"Sin Enfermedad": C_NO_DISEASE, "Enfermedad Cardíaca": C_DISEASE}

    try:
        if chart_type == "scatter":
            fig = px.scatter(df2, x=x_col, y=y_col, color=c_col,
                             color_discrete_map=color_map, opacity=0.7,
                             labels={x_col: x_label, y_col: y_label, c_col: c_label},
                             title=f"{x_label} vs {y_label}",
                             trendline="ols" if (x_var in NUMERICAL_VARS and y_var in NUMERICAL_VARS) else None)
        elif chart_type == "histogram":
            fig = px.histogram(df2, x=x_col, color=c_col,
                               color_discrete_map=color_map, barmode="overlay", opacity=0.72,
                               labels={x_col: x_label, c_col: c_label},
                               title=f"Histograma de {x_label}")
        elif chart_type == "box":
            fig = px.box(df2, x=c_col, y=y_col if y_var in NUMERICAL_VARS else x_col,
                         color=c_col, color_discrete_map=color_map, points="outliers",
                         labels={c_col: c_label, y_col: y_label, x_col: x_label},
                         title=f"Box Plot: {y_label} por {c_label}")
        elif chart_type == "violin":
            num_col = y_col if y_var in NUMERICAL_VARS else x_col
            num_lbl = y_label if y_var in NUMERICAL_VARS else x_label
            fig = px.violin(df2, x=c_col, y=num_col, color=c_col,
                            color_discrete_map=color_map, box=True, points="outliers",
                            labels={c_col: c_label, num_col: num_lbl},
                            title=f"Violin: {num_lbl} por {c_label}")
        elif chart_type == "bar":
            grp_col = x_col if x_var in CATEGORICAL_VARS else c_col
            cnt = df2.groupby([grp_col, c_col]).size().reset_index(name="Conteo") \
                  if grp_col != c_col else \
                  df2.groupby(grp_col).size().reset_index(name="Conteo")
            if grp_col == c_col:
                cnt.columns = [grp_col, "Conteo"]
                fig = px.bar(cnt, x=grp_col, y="Conteo",
                             labels={grp_col: x_label}, title=f"Distribución de {x_label}")
            else:
                fig = px.bar(cnt, x=grp_col, y="Conteo", color=c_col,
                             color_discrete_map=color_map, barmode="group", text="Conteo",
                             labels={grp_col: x_label, c_col: c_label},
                             title=f"{x_label} vs {c_label}")
                fig.update_traces(textposition="outside")
        else:
            fig = px.scatter(df2, x=x_col, y=y_col, color=c_col,
                             color_discrete_map=color_map, opacity=0.7)

        _apply(fig, height=460)
        return jsonify(json.loads(j(fig)))
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ── API: Predict ──────────────────────────────────────────────────────────────
@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        data = request.get_json()
        features = [float(data.get(col, 0)) for col in FEATURE_COLS]
        X = pd.DataFrame([features], columns=FEATURE_COLS)
        pred = int(model.predict(X)[0])
        prob = float(model.predict_proba(X)[0][1])

        # Risk level based on probability
        if prob < 0.30:
            risk_level, risk_color = "Bajo",  "#27ae60"
        elif prob < 0.60:
            risk_level, risk_color = "Moderado", "#f39c12"
        else:
            risk_level, risk_color = "Alto", "#e74c3c"

        return jsonify({
            "prediction":  pred,
            "label":       "Enfermedad Cardíaca" if pred == 1 else "Sin Enfermedad",
            "probability": round(prob * 100, 1),
            "risk_level":  risk_level,
            "risk_color":  risk_color,
            "model_used":  MODEL_STATS["name"],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ── Route ─────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    stats = get_stats()
    charts = dict(
        target_donut              = chart_target_donut(),
        age_distribution          = chart_age_distribution(),
        sex_target                = chart_sex_target(),
        numerical_distributions   = chart_numerical_distributions(),
        box_plots                 = chart_box_plots(),
        violin_thalach            = chart_violin_thalach(),
        cp_target                 = chart_cp_target(),
        thal_target               = chart_thal_target(),
        slope_target              = chart_slope_target(),
        ca_target                 = chart_ca_target(),
        restecg_target            = chart_restecg_target(),
        exang_target              = chart_exang_target(),
        fbs_target                = chart_fbs_target(),
        correlation_heatmap       = chart_correlation_heatmap(),
        correlation_target        = chart_correlation_target(),
        scatter_age_thalach       = chart_scatter_age_thalach(),
        scatter_chol_age          = chart_scatter_chol_age(),
        disease_rate_age          = chart_disease_rate_age(),
        disease_rate_sex          = chart_disease_rate_sex(),
        disease_rate_cp           = chart_disease_rate_cp(),
        parallel_coords           = chart_parallel_coords(),
        feature_importance        = chart_feature_importance(),
        confusion_matrix          = chart_confusion_matrix(),
    )
    desc_stats_html = get_descriptive_stats()
    missing_values  = int(df.isnull().sum().sum())
    df_head_html    = df.head(10).to_html(
        classes="table table-sm table-hover table-striped mb-0", border=0, index=False)
    return render_template(
        "index.html",
        stats=stats, charts=charts,
        model_stats=MODEL_STATS,
        feature_importance=FEATURE_IMPORTANCE,
        var_labels=VAR_LABELS,
        categorical_vars=list(CATEGORICAL_VARS),
        numerical_vars=list(NUMERICAL_VARS),
        feature_cols=FEATURE_COLS,
        cat_maps=CAT_MAPS,
        desc_stats_html=desc_stats_html,
        missing_values=missing_values,
        df_head_html=df_head_html,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
