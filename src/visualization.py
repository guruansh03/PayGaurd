"""
visualization.py
YOUR FILE -- Member 1 (runs after analysis.py)

Produces 5 charts, saved to outputs/charts/:
  1. pca_plot.html        -- PCA 2D scatter, colored by anomaly score
  2. tsne_plot.html       -- t-SNE 2D scatter
  3. score_dist.html      -- Anomaly score histogram (all 3 models)
  4. confusion_matrix.png -- Heatmap
  5. shap_summary.png     -- Top 10 SHAP features bar chart

get_pca_plot() is also called by app.py for inline display.
"""

import os
import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    ANOMALY_SCORES_PATH, X_TEST_PATH, Y_TEST_PATH,
    CHARTS_DIR, THRESHOLD, OUTPUTS_DIR
)


def _load():
    scores = pd.read_csv(ANOMALY_SCORES_PATH)
    X_test = pd.read_csv(X_TEST_PATH)
    y_true = pd.read_csv(Y_TEST_PATH).values.ravel()
    return scores, X_test, y_true


# ─── 1. PCA scatter ──────────────────────────────────────────────────────────

def get_pca_plot(X_test: pd.DataFrame = None, scores: pd.Series = None, threshold: float = None):
    """Returns a Plotly figure. Called by app.py too.
    L-14: threshold is now a parameter -- respects user's sidebar slider."""
    if threshold is None:
        threshold = THRESHOLD
    if X_test is None or scores is None:
        s, X_test, _ = _load()
        scores = s['if_score']

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_test)
    df = pd.DataFrame({'PC1': coords[:, 0], 'PC2': coords[:, 1], 'score': scores})
    df['label'] = (scores >= threshold).map({True: 'Anomaly', False: 'Normal'})

    fig = px.scatter(
        df, x='PC1', y='PC2', color='label',
        color_discrete_map={'Normal': '#4C72B0', 'Anomaly': '#DD4444'},
        opacity=0.5, title='PCA -- Transaction Anomaly Map',
        hover_data={'score': ':.3f'}
    )
    fig.update_traces(marker=dict(size=4))
    return fig


def save_pca_plot():
    fig = get_pca_plot()
    path = os.path.join(CHARTS_DIR, 'pca_plot.html')
    fig.write_html(path)
    print(f"Saved: {path}")


# ─── 2. t-SNE scatter ────────────────────────────────────────────────────────

def get_tsne_plot(X_test: pd.DataFrame = None, scores: pd.Series = None, sample_n=5000, threshold: float = None):
    """L-14: threshold is now a parameter -- respects user's sidebar slider."""
    if threshold is None:
        threshold = THRESHOLD
    if X_test is None or scores is None:
        s, X_test, _ = _load()
        scores = s['if_score']

    # t-SNE is slow -- subsample for speed
    if len(X_test) > sample_n:
        idx = np.random.RandomState(42).choice(len(X_test), sample_n, replace=False)
        X_sub = X_test.iloc[idx]
        s_sub = scores.iloc[idx] if hasattr(scores, 'iloc') else scores[idx]
    else:
        X_sub, s_sub = X_test, scores

    from sklearn.manifold import TSNE
    print(f"Running t-SNE on {len(X_sub)} samples (this takes ~1 min)...")
    coords = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(X_sub)

    df = pd.DataFrame({'x': coords[:, 0], 'y': coords[:, 1], 'score': s_sub})
    df['label'] = (s_sub >= threshold).map({True: 'Anomaly', False: 'Normal'})

    fig = px.scatter(
        df, x='x', y='y', color='label',
        color_discrete_map={'Normal': '#4C72B0', 'Anomaly': '#DD4444'},
        opacity=0.5, title='t-SNE -- Transaction Clusters'
    )
    fig.update_traces(marker=dict(size=4))
    return fig


def save_tsne_plot():
    fig = get_tsne_plot()
    path = os.path.join(CHARTS_DIR, 'tsne_plot.html')
    fig.write_html(path)
    print(f"Saved: {path}")


# ─── 3. Score distribution histogram ─────────────────────────────────────────

def get_score_dist_plot(scores: pd.DataFrame = None):
    if scores is None:
        scores, _, _ = _load()

    fig = go.Figure()
    colors = {'if_score': '#4C3EE8', 'ae_score': '#4CC9F0', 'lof_score': '#F7B731'}
    names  = {'if_score': 'Isolation Forest', 'ae_score': 'Autoencoder', 'lof_score': 'LOF'}

    for col in ['if_score', 'ae_score', 'lof_score']:
        if col in scores.columns:
            fig.add_trace(go.Histogram(
                x=scores[col], name=names[col],
                opacity=0.6, marker_color=colors[col],
                nbinsx=80
            ))

    fig.add_vline(x=THRESHOLD, line_dash='dash', line_color='black',
                  annotation_text=f'Threshold={THRESHOLD}')
    fig.update_layout(
        barmode='overlay',
        title='Anomaly Score Distribution',
        xaxis_title='Anomaly Score',
        yaxis_title='Count'
    )
    return fig


def save_score_dist():
    fig = get_score_dist_plot()
    path = os.path.join(CHARTS_DIR, 'score_dist.html')
    fig.write_html(path)
    print(f"Saved: {path}")


# ─── 4. Confusion matrix ─────────────────────────────────────────────────────

def save_confusion_matrix(score_col='if_score'):
    scores, _, y_true = _load()
    y_pred = (scores[score_col] >= THRESHOLD).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Normal', 'Fraud'],
                yticklabels=['Normal', 'Fraud'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix -- Isolation Forest')
    plt.tight_layout()

    path = os.path.join(CHARTS_DIR, 'confusion_matrix.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


# ─── 5. SHAP summary bar chart ───────────────────────────────────────────────

def save_shap_chart():
    imp_path = os.path.join(OUTPUTS_DIR, 'shap_importance.csv')
    if not os.path.exists(imp_path):
        print(f"Missing {imp_path} -- run analysis.py first")
        return

    imp = pd.read_csv(imp_path).head(10).sort_values('mean_abs_shap')

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(imp['feature'], imp['mean_abs_shap'], color='#4C72B0')
    ax.set_xlabel('Mean |SHAP value|')
    ax.set_title('Top 10 Features -- Isolation Forest (SHAP)')
    plt.tight_layout()

    path = os.path.join(CHARTS_DIR, 'shap_summary.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


# ─── Run all ─────────────────────────────────────────────────────────────────

def run_all():
    os.makedirs(CHARTS_DIR, exist_ok=True)
    save_pca_plot()
    save_tsne_plot()     # slow -- ~1 min
    save_score_dist()
    save_confusion_matrix()
    save_shap_chart()
    print("\nAll charts saved to", CHARTS_DIR)


if __name__ == '__main__':
    run_all()
