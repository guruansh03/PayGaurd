"""
analysis.py
Runs after models.py.

Computes:
  - Precision, Recall, F1, ROC-AUC per model (including Ensemble)
  - SHAP values for Isolation Forest (TreeExplainer)
  - FIX 9.6: Gradient saliency for Autoencoder (d(MSE)/d(input) via autograd)
  - FIX 9.6: LOF feature importance via permutation-based local density attribution

Saves:
  outputs/shap_values.npy
  outputs/shap_importance.csv
  outputs/ae_saliency.npy
  outputs/ae_importance.csv
  outputs/lof_feature_importance.csv
"""

import os
import sys
import pickle
import numpy as np
from models import ApproximateLOFModel  # needed so pickle can deserialise the LOF model
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    ANOMALY_SCORES_PATH, X_TEST_PATH, Y_TEST_PATH,
    IF_MODEL_PATH, AE_MODEL_PATH, AE_SCALER_PATH,
    AE_HIDDEN, THRESHOLD, OUTPUTS_DIR,
)


def evaluate_model(scores: pd.Series, y_true: np.ndarray, name: str):
    print(f"\n=== {name} ===")
    y_pred = (scores >= THRESHOLD).astype(int)
    print(classification_report(y_true, y_pred, target_names=['Normal', 'Fraud']))
    try:
        auc = roc_auc_score(y_true, scores)
        print(f"ROC-AUC: {auc:.4f}")
    except Exception as e:
        print(f"ROC-AUC: N/A ({e})")
        auc = None
    cm = confusion_matrix(y_true, y_pred)
    print(f"Confusion matrix:\n{cm}")
    return y_pred, auc


def run_shap(X_test: pd.DataFrame):
    print("\n=== SHAP (Isolation Forest) ===")
    try:
        import shap
    except ImportError:
        print("shap not installed — run: pip install shap")
        return None

    with open(IF_MODEL_PATH, 'rb') as f:
        model = pickle.load(f)

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    out_path = os.path.join(OUTPUTS_DIR, 'shap_values.npy')
    np.save(out_path, shap_values)
    print(f"Saved SHAP values → {out_path}")

    importance = pd.DataFrame({
        'feature':       X_test.columns,
        'mean_abs_shap': np.abs(shap_values).mean(axis=0),
    }).sort_values('mean_abs_shap', ascending=False)

    print("\nTop 10 features by SHAP importance:")
    print(importance.head(10).to_string(index=False))
    importance.to_csv(os.path.join(OUTPUTS_DIR, 'shap_importance.csv'), index=False)
    return shap_values, importance


def run_ae_saliency(X_test: pd.DataFrame):
    """
    FIX 9.6: Input-gradient saliency for the Autoencoder.
    d(MSE(AE(x), x))/d(x) via PyTorch autograd.
    Mean abs gradient per feature = importance ranking.
    """
    print("\n=== AE Gradient Saliency (FIX 9.6) ===")
    try:
        import torch
        import torch.nn as nn
        from src.models import Autoencoder
    except ImportError as e:
        print(f"Skipping AE saliency: {e}")
        return None

    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dim = X_test.shape[1]

    with open(AE_SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)

    X_norm = scaler.transform(X_test.values).astype(np.float32)

    ae = Autoencoder(input_dim, AE_HIDDEN).to(device)
    ae.load_state_dict(torch.load(AE_MODEL_PATH, map_location=device,
                                  weights_only=True))
    ae.eval()

    criterion = nn.MSELoss(reduction='mean')
    saliency  = np.zeros_like(X_norm)
    batch_size = 512
    n_batches  = (len(X_norm) + batch_size - 1) // batch_size

    for i, start in enumerate(range(0, len(X_norm), batch_size)):
        batch = torch.tensor(
            X_norm[start:start + batch_size], requires_grad=True
        ).to(device)
        recon = ae(batch)
        loss  = criterion(recon, batch)
        loss.backward()
        saliency[start:start + batch_size] = batch.grad.cpu().detach().numpy()
        if (i + 1) % 20 == 0:
            print(f"  Saliency batch {i+1}/{n_batches}")

    saliency = np.abs(saliency)

    np.save(os.path.join(OUTPUTS_DIR, 'ae_saliency.npy'), saliency)
    print(f"Saved AE saliency → outputs/ae_saliency.npy")

    importance = pd.DataFrame({
        'feature':       X_test.columns,
        'mean_abs_grad': saliency.mean(axis=0),
    }).sort_values('mean_abs_grad', ascending=False)

    importance.to_csv(os.path.join(OUTPUTS_DIR, 'ae_importance.csv'), index=False)
    print(f"Saved AE feature importance → outputs/ae_importance.csv")
    print("\nTop 10 features (AE gradient saliency):")
    print(importance.head(10).to_string(index=False))
    return saliency, importance


def run_lof_importance(X_test: pd.DataFrame):
    """
    B-04: LOF feature importance via permutation-based scoring.
    For each feature, shuffle it and measure the change in LOF anomaly scores.
    Features whose shuffling changes scores most are most important.
    """
    print("\n=== LOF Feature Importance (B-04) ===")
    import pickle as _pkl
    lof_path = os.path.join(OUTPUTS_DIR, 'lof_model.pkl')
    lof_scaler_path = os.path.join(OUTPUTS_DIR, 'lof_scaler.pkl')
    if not os.path.exists(lof_path):
        print("lof_model.pkl not found — run models.py first")
        return None

    with open(lof_path, 'rb') as f:
        lof_model = _pkl.load(f)
    with open(lof_scaler_path, 'rb') as f:
        lof_scaler = _pkl.load(f)

    X_vals = X_test.values.astype(np.float32)
    # Subsample for speed
    rng = np.random.default_rng(42)
    n_sample = min(5000, len(X_vals))
    idx = rng.choice(len(X_vals), n_sample, replace=False)
    X_sub = X_vals[idx]

    raw_base = -lof_model.decision_function(X_sub)
    scores_base = np.clip(lof_scaler.transform(raw_base.reshape(-1,1)).ravel(), 0, 1)

    importances = []
    for j in range(X_sub.shape[1]):
        X_perm = X_sub.copy()
        rng.shuffle(X_perm[:, j])
        raw_perm = -lof_model.decision_function(X_perm)
        scores_perm = np.clip(lof_scaler.transform(raw_perm.reshape(-1,1)).ravel(), 0, 1)
        imp = float(np.mean(np.abs(scores_perm - scores_base)))
        importances.append(imp)

    importance = pd.DataFrame({
        'feature': X_test.columns,
        'mean_dist_contrib': importances,
    }).sort_values('mean_dist_contrib', ascending=False)

    importance.to_csv(os.path.join(OUTPUTS_DIR, 'lof_feature_importance.csv'), index=False)
    print(f"Saved LOF feature importance → outputs/lof_feature_importance.csv")
    print("\nTop 10 features (LOF permutation importance):")
    print(importance.head(10).to_string(index=False))
    return importance


def run_all():
    scores = pd.read_csv(ANOMALY_SCORES_PATH)
    X_test = pd.read_csv(X_TEST_PATH)
    y_true = pd.read_csv(Y_TEST_PATH).values.ravel()

    results = {}
    for col in ['if_score', 'ae_score', 'lof_score', 'ensemble_score']:
        if col in scores.columns:
            name = {
                'if_score':       'Isolation Forest',
                'ae_score':       'Autoencoder',
                'lof_score':      'LOF',
                'ensemble_score': 'Ensemble',
            }[col]
            y_pred, auc = evaluate_model(scores[col], y_true, name)
            results[name] = {'y_pred': y_pred, 'auc': auc}

    shap_out   = run_shap(X_test)
    ae_sal     = run_ae_saliency(X_test)       # FIX 9.6
    lof_imp    = run_lof_importance(X_test)     # B-04: was run_dbscan_importance

    return results, shap_out, ae_sal, lof_imp


if __name__ == '__main__':
    run_all()