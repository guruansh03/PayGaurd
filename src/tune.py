"""
tune.py  --  run after models.py

1. Scores the VALIDATION split with the saved IF / AE / LOF models.
2. Picks, on VAL only:
     - an F1-optimal threshold for each model (grid 0.01 .. 0.99), and
     - ensemble weights (simplex grid, step 0.05) + ensemble threshold.
3. Writes outputs/tuned_params.json  (config.py loads it automatically).
4. Scores the TEST split once and writes outputs/metrics.json with
   ROC-AUC, PR-AUC, precision, recall, F1, flag rate, confusion matrix and
   bootstrap 95% CIs. Also rewrites outputs/anomaly_scores.csv so the
   dashboard shows the tuned ensemble.

The test split is never used for any choice.
"""

import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (ANOMALY_SCORES_PATH, METRICS_PATH, RANDOM_SEED, TUNED_PARAMS_PATH,
                    X_TEST_PATH, X_VAL_PATH, Y_TEST_PATH, Y_VAL_PATH)
from models import score_ae, score_if, score_lof

GRID = np.round(np.arange(0.01, 1.00, 0.01), 2)
KEYS = {'Isolation Forest': 'if', 'Autoencoder': 'ae', 'LOF': 'lof'}


def best_threshold(scores, y):
    """F1-optimal threshold on the 0.01 grid (flag if score >= t). Vectorised via histograms."""
    y = np.asarray(y).astype(bool)
    edges = np.append(GRID, np.inf)
    pos = np.histogram(scores[y], bins=edges)[0][::-1].cumsum()[::-1]    # flagged frauds for each t
    neg = np.histogram(scores[~y], bins=edges)[0][::-1].cumsum()[::-1]   # flagged normals for each t
    P = y.sum()
    prec = np.divide(pos, pos + neg, out=np.zeros(len(GRID)), where=(pos + neg) > 0)
    rec = pos / max(P, 1)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(len(GRID)), where=(prec + rec) > 0)
    i = int(np.argmax(f1))
    return float(GRID[i]), float(f1[i])


def simplex(step=0.05):
    n = int(round(1 / step))
    for a in range(n + 1):
        for b in range(n + 1 - a):
            yield round(a * step, 2), round(b * step, 2), round((n - a - b) * step, 2)


def report(scores, y, t, n_boot=300):
    y = np.asarray(y).astype(int)
    pred = (scores >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    out = {
        'threshold': t,
        'roc_auc': float(roc_auc_score(y, scores)),
        'pr_auc': float(average_precision_score(y, scores)),
        'precision': float(precision_score(y, pred, zero_division=0)),
        'recall': float(recall_score(y, pred, zero_division=0)),
        'f1': float(f1_score(y, pred, zero_division=0)),
        'flag_rate': float(pred.mean()),
        'confusion': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)},
    }
    rng = np.random.default_rng(RANDOM_SEED)
    aucs, f1s = [], []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        aucs.append(roc_auc_score(y[i], scores[i]))
        f1s.append(f1_score(y[i], pred[i], zero_division=0))
    out['roc_auc_ci95'] = [float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))]
    out['f1_ci95'] = [float(np.percentile(f1s, 2.5)), float(np.percentile(f1s, 97.5))]
    return out


def component_scores(X):
    X = X.values.astype(np.float32)
    return {'if': score_if(X), 'ae': score_ae(X), 'lof': score_lof(X)}


def main():
    X_val, y_val = pd.read_csv(X_VAL_PATH), pd.read_csv(Y_VAL_PATH).values.ravel()
    X_test, y_test = pd.read_csv(X_TEST_PATH), pd.read_csv(Y_TEST_PATH).values.ravel()

    print(f"Scoring VAL ({len(X_val):,} rows, {int(y_val.sum())} fraud)...")
    sv = component_scores(X_val)

    thresholds, val_f1 = {}, {}
    for name, k in KEYS.items():
        thresholds[name], val_f1[name] = best_threshold(sv[k], y_val)
        print(f"  {name:17s} VAL best threshold={thresholds[name]:.2f}  F1={val_f1[name]:.3f}")

    best = (-1.0, None, None)
    for a, b, c in simplex(0.05):
        ens = a * sv['if'] + b * sv['ae'] + c * sv['lof']
        t, f = best_threshold(ens, y_val)
        if f > best[0]:
            best = (f, {'if': a, 'ae': b, 'lof': c}, t)
    val_f1['Ensemble'], weights, thresholds['Ensemble'] = best
    print(f"  Ensemble          VAL weights={weights} threshold={thresholds['Ensemble']:.2f}  F1={best[0]:.3f}")

    tuned = {
        'thresholds': thresholds,
        'ensemble_weights': weights,
        'val_f1': val_f1,
        'tuned_on': 'validation split',
        'created_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
    }
    with open(TUNED_PARAMS_PATH, 'w') as f:
        json.dump(tuned, f, indent=2)
    print(f"Saved -> {TUNED_PARAMS_PATH}")

    print(f"\nScoring TEST ({len(X_test):,} rows, {int(y_test.sum())} fraud) -- reported once")
    st = component_scores(X_test)
    st['ensemble'] = weights['if'] * st['if'] + weights['ae'] * st['ae'] + weights['lof'] * st['lof']
    pd.DataFrame({'if_score': st['if'], 'lof_score': st['lof'], 'ae_score': st['ae'],
                  'ensemble_score': st['ensemble']}).to_csv(ANOMALY_SCORES_PATH, index=False)

    metrics = {'split': 'test', 'n_rows': int(len(y_test)), 'n_fraud': int(y_test.sum()), 'models': {}}
    for name, k in list(KEYS.items()) + [('Ensemble', 'ensemble')]:
        m = report(st[k], y_test, thresholds[name])
        metrics['models'][name] = m
        print(f"  {name:17s} ROC-AUC={m['roc_auc']:.4f} PR-AUC={m['pr_auc']:.3f} "
              f"P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} flag={m['flag_rate']*100:.2f}%")
    with open(METRICS_PATH, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved -> {METRICS_PATH}")
    return tuned, metrics


if __name__ == '__main__':
    main()
