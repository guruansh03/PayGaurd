# config.py — single source of truth, imported by all modules
#
# Thresholds and ensemble weights are TUNED on the validation split by
# src/tune.py and written to outputs/tuned_params.json. When that file exists
# its values override the defaults below, so the dashboard, analysis and tests
# all use the same numbers. Final metrics are reported on the TEST split only
# (outputs/metrics.json).

import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
_p = lambda *parts: os.path.join(ROOT, *parts)

DATA_PATH           = _p('data', 'upi_transactions.csv')
OUTPUTS_DIR         = _p('outputs') + os.sep
CHARTS_DIR          = _p('outputs', 'charts') + os.sep

ANOMALY_SCORES_PATH = _p('outputs', 'anomaly_scores.csv')   # test-split scores
X_TEST_PATH         = _p('outputs', 'X_test.csv')
Y_TEST_PATH         = _p('outputs', 'y_test.csv')
X_VAL_PATH          = _p('outputs', 'X_val.csv')            # tuning split
Y_VAL_PATH          = _p('outputs', 'y_val.csv')
IF_MODEL_PATH       = _p('outputs', 'isolation_forest.pkl')
IF_SCALER_PATH      = _p('outputs', 'isolation_forest_scaler.pkl')
AE_MODEL_PATH       = _p('outputs', 'autoencoder.pt')            # torch checkpoint (training)
AE_WEIGHTS_PATH     = _p('outputs', 'autoencoder_weights.npz')   # numpy weights (inference)
AE_SCALER_PATH      = _p('outputs', 'autoencoder_scaler.pkl')
AE_MSE_SCALER_PATH  = _p('outputs', 'autoencoder_mse_scaler.pkl')
SENDER_STATS_PATH   = _p('outputs', 'sender_stats.csv')
RECEIVER_STATS_PATH = _p('outputs', 'receiver_stats.csv')
TUNED_PARAMS_PATH   = _p('outputs', 'tuned_params.json')
METRICS_PATH        = _p('outputs', 'metrics.json')

CONTAMINATION = 'auto'   # no label-derived contamination hint (unsupervised IF)
RANDOM_SEED   = 42

# Autoencoder
AE_EPOCHS              = 30
AE_BATCH_SIZE          = 256
AE_LR                  = 1e-3
AE_HIDDEN              = [32, 16, 8, 16, 32]   # bottleneck 8 for 23 input features
AE_EARLY_STOP_PATIENCE = 5

# LOF (FAISS approximate kNN)
LOF_N_NEIGHBORS = 20
LOF_MODEL_PATH  = _p('outputs', 'lof_model.pkl')
LOF_SCALER_PATH = _p('outputs', 'lof_scaler.pkl')

# ── Defaults (used only until src/tune.py has run) ───────────────────────────
THRESHOLD        = 0.60   # ensemble
IF_THRESHOLD     = 0.70
AE_THRESHOLD     = 0.30
LOF_THRESHOLD    = 0.70
ENSEMBLE_WEIGHTS = {'if': 0.55, 'ae': 0.35, 'lof': 0.10}
TUNED            = False


def _load_tuned():
    global THRESHOLD, IF_THRESHOLD, AE_THRESHOLD, LOF_THRESHOLD, ENSEMBLE_WEIGHTS, TUNED
    try:
        with open(TUNED_PARAMS_PATH) as f:
            t = json.load(f)
    except (OSError, ValueError):
        return
    th = t.get('thresholds', {})
    THRESHOLD     = float(th.get('Ensemble', THRESHOLD))
    IF_THRESHOLD  = float(th.get('Isolation Forest', IF_THRESHOLD))
    AE_THRESHOLD  = float(th.get('Autoencoder', AE_THRESHOLD))
    LOF_THRESHOLD = float(th.get('LOF', LOF_THRESHOLD))
    w = t.get('ensemble_weights')
    if w and abs(sum(w.values()) - 1.0) < 1e-6:
        ENSEMBLE_WEIGHTS = {k: float(w[k]) for k in ('if', 'ae', 'lof')}
    TUNED = True


_load_tuned()

MODEL_THRESHOLDS = {
    'Isolation Forest': IF_THRESHOLD,
    'Autoencoder':      AE_THRESHOLD,
    'LOF':              LOF_THRESHOLD,
    'Ensemble':         THRESHOLD,
}
