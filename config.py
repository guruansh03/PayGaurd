# config.py — single source of truth, imported by all modules

DATA_PATH           = 'data/upi_transactions.csv'
OUTPUTS_DIR         = 'outputs/'
CHARTS_DIR          = 'outputs/charts/'

ANOMALY_SCORES_PATH = 'outputs/anomaly_scores.csv'
X_TEST_PATH         = 'outputs/X_test.csv'
Y_TEST_PATH         = 'outputs/y_test.csv'
X_VAL_PATH          = 'outputs/X_val.csv'              # C-01: val split for threshold/weight tuning
Y_VAL_PATH          = 'outputs/y_val.csv'              # C-01: val split — report final metrics on test only
IF_MODEL_PATH       = 'outputs/isolation_forest.pkl'
IF_SCALER_PATH      = 'outputs/isolation_forest_scaler.pkl'   # FIX 3.1: persist IF score scaler
AE_MODEL_PATH       = 'outputs/autoencoder.pt'
AE_SCALER_PATH      = 'outputs/autoencoder_scaler.pkl'        # FIX 3.1: persist AE feature scaler
AE_MSE_SCALER_PATH  = 'outputs/autoencoder_mse_scaler.pkl'    # FIX-02: persist AE MSE score scaler
SENDER_STATS_PATH   = 'outputs/sender_stats.csv'              # FIX 3.6: persist per-sender averages
RECEIVER_STATS_PATH = 'outputs/receiver_stats.csv'            # IMP-01: receiver graph feature lookup

THRESHOLD     = 0.61    # ensemble default — retune on VAL set after retrain (C-01/C-02)
                        # Per-model optimal thresholds (from score distribution analysis):
                        # IMPORTANT: After retrain, re-derive these from X_val scores, NOT X_test.
IF_THRESHOLD      = 0.73   # IF AUC=0.9358, best F1 at 0.73 — retune on val
AE_THRESHOLD      = 0.27   # AE — retune after C-05 normal-only retrain
# LOF_THRESHOLD defined below with LOF params (C-02)
CONTAMINATION = 'auto' # FIX 3.4: was 0.01 — 'auto' removes supervised contamination hint
RANDOM_SEED   = 42
TEST_SIZE     = 0.2

# Autoencoder
AE_EPOCHS             = 30
AE_BATCH_SIZE         = 256
AE_LR                 = 1e-3
AE_HIDDEN             = [32, 16, 8, 16, 32]  # C-01: bottleneck=8 (42% of 19 dims, was 4/21%)
AE_EARLY_STOP_PATIENCE = 5   # FIX 9.9: stop if val loss doesn't improve for this many epochs

# LOF (C-02: replaced DBSCAN)
LOF_N_NEIGHBORS  = 20           # standard for tabular fraud detection
LOF_ALGORITHM    = 'ball_tree'  # fastest for ~19 dims
LOF_MODEL_PATH   = 'outputs/lof_model.pkl'
LOF_SCALER_PATH  = 'outputs/lof_scaler.pkl'
LOF_THRESHOLD    = 0.72         # placeholder — recalibrate after retrain

# Ensemble weights — retune on VAL set after retrain (C-01/C-02)
# Current values are test-set derived (pre Phase 3). After retrain:
#   1. Score X_val with all 3 models
#   2. Grid-search weights that maximize F1 on X_val
#   3. Report final metrics on X_test only
ENSEMBLE_WEIGHTS = {
    'if':  0.55,   # retune on val
    'ae':  0.35,   # retune on val (C-05 normal-only will change AE AUC)
    'lof': 0.10,   # C-02: was 0.05 for DBSCAN
}
