"""
data_processing.py
UPI-specific pipeline:
   1. Load upi_transactions.csv
   2. EDA
   3. Feature selection + scaling
   4. Stratified 70/15/15 split (train/val/test) + 5-fold CV evaluation
   5. Save X_test.csv, y_test.csv, X_val.csv, y_val.csv, sender_stats.csv

Phase 0 fixes (original bugs):
  - Saves per-sender historical average (amount_mean, amount_std) to
    sender_stats.csv so inference can compute amount_vs_sender_avg
    correctly instead of dividing by the current-batch mean (useless
    for single-row uploads).
  - velocity_24h computed from actual data, not velocity_1h x 3 hack.
  - Scaler object saved alongside X_test.

Phase 2 fixes (this version):
  - FIX 9.10: evaluate_cv() runs 5-fold StratifiedKFold on Isolation
    Forest and reports mean +/- std for Precision, Recall, F1, ROC-AUC.
    Gives confidence intervals on all reported metrics.
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import MinMaxScaler

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DATA_PATH, RANDOM_SEED, TEST_SIZE,
    OUTPUTS_DIR, X_TEST_PATH, Y_TEST_PATH, SENDER_STATS_PATH, RECEIVER_STATS_PATH,
    CONTAMINATION, THRESHOLD,
    X_VAL_PATH, Y_VAL_PATH,
)

FEATURE_COLS = [
    'amount', 'amount_log', 'amount_vs_sender_avg', 'is_round_amount',
    'txn_hour', 'txn_day', 'is_weekend', 'is_night',
    'velocity_1h', 'velocity_24h',
    'is_new_device', 'cross_bank', 'cross_state',
    'upi_app_enc', 'merchant_cat_enc',
    'sender_bank_enc', 'receiver_bank_enc',
    'sender_state_enc', 'receiver_state_enc',
    'unique_receivers_1h',    # IMP-03: distinct receivers in last 1h
    'amount_entropy_1h',      # IMP-03: Shannon entropy of 1h amount distribution
    'receiver_txn_count_24h', # IMP-01: inbound txns to this receiver in 24h
    'receiver_amount_sum_24h',# IMP-01: total INR inflow to receiver in 24h
]

SCALE_COLS = [
    'amount', 'amount_log', 'amount_vs_sender_avg',
    'velocity_1h', 'velocity_24h', 'txn_hour', 'txn_day',
    'unique_receivers_1h',     # IMP-03
    'amount_entropy_1h',       # IMP-03
    'receiver_txn_count_24h',  # IMP-01
    'receiver_amount_sum_24h', # IMP-01
]


def load_and_clean():
    print("=== Loading UPI dataset ===")
    df = pd.read_csv(DATA_PATH, parse_dates=['timestamp'])

    print(f"Shape         : {df.shape}")
    print(f"Null values   : {df.isnull().sum().sum()}")
    print(f"Fraud cases   : {df['fraud'].sum():,} ({df['fraud'].mean()*100:.3f}%)")
    print(f"Normal cases  : {(df['fraud'] == 0).sum():,}")
    print(f"\nTop UPI apps:\n{df['upi_app'].value_counts()}")
    print(f"\nTop merchant categories:\n{df['merchant_category'].value_counts()}")
    print(f"\nAmount stats (INR):\n{df['amount'].describe()}")
    print(f"\nVelocity (1h) stats:\n{df['velocity_1h'].describe()}")

    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}\nRun generate_upi_data.py first.")

    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    X = df[FEATURE_COLS].copy()
    y = df['fraud']

    # C-01/C-02: 3-way split -- train/val/test (70/15/15)
    # Tune thresholds and ensemble weights on VAL. Report final metrics on TEST.
    # This prevents overfitting thresholds/weights to the test set.
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y,
        test_size=0.15,
        random_state=RANDOM_SEED,
        stratify=y,
    )
    # Split remaining 85% into train (70%) and val (15%)
    val_frac = 0.15 / 0.85  # ~17.6% of train_full -> 15% of total
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=val_frac,
        random_state=RANDOM_SEED,
        stratify=y_train_full,
    )

    # FIX L-04: save per-sender stats from TRAINING data ONLY (after split).
    # Previously computed on full dataset, leaking test-set amount distributions.
    train_indices_for_stats = X_train.index
    sender_stats = (
        df.loc[train_indices_for_stats].groupby('sender_upi')['amount']
        .agg(amount_mean='mean', amount_std='std', txn_count='count')
        .reset_index()
    )
    sender_stats.to_csv(SENDER_STATS_PATH, index=False)
    print(f"Saved sender stats -> {SENDER_STATS_PATH}  ({len(sender_stats):,} senders, training-only)")

    # IMP-01: save receiver stats from TRAINING data only
    if 'receiver_txn_count_24h' in df.columns and 'receiver_amount_sum_24h' in df.columns:
        rec_stats = (
            df.loc[train_indices_for_stats].groupby('receiver_upi')[['receiver_txn_count_24h', 'receiver_amount_sum_24h']]
            .agg(['mean', 'max'])
            .reset_index()
        )
        rec_stats.columns = ['receiver_upi',
                              'receiver_txn_count_24h_mean', 'receiver_txn_count_24h_max',
                              'receiver_amount_sum_24h_mean', 'receiver_amount_sum_24h_max']
        rec_stats.to_csv(RECEIVER_STATS_PATH, index=False)
        print(f"Saved receiver stats -> {RECEIVER_STATS_PATH}  ({len(rec_stats):,} receivers, training-only)")
    else:
        print("Warning: receiver graph columns missing -- run generate_upi_data.py (IMP-01 version) first.")

    # C-06: Recompute amount_vs_sender_avg from TRAINING data only.
    # The raw dataset computes this over ALL rows (train+test), leaking test info into training.
    # Fix: compute sender averages from training split, then map to val+test.
    train_indices = X_train.index
    train_sender_avg = df.loc[train_indices].groupby('sender_upi')['amount'].mean()
    overall_avg = df.loc[train_indices, 'amount'].mean()

    for split_X in [X_train, X_val, X_test]:
        split_senders = df.loc[split_X.index, 'sender_upi']
        sender_avg_mapped = split_senders.map(train_sender_avg).fillna(overall_avg)
        split_X['amount_vs_sender_avg'] = df.loc[split_X.index, 'amount'].values / (sender_avg_mapped.values + 1)

    # Scale features AFTER the leakage fix
    scaler = StandardScaler()
    X_train[SCALE_COLS] = scaler.fit_transform(X_train[SCALE_COLS])
    X_val[SCALE_COLS]   = scaler.transform(X_val[SCALE_COLS])
    X_test[SCALE_COLS]  = scaler.transform(X_test[SCALE_COLS])

    # Save feature scaler
    scaler_path = os.path.join(OUTPUTS_DIR, 'feature_scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)

    print(f"\nTrain size   : {X_train.shape[0]:,}")
    print(f"Val size     : {X_val.shape[0]:,}")
    print(f"Test size    : {X_test.shape[0]:,}")
    print(f"Fraud in train: {y_train.sum()} ({y_train.mean()*100:.3f}%)")
    print(f"Fraud in val  : {y_val.sum()} ({y_val.mean()*100:.3f}%)")
    print(f"Fraud in test : {y_test.sum()} ({y_test.mean()*100:.3f}%)")

    X_test.to_csv(X_TEST_PATH, index=False)
    y_test.to_csv(Y_TEST_PATH, index=False)
    X_val.to_csv(X_VAL_PATH, index=False)
    y_val.to_csv(Y_VAL_PATH, index=False)
    print(f"Saved: {X_TEST_PATH}, {Y_TEST_PATH}")
    print(f"Saved: {X_VAL_PATH}, {Y_VAL_PATH}")

    # FIX 9.1: build SQLite velocity store from full training data
    try:
        from velocity_store import VelocityStore
        store = VelocityStore()
        store.build_from_training(df)
        print(f"Velocity store built -> outputs/velocity.db")
    except Exception as e:
        print(f"Warning: velocity store build failed ({e}) -- inference will fall back to sender_stats")

    return X_train, X_test, y_train, y_test, X_val, y_val


# ── 5-fold cross-validation evaluation (FIX 9.10) ────────────────────────────

def evaluate_cv(X, y, n_splits=5, threshold=None):
    """
    C-08: 5-fold StratifiedKFold for IF and LOF.
    C-09: Score scaler fitted on TRAINING fold, transforms VAL fold.
    FIX-07: threshold parameterized.
    """
    if threshold is None:
        threshold = THRESHOLD

    from sklearn.neighbors import LocalOutlierFactor

    for model_name in ['Isolation Forest', 'LOF']:
        print(f"\n=== 5-Fold Stratified CV -- {model_name} ===")
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)

        precisions, recalls, f1s, aucs = [], [], [], []

        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), start=1):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # Scale inside fold to avoid leakage
            fold_scaler = MinMaxScaler()
            X_tr_s  = fold_scaler.fit_transform(X_tr)
            X_val_s = fold_scaler.transform(X_val)

            if model_name == 'Isolation Forest':
                model = IsolationForest(contamination=CONTAMINATION,
                                        random_state=RANDOM_SEED, n_estimators=100, n_jobs=-1)
                model.fit(X_tr_s)
                raw_train_scores = -model.decision_function(X_tr_s)
                raw_val_scores   = -model.decision_function(X_val_s)
            else:  # LOF
                # Evaluate LOF on the full training fold.
                X_lof_fit = X_tr_s
                model = LocalOutlierFactor(
                    n_neighbors=20, algorithm='ball_tree', novelty=True, n_jobs=-1)
                model.fit(X_lof_fit)
                raw_train_scores = -model.decision_function(X_lof_fit)
                raw_val_scores   = -model.decision_function(X_val_s)

            # C-09: Fit score scaler on TRAINING fold scores, transform val fold
            score_scaler = MinMaxScaler()
            score_scaler.fit(raw_train_scores.reshape(-1, 1))
            scores_norm = np.clip(score_scaler.transform(raw_val_scores.reshape(-1, 1)).ravel(), 0.0, 1.0)
            y_pred = (scores_norm >= threshold).astype(int)

            p  = precision_score(y_val, y_pred, zero_division=0)
            r  = recall_score(y_val, y_pred, zero_division=0)
            f  = f1_score(y_val, y_pred, zero_division=0)
            try:
                a = roc_auc_score(y_val, scores_norm)
            except Exception:
                a = float('nan')

            precisions.append(p); recalls.append(r); f1s.append(f); aucs.append(a)
            print(f"  Fold {fold}: Precision={p:.3f}  Recall={r:.3f}  F1={f:.3f}  AUC={a:.4f}")

        print(f"\n  MEAN  Precision={sum(precisions)/len(precisions):.3f}"
              f"  Recall={sum(recalls)/len(recalls):.3f}"
              f"  F1={sum(f1s)/len(f1s):.3f}"
              f"  AUC={sum(aucs)/len(aucs):.4f}")
        print(f"  STDEV Precision={float(pd.Series(precisions).std()):.3f}"
              f"  Recall={float(pd.Series(recalls).std()):.3f}"
              f"  F1={float(pd.Series(f1s).std()):.3f}"
              f"  AUC={float(pd.Series(aucs).std()):.4f}")
    return {'precision': precisions, 'recall': recalls, 'f1': f1s, 'auc': aucs}


if __name__ == '__main__':
    X_train, X_test, y_train, y_test, X_val, y_val = load_and_clean()
    print("\nData processing complete.")

    # Run CV on the full labelled dataset (labels used for eval only, not training)
    full_df = pd.read_csv(DATA_PATH)
    from sklearn.preprocessing import StandardScaler as SS
    X_all = full_df[FEATURE_COLS].copy()
    ss = SS()
    X_all[SCALE_COLS] = ss.fit_transform(X_all[SCALE_COLS])
    y_all = full_df['fraud']
    evaluate_cv(X_all, y_all)
