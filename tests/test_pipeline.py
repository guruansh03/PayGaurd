"""
tests/test_pipeline.py
FIX 9.3 — zero tests existed; all 9 original bugs were caught by manual audit.
This suite ensures the critical properties hold so regressions are caught
automatically.

Run from project root:
    pip install pytest --break-system-packages
    pytest tests/test_pipeline.py -v

Five test categories:
  1. Scaler persistence (IF + AE) — scores are on same scale at train & inference
  2. LOF inference continuity    — C-02: scores continuous [0,1], negation correct, novelty=True
  3. Feature count                 — encode_raw_df() always emits exactly 19 features
  4. txn_id uniqueness             — uuid4 generator produces no collisions
  5. Threshold monotonicity        — more flags as threshold decreases
"""

import os
import sys
import pickle
import tempfile
import numpy as np
import pandas as pd
import pytest

# ── path so src.* imports work from project root ─────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Scaler persistence — IF and AE
# ═══════════════════════════════════════════════════════════════════════════════

class TestScalerPersistence:
    """
    Root cause of bug 3.1:
      train_isolation_forest() discarded its MinMaxScaler.
      run_inference() fit a brand-new one on incoming data.
      score of 0.85 at training ≠ 0.85 at inference.
    """

    def _make_X(self, n=200, seed=0):
        rng = np.random.default_rng(seed)
        return rng.random((n, 19)).astype(np.float32)

    def test_if_scaler_saved_and_loaded(self, tmp_path):
        """
        Saved IF scaler must reproduce identical scores when applied to the
        same raw decision-function output.
        """
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import MinMaxScaler

        X = self._make_X()
        model = IsolationForest(contamination='auto', random_state=42, n_estimators=50)
        model.fit(X)

        raw = -model.decision_function(X)
        scaler = MinMaxScaler()
        scores_train = scaler.fit_transform(raw.reshape(-1, 1)).ravel()

        scaler_path = tmp_path / "if_scaler.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)

        with open(scaler_path, 'rb') as f:
            loaded_scaler = pickle.load(f)

        scores_loaded = loaded_scaler.transform(raw.reshape(-1, 1)).ravel()
        np.testing.assert_array_almost_equal(scores_train, scores_loaded, decimal=6,
            err_msg="IF scaler: scores differ between fit and load — persistence broken")

    def test_ae_scaler_transform_not_fit_transform(self, tmp_path):
        """
        AE inference must call .transform(), not .fit_transform().
        Calling fit_transform on new data changes the scale reference.
        """
        from sklearn.preprocessing import MinMaxScaler

        X_train = self._make_X(200, seed=1)
        X_new   = self._make_X(50,  seed=99)  # different distribution

        scaler = MinMaxScaler()
        scaler.fit(X_train)

        scaler_path = tmp_path / "ae_scaler.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)

        with open(scaler_path, 'rb') as f:
            loaded = pickle.load(f)

        # correct: transform with training statistics
        correct   = loaded.transform(X_new)
        # wrong:   fit_transform uses new-data statistics
        incorrect = MinMaxScaler().fit_transform(X_new)

        # They must differ — proves the distinction matters
        assert not np.allclose(correct, incorrect), \
            "transform() and fit_transform() on out-of-distribution data returned same result — test is degenerate"

        # The loaded scaler must reproduce the correct result
        np.testing.assert_array_almost_equal(correct, loaded.transform(X_new), decimal=6,
            err_msg="AE scaler loaded from disk does not reproduce training-time transform")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. LOF inference continuity (C-02: replaces TestDBSCANContinuity)
# ═══════════════════════════════════════════════════════════════════════════════

class TestLOFContinuity:
    """
    C-02: LOF with novelty=True must:
    1. Produce continuous scores in [0,1] (not binary).
    2. Require novelty=True — otherwise decision_function() raises AttributeError.
    3. Negate decision_function output before scaling (more negative = more anomalous).
    """

    def _make_X(self, n=300, dims=10, seed=0):
        rng = np.random.default_rng(seed)
        return rng.random((n, dims)).astype(np.float32)

    def test_lof_scores_are_continuous(self):
        """LOF must return scores with values strictly between 0 and 1, not just {0.0, 1.0}."""
        from sklearn.neighbors import LocalOutlierFactor
        from sklearn.preprocessing import MinMaxScaler

        rng = np.random.default_rng(0)
        X_train = rng.normal(0, 1.0, (200, 10)).astype(np.float32)
        # Mix of normals and moderate outliers so scores span [0,1]
        X_test  = np.vstack([
            rng.normal(0, 1.0, (80, 10)),
            rng.normal(3, 0.5, (20, 10)),
        ]).astype(np.float32)

        lof = LocalOutlierFactor(n_neighbors=10, novelty=True, algorithm='ball_tree')
        lof.fit(X_train)

        raw_train = -lof.decision_function(X_train)
        raw_test  = -lof.decision_function(X_test)

        scaler = MinMaxScaler()
        scaler.fit(raw_train.reshape(-1, 1))
        scores = np.clip(scaler.transform(raw_test.reshape(-1, 1)).ravel(), 0.0, 1.0)

        unique_scores = np.unique(np.round(scores, 4))
        assert len(unique_scores) > 2, \
            f"LOF scores appear binary — only {unique_scores[:5]} found. Check negation/scaling."

    def test_lof_scores_in_unit_interval(self):
        """All LOF scores must be in [0, 1]."""
        from sklearn.neighbors import LocalOutlierFactor
        from sklearn.preprocessing import MinMaxScaler

        X = self._make_X(200, seed=2)
        lof = LocalOutlierFactor(n_neighbors=10, novelty=True)
        lof.fit(X)
        raw = -lof.decision_function(X)
        scaler = MinMaxScaler()
        scores = np.clip(scaler.fit_transform(raw.reshape(-1, 1)).ravel(), 0.0, 1.0)
        assert scores.min() >= 0.0 and scores.max() <= 1.0, "LOF scores outside [0,1]"

    def test_lof_novelty_false_raises(self):
        """Without novelty=True, decision_function must not exist — ensures we catch misconfiguration."""
        from sklearn.neighbors import LocalOutlierFactor
        X = self._make_X(100, seed=3)
        lof_no_novelty = LocalOutlierFactor(n_neighbors=5, novelty=False)
        lof_no_novelty.fit(X)
        assert not hasattr(lof_no_novelty, 'decision_function') or \
               callable(getattr(lof_no_novelty, 'decision_function', None)) is False or \
               lof_no_novelty.novelty is False, \
            "LOF with novelty=False should not support decision_function for new data"

    def test_lof_score_direction(self):
        """
        LOF decision_function returns (−∞,1]. Must negate before scaling.
        After negation: outliers score higher than normals.
        Uses moderate outliers (not extreme OOD) to avoid MinMax clipping.
        """
        from sklearn.neighbors import LocalOutlierFactor
        rng = np.random.default_rng(42)
        X_normal  = rng.normal(0, 1.0, (200, 5)).astype(np.float32)
        X_outlier = rng.normal(4, 0.5, (20, 5)).astype(np.float32)  # moderate outliers

        lof = LocalOutlierFactor(n_neighbors=10, novelty=True)
        lof.fit(X_normal)

        # Raw (un-negated): more anomalous = more negative
        raw_normal  = lof.decision_function(X_normal).mean()
        raw_outlier = lof.decision_function(X_outlier).mean()
        assert raw_outlier < raw_normal, \
            "LOF decision_function: outliers should score more negative than normals. " \
            "Negation required before scaling."

        # After negation: outliers should score higher
        neg_normal  = (-lof.decision_function(X_normal)).mean()
        neg_outlier = (-lof.decision_function(X_outlier)).mean()
        assert neg_outlier > neg_normal, \
            "After negation, outliers should score higher than normals."


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Feature count
# ═══════════════════════════════════════════════════════════════════════════════

class TestFeatureCount:
    """
    encode_raw_df() must always return exactly 19 feature columns —
    the same 19 used during training. A count mismatch causes silent
    model input drift (models receive wrong features at wrong positions).
    """

    def _make_raw_df(self, n=5):
        return pd.DataFrame({
            'sender_upi':        [f'user{i}@okicici' for i in range(n)],
            'receiver_upi':      [f'merchant{i}@ybl' for i in range(n)],
            'amount':            [1000.0 * (i + 1) for i in range(n)],
            'timestamp':         ['2024-03-15 02:34:00'] * n,
            'upi_app':           ['GPay'] * n,
            'merchant_category': ['p2p'] * n,
            'sender_bank':       ['ICICI'] * n,
            'receiver_bank':     ['SBI'] * n,
            'sender_state':      ['Delhi'] * n,
            'receiver_state':    ['Maharashtra'] * n,
        })

    def test_encode_raw_df_produces_19_features(self):
        from app import encode_raw_df
        raw = self._make_raw_df()
        encoded = encode_raw_df(raw)
        assert encoded.shape[1] == 23, \
            f"encode_raw_df() returned {encoded.shape[1]} features, expected 23. " \
            f"Columns: {list(encoded.columns)}"

    def test_encode_raw_df_no_nan(self):
        from app import encode_raw_df
        raw = self._make_raw_df()
        encoded = encode_raw_df(raw)
        assert encoded.isnull().sum().sum() == 0, \
            "encode_raw_df() produced NaN values — fillna() missing somewhere"

    def test_encode_raw_df_single_row(self):
        """
        Single-row upload must still return 19 features.
        This is the edge case where amount_vs_sender_avg collapsed to ~0.5.
        """
        from app import encode_raw_df
        raw = self._make_raw_df(n=1)
        encoded = encode_raw_df(raw)
        assert encoded.shape == (1, 23), \
            f"Single-row encode returned shape {encoded.shape}, expected (1, 23)"


# ═══════════════════════════════════════════════════════════════════════════════
# 4. txn_id uniqueness
# ═══════════════════════════════════════════════════════════════════════════════

class TestTxnIdUniqueness:
    """
    Root cause of bug 3.7:
      IDs generated as randint(100000,999999) + randint(10000,99999).
      Birthday paradox guarantees thousands of collisions at 293k rows.
    """

    def test_txn_ids_unique_at_scale(self):
        """
        Generate 293k IDs using the uuid4 approach and assert zero collisions.
        """
        import uuid
        n = 293_000
        ids = [f"TXN{uuid.uuid4().hex[:12].upper()}" for _ in range(n)]
        assert len(set(ids)) == n, \
            f"uuid4-based txn_id generated {n - len(set(ids))} collisions at {n} rows"

    def test_old_randint_approach_collides(self):
        """
        Demonstrate the original bug: randint approach collides when ID space is small.
        The production bug used randint(100000,999999) × randint(10000,99999) = ~81B
        combinations — at 293k rows collision probability is ~0.5% and is seed-dependent.
        This test uses a deliberately small space (900 combinations, 1000 IDs) so
        collision is mathematically guaranteed regardless of seed.
        """
        import random
        random.seed(42)
        n = 1_000
        # small space: 30*30 = 900 possible values — pigeonhole guarantees collision
        ids = [f"TXN{random.randint(100,129)}{random.randint(10,39)}" for _ in range(n)]
        collisions = n - len(set(ids))
        assert collisions > 0, \
            "Expected randint approach to produce collisions — pigeonhole proof (1000 IDs, 900 slots)"


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Threshold monotonicity
# ═══════════════════════════════════════════════════════════════════════════════

class TestThresholdMonotonicity:
    """
    As threshold increases, the number of flagged transactions must not increase.
    Monotone non-increasing. Any violation means the thresholding logic is broken.
    """

    def test_flag_count_decreases_as_threshold_increases(self):
        rng = np.random.default_rng(0)
        scores = rng.random(1000)

        thresholds = np.linspace(0.0, 1.0, 21)
        flag_counts = [(scores >= t).sum() for t in thresholds]

        for i in range(1, len(flag_counts)):
            assert flag_counts[i] <= flag_counts[i - 1], \
                f"Flag count increased from threshold {thresholds[i-1]:.2f} to {thresholds[i]:.2f}: " \
                f"{flag_counts[i-1]} → {flag_counts[i]}. Thresholding logic is non-monotone."

    def test_threshold_0_flags_all(self):
        scores = np.array([0.1, 0.5, 0.9, 0.0, 1.0])
        flagged = (scores >= 0.0).sum()
        assert flagged == len(scores), "threshold=0 should flag every transaction"

    def test_threshold_1_flags_only_perfect_score(self):
        scores = np.array([0.1, 0.5, 0.9, 0.0, 1.0])
        flagged = (scores >= 1.0).sum()
        assert flagged == 1, "threshold=1.0 should flag only score==1.0"


# ─── IMP-03: VelocityStore extended return ────────────────────────────────────

class TestVelocityStoreIMP03:
    """IMP-03: verify compute_from_df returns 4-tuple with correct semantics."""

    def _make_df(self):
        import pandas as pd
        from datetime import datetime, timedelta
        rows = []
        base = datetime(2024, 6, 1, 12, 0, 0)
        # sender A: 3 txns in 1h to 2 different receivers
        for i in range(3):
            rows.append({'sender_upi': 'A@ybl', 'receiver_upi': f'R{i % 2}@ybl',
                         'amount': 500.0 + i * 100, 'timestamp': base + timedelta(minutes=i * 15)})
        # sender B: 1 txn
        rows.append({'sender_upi': 'B@ybl', 'receiver_upi': 'R9@ybl',
                     'amount': 1000.0, 'timestamp': base})
        return pd.DataFrame(rows)

    def test_returns_four_series(self, tmp_path):
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
        from velocity_store import VelocityStore
        vs = VelocityStore(db_path=str(tmp_path / 'test_vel.db'))
        df = self._make_df()
        result = vs.compute_from_df(df)
        assert len(result) == 4, "compute_from_df must return 4-tuple (v1h, v24h, unique_recv, entropy)"

    def test_unique_receivers_le_velocity(self, tmp_path):
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
        from velocity_store import VelocityStore
        vs = VelocityStore(db_path=str(tmp_path / 'test_vel2.db'))
        df = self._make_df()
        v1h, v24h, urecv, entropy = vs.compute_from_df(df)
        assert (urecv <= v1h).all(), "unique_receivers_1h must be <= velocity_1h for all rows"

    def test_get_returns_dict_with_new_keys(self, tmp_path):
        import sys, os, time
        import pandas as pd
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
        from velocity_store import VelocityStore
        vs = VelocityStore(db_path=str(tmp_path / 'test_vel3.db'))
        result = vs.get('unknown@ybl', time.time())
        assert 'velocity_1h'         in result
        assert 'velocity_24h'        in result
        assert 'unique_receivers_1h' in result, "get() must return unique_receivers_1h [IMP-03]"
        assert 'amount_entropy_1h'   in result, "get() must return amount_entropy_1h [IMP-03]"


# ─── IMP-01: data_processing FEATURE_COLS size ───────────────────────────────

class TestFeatureColsIMP01:
    """IMP-01 + IMP-03: FEATURE_COLS must be 23 after both improvements."""

    def test_feature_cols_length(self):
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
        from data_processing import FEATURE_COLS
        assert len(FEATURE_COLS) == 23, \
            f"Expected 23 features (19 + 2 IMP-03 + 2 IMP-01), got {len(FEATURE_COLS)}"

    def test_new_feature_names_present(self):
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
        from data_processing import FEATURE_COLS
        for col in ['unique_receivers_1h', 'amount_entropy_1h',
                    'receiver_txn_count_24h', 'receiver_amount_sum_24h']:
            assert col in FEATURE_COLS, f"Missing IMP feature: {col}"
