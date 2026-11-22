"""
Regression tests for the Nov-2026 audit fixes.

  1. Timestamp -> Unix seconds is resolution-safe (pandas 3 stores datetimes in us/s).
  2. Velocity windows really are 1h / 24h.
  3. encode_raw_df() emits features in the TRAINING column order.
  4. Approximate LOF matches sklearn's exact novelty LOF on new points
     (the nearest neighbour is no longer dropped for query points).
  5. LOF save/load round-trip is exact and pickle-path independent.
  6. NumPy autoencoder forward pass == torch forward pass.
  7. Tuned thresholds / weights are loaded by config and are sane.
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))


# 1 ───────────────────────────────────────────────────────────────────────────
def test_to_unix_seconds_resolution_safe():
    from velocity_store import to_unix_seconds
    s = pd.Series(['2024-01-01 01:00:00', '2024-01-01 01:00:00.123456', None])
    out = to_unix_seconds(s)
    assert out[0] == 1704070800          # exact epoch seconds, not /1000
    assert out[1] == 1704070800
    assert out[2] == 0                    # unparseable -> 0
    for unit in ('s', 'ms', 'us', 'ns'):
        ts = pd.Series(pd.to_datetime(['2024-01-01 01:00:00'])).astype(f'datetime64[{unit}]')
        assert to_unix_seconds(ts)[0] == 1704070800, unit


# 2 ───────────────────────────────────────────────────────────────────────────
def test_velocity_windows_are_real_hours(tmp_path):
    from velocity_store import VelocityStore
    df = pd.DataFrame({
        'sender_upi': ['a'] * 4,
        'receiver_upi': ['r1', 'r2', 'r3', 'r4'],
        'amount': [100.0, 200.0, 300.0, 400.0],
        'timestamp': ['2024-01-01 10:00:00', '2024-01-01 10:10:00',
                      '2024-01-01 10:20:00', '2024-01-03 10:20:00'],   # last one 2 days later
    })
    store = VelocityStore(db_path=str(tmp_path / 'v.db'))
    v1h, v24h, urecv, _ = store.compute_from_df(df)
    assert v1h.tolist() == [0, 1, 2, 0]
    assert v24h.tolist() == [0, 1, 2, 0]
    assert urecv.tolist() == [0, 1, 2, 0]


@pytest.mark.skipif(not os.path.exists(os.path.join(ROOT, 'data', 'upi_transactions.csv')),
                    reason='dataset not generated')
def test_generated_velocity_is_plausible():
    df = pd.read_csv(os.path.join(ROOT, 'data', 'upi_transactions.csv'),
                     usecols=['velocity_1h', 'fraud'])
    normal = df[df.fraud == 0].velocity_1h
    # ~290k txns / 15k senders / 1 year -> a normal sender almost never has 2+ txns in an hour
    assert normal.mean() < 0.05
    assert df[df.fraud == 1].velocity_1h.max() >= 4     # burst pattern exists


# 3 ───────────────────────────────────────────────────────────────────────────
def test_encode_raw_df_uses_training_column_order():
    from app import encode_raw_df
    from data_processing import FEATURE_COLS
    raw = pd.DataFrame({
        'sender_upi': ['u1@ybl', 'u2@ybl'], 'receiver_upi': ['m1@ybl', 'm2@ybl'],
        'amount': [500.0, 90000.0], 'timestamp': ['2024-03-15 02:34:00'] * 2,
        'upi_app': ['GPay'] * 2, 'merchant_category': ['p2p'] * 2,
        'sender_bank': ['ICICI'] * 2, 'receiver_bank': ['SBI'] * 2,
        'sender_state': ['Delhi'] * 2, 'receiver_state': ['Kerala'] * 2,
    })
    enc = encode_raw_df(raw)
    assert list(enc.columns) == list(FEATURE_COLS)


# 4 ───────────────────────────────────────────────────────────────────────────
def _block_faiss(monkeypatch):
    monkeypatch.setitem(sys.modules, 'faiss', None)     # import faiss -> ImportError


def test_lof_matches_sklearn_exact_on_new_points(monkeypatch):
    _block_faiss(monkeypatch)
    from sklearn.neighbors import LocalOutlierFactor
    from models import _lof_scores_from_knn
    rng = np.random.default_rng(0)
    X_fit = rng.normal(size=(400, 5)).astype(np.float32)
    X_new = np.vstack([rng.normal(size=(50, 5)), rng.normal(6, 1, size=(5, 5))]).astype(np.float32)

    _, ours, _, _ = _lof_scores_from_knn(X_fit, X_new, k=10)
    ref = -LocalOutlierFactor(n_neighbors=10, novelty=True).fit(X_fit).score_samples(X_new)
    np.testing.assert_allclose(ours, ref, rtol=1e-4)
    assert ours[-5:].min() > ours[:50].max()            # far points are the outliers


# 5 ───────────────────────────────────────────────────────────────────────────
def test_lof_save_load_roundtrip(tmp_path, monkeypatch):
    _block_faiss(monkeypatch)
    from models import ApproximateLOFModel, _make_knn, _fit_lof_stats, save_lof_model, load_lof_model
    rng = np.random.default_rng(1)
    X = rng.normal(size=(300, 4)).astype(np.float32)
    knn, _ = _make_knn(X)
    k_dist, lrd, _, _ = _fit_lof_stats(knn, X, 8)
    m = ApproximateLOFModel(8, lrd, k_dist, X_fit=X)
    p = tmp_path / 'lof.pkl'
    save_lof_model(m, str(p))
    m2 = load_lof_model(str(p))
    Xq = rng.normal(size=(20, 4)).astype(np.float32)
    np.testing.assert_array_equal(m.decision_function(Xq), m2.decision_function(Xq))


def test_lof_faiss_roundtrip_if_available(tmp_path):
    pytest.importorskip('faiss')
    import faiss
    from models import ApproximateLOFModel, _make_knn, _fit_lof_stats, save_lof_model, load_lof_model
    rng = np.random.default_rng(2)
    X = rng.normal(size=(2000, 6)).astype(np.float32)
    knn, index = _make_knn(X)
    k_dist, lrd, _, _ = _fit_lof_stats(knn, X, 10)
    m = ApproximateLOFModel(10, lrd, k_dist, faiss_index_bytes=np.asarray(faiss.serialize_index(index)))
    save_lof_model(m, str(tmp_path / 'lof.pkl'))
    m2 = load_lof_model(str(tmp_path / 'lof.pkl'))
    Xq = rng.normal(size=(50, 6)).astype(np.float32)
    np.testing.assert_allclose(m.decision_function(Xq), m2.decision_function(Xq))


# 6 ───────────────────────────────────────────────────────────────────────────
def test_numpy_autoencoder_matches_torch(tmp_path):
    torch = pytest.importorskip('torch')
    from models import build_autoencoder, export_ae_weights, ae_forward_numpy
    torch.manual_seed(0)
    m = build_autoencoder(23, [32, 16, 8, 16, 32]).eval()
    p = tmp_path / 'w.npz'
    export_ae_weights(m, str(p))
    X = np.random.default_rng(0).random((64, 23)).astype(np.float32)
    with torch.no_grad():
        ref = m(torch.tensor(X)).numpy()
    np.testing.assert_allclose(ae_forward_numpy(dict(np.load(p)), X), ref, atol=1e-5)


# 7 ───────────────────────────────────────────────────────────────────────────
@pytest.mark.skipif(not os.path.exists(os.path.join(ROOT, 'outputs', 'tuned_params.json')),
                    reason='run src/tune.py first')
def test_config_uses_tuned_params():
    import config
    with open(config.TUNED_PARAMS_PATH) as f:
        t = json.load(f)
    assert config.TUNED
    assert abs(sum(config.ENSEMBLE_WEIGHTS.values()) - 1) < 1e-6
    assert config.MODEL_THRESHOLDS == {k: float(v) for k, v in t['thresholds'].items()}
    assert all(0 < v < 1 for v in config.MODEL_THRESHOLDS.values())
