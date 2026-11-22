"""
models.py
Three unsupervised anomaly detectors + weighted ensemble:
  1. Isolation Forest  -- tree-based outlier scoring
  2. LOF               -- Local Outlier Factor on FAISS approximate kNN
  3. Autoencoder       -- PyTorch for training; NumPy forward pass for inference
  4. Ensemble          -- weighted combination; weights tuned on the VAL split (src/tune.py)

Score convention for every model: higher = more anomalous, normalised to [0, 1]
with a MinMaxScaler fitted on TRAINING scores only, then clipped.

Audit fixes in this version
  - IF score scaler was fitted on TEST scores (leak + different scale from AE/LOF).
    Now fitted on training scores, same as LOF and AE.
  - LOF dropped the nearest neighbour for every query point. That is only correct
    when the query set IS the fitted set (a point is its own neighbour). For new
    points (val/test/live) it threw away the true nearest neighbour. Now handled
    with an explicit exclude_self flag.
  - LOF model is saved as a plain dict (no class pickling), so it loads from any
    entry point (script, pytest, streamlit). Legacy pickles still load.
    The FAISS index already stores the vectors, so X_fit is no longer duplicated
    in the file when FAISS is used (~halves the file size).
  - Autoencoder inference runs in NumPy from outputs/autoencoder_weights.npz,
    so the deployed dashboard does not need torch (too large for Streamlit Cloud).
    torch is only imported when training or computing saliency.
"""

import os
import sys
import pickle
import logging
import functools
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    CONTAMINATION, RANDOM_SEED, OUTPUTS_DIR,
    IF_MODEL_PATH, IF_SCALER_PATH,
    AE_MODEL_PATH, AE_WEIGHTS_PATH, AE_SCALER_PATH, AE_MSE_SCALER_PATH,
    ANOMALY_SCORES_PATH,
    AE_EPOCHS, AE_BATCH_SIZE, AE_LR, AE_HIDDEN,
    AE_EARLY_STOP_PATIENCE,
    ENSEMBLE_WEIGHTS,
    LOF_N_NEIGHBORS, LOF_MODEL_PATH, LOF_SCALER_PATH,
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
log = logging.getLogger(__name__)

MODEL_NAMES = ['Isolation Forest', 'Autoencoder', 'LOF', 'Ensemble']


def _minmax_apply(scaler, raw):
    return np.clip(scaler.transform(np.asarray(raw).reshape(-1, 1)).ravel(), 0.0, 1.0)


# ─── Isolation Forest ────────────────────────────────────────────────────────

def train_isolation_forest(X_train, X_test):
    log.info("=== Isolation Forest ===")
    model = IsolationForest(
        contamination=CONTAMINATION,
        random_state=RANDOM_SEED,
        n_estimators=100,
        n_jobs=-1,
    )
    X_train = np.asarray(X_train, dtype=np.float32)   # numpy in, numpy at inference (no name warnings)
    X_test = np.asarray(X_test, dtype=np.float32)
    model.fit(X_train)

    # Score scaler fitted on TRAINING scores (was: fitted on test scores -> leak)
    raw_train = -model.decision_function(X_train)
    scaler = MinMaxScaler().fit(raw_train.reshape(-1, 1))

    raw_test = -model.decision_function(X_test)
    scores_norm = _minmax_apply(scaler, raw_test)

    with open(IF_MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    with open(IF_SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)

    log.info(f"Saved IF model  -> {IF_MODEL_PATH}")
    log.info(f"Saved IF scaler -> {IF_SCALER_PATH}")
    return scores_norm


# ─── LOF on approximate kNN ──────────────────────────────────────────────────
#
# LOF(x) = mean over k neighbours o of lrd(o) / lrd(x)
# lrd(x) = 1 / mean(reach_dist(x, o)),  reach_dist(x, o) = max(k_dist(o), d(x, o))
#
# FAISS IVFFlat gives approximate neighbours in O(nprobe * k) per query, so all
# training rows are used (no subsampling). Falls back to an exact BallTree.

def _make_knn(X_fit32):
    """Return (knn_query(X, k, exclude_self) -> (dist, idx), faiss_index_or_None)."""
    try:
        import faiss
    except ImportError:
        faiss = None

    n, d = X_fit32.shape
    if faiss is not None:
        nlist = max(4, int(np.sqrt(n)))
        quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
        index.train(X_fit32)
        index.add(X_fit32)
        index.nprobe = max(1, nlist // 10)
        log.info(f"FAISS IVFFlat: nlist={nlist}, nprobe={index.nprobe}, n={n:,}, d={d}")
        return _faiss_query_fn(index), index

    log.info("faiss not available -- using sklearn BallTree (exact, slower)")
    from sklearn.neighbors import BallTree
    tree = BallTree(X_fit32, leaf_size=40)

    def knn_query(X_q, k, exclude_self=False):
        D, I = tree.query(np.asarray(X_q, dtype=np.float32), k=k + int(exclude_self))
        return (D[:, 1:], I[:, 1:]) if exclude_self else (D, I)
    return knn_query, None


def _faiss_query_fn(index):
    def knn_query(X_q, k, exclude_self=False):
        D, I = index.search(np.ascontiguousarray(X_q, dtype=np.float32), k + int(exclude_self))
        D = np.sqrt(np.clip(D, 0, None))            # FAISS returns squared L2
        if exclude_self:
            D, I = D[:, 1:], I[:, 1:]
        # IVF can return -1 when a probed cell has fewer than k points; reuse nearest valid
        if (I < 0).any():
            bad = I < 0
            I = np.where(bad, I[:, :1], I)
            D = np.where(bad, D[:, :1], D)
        return D, I
    return knn_query


def _lof_from_neighbours(dist_q, idx_q, k_dist_fit, lrd_fit):
    reach = np.maximum(dist_q, k_dist_fit[idx_q])
    lrd_q = 1.0 / (reach.mean(axis=1) + 1e-10)
    return (lrd_fit[idx_q] / lrd_q[:, None]).mean(axis=1)


def _fit_lof_stats(knn_query, X_fit32, k, chunk=10_000):
    """k-distance and lrd of every training point (self excluded from its own neighbours)."""
    n = len(X_fit32)
    dist = np.empty((n, k), dtype=np.float32)
    idx = np.empty((n, k), dtype=np.int64)
    for i in range(0, n, chunk):
        dist[i:i + chunk], idx[i:i + chunk] = knn_query(X_fit32[i:i + chunk], k, exclude_self=True)
    k_dist_fit = dist[:, -1]
    reach = np.maximum(dist, k_dist_fit[idx])
    lrd_fit = 1.0 / (reach.mean(axis=1) + 1e-10)
    return k_dist_fit, lrd_fit, dist, idx


def _lof_scores_from_knn(X_fit: np.ndarray, X_query: np.ndarray, k: int):
    """
    Raw LOF scores for training points and for NEW query points.
    Returns (lof_train, lof_query, faiss_index_or_None, knn_query).
    Higher = more anomalous; ~1 for inliers.
    """
    X_fit32 = np.ascontiguousarray(X_fit, dtype=np.float32)
    X_q32 = np.ascontiguousarray(X_query, dtype=np.float32)
    knn_query, index = _make_knn(X_fit32)
    k_dist_fit, lrd_fit, dist_fit, idx_fit = _fit_lof_stats(knn_query, X_fit32, k)

    lof_train = _lof_from_neighbours(dist_fit, idx_fit, k_dist_fit, lrd_fit)
    CHUNK = 10_000
    lof_q = []
    for i in range(0, len(X_q32), CHUNK):
        d, ix = knn_query(X_q32[i:i + CHUNK], k, exclude_self=False)   # new points: keep nearest
        lof_q.append(_lof_from_neighbours(d, ix, k_dist_fit, lrd_fit))
    lof_query = np.concatenate(lof_q) if lof_q else np.empty(0)
    return lof_train, lof_query, index, knn_query


class ApproximateLOFModel:
    """
    Inference wrapper. decision_function(X) follows the sklearn convention:
    values in (-inf, 1], more negative = more anomalous.
    """
    def __init__(self, k, lrd_fit, k_dist_fit, faiss_index_bytes=None, X_fit=None):
        self.k = int(k)
        self.lrd_fit = np.asarray(lrd_fit)
        self._k_dist_fit = np.asarray(k_dist_fit)
        self.faiss_index_bytes = faiss_index_bytes
        self.X_fit = X_fit                       # only kept for the BallTree fallback
        self.use_faiss = faiss_index_bytes is not None
        self._knn_query = None

    # -- persistence (plain dict: loads from any entry point) --
    def to_state(self):
        return {
            'format': 'payguard-lof-v2',
            'k': self.k,
            'lrd_fit': self.lrd_fit,
            'k_dist_fit': self._k_dist_fit,
            'faiss_index_bytes': self.faiss_index_bytes,
            'X_fit': None if self.use_faiss else self.X_fit,
        }

    @classmethod
    def from_state(cls, st):
        return cls(st['k'], st['lrd_fit'], st['k_dist_fit'], st.get('faiss_index_bytes'), st.get('X_fit'))

    def _build_query_fn(self):
        if self.use_faiss:
            import faiss
            index = faiss.deserialize_index(np.asarray(self.faiss_index_bytes, dtype=np.uint8))
            index.nprobe = max(1, index.nlist // 10)
            self._knn_query = _faiss_query_fn(index)
        else:
            from sklearn.neighbors import BallTree
            tree = BallTree(np.asarray(self.X_fit, dtype=np.float32), leaf_size=40)

            def knn_query(X_q, k, exclude_self=False):
                D, I = tree.query(np.asarray(X_q, dtype=np.float32), k=k + int(exclude_self))
                return (D[:, 1:], I[:, 1:]) if exclude_self else (D, I)
            self._knn_query = knn_query

    def decision_function(self, X):
        if self._knn_query is None:
            self._build_query_fn()
        X32 = np.ascontiguousarray(X, dtype=np.float32)
        out = []
        for i in range(0, len(X32), 10_000):
            d, ix = self._knn_query(X32[i:i + 10_000], self.k, exclude_self=False)
            out.append(_lof_from_neighbours(d, ix, self._k_dist_fit, self.lrd_fit))
        lof = np.concatenate(out) if out else np.empty(0)
        return 1.0 - lof

    # legacy-pickle compatibility (older files pickled the object itself)
    def __setstate__(self, state):
        self.__dict__.update(state)
        self.__dict__.setdefault('_knn_query', None)
        if 'use_faiss' not in self.__dict__:
            self.use_faiss = self.__dict__.get('faiss_index_bytes') is not None


def save_lof_model(model: ApproximateLOFModel, path=LOF_MODEL_PATH):
    with open(path, 'wb') as f:
        pickle.dump(model.to_state(), f, protocol=pickle.HIGHEST_PROTOCOL)


def load_lof_model(path=LOF_MODEL_PATH) -> ApproximateLOFModel:
    import types
    # Legacy files pickled `__main__.ApproximateLOFModel`; make that name resolvable.
    main = sys.modules.setdefault('__main__', types.ModuleType('__main__'))
    if not hasattr(main, 'ApproximateLOFModel'):
        main.ApproximateLOFModel = ApproximateLOFModel
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    if isinstance(obj, dict) and obj.get('format') == 'payguard-lof-v2':
        return ApproximateLOFModel.from_state(obj)
    return obj


def train_lof(X_train, X_test):
    log.info("=== LOF (full-data, approximate kNN) ===")
    X_fit = np.ascontiguousarray(getattr(X_train, 'values', X_train), dtype=np.float32)
    X_q = np.ascontiguousarray(getattr(X_test, 'values', X_test), dtype=np.float32)
    k = LOF_N_NEIGHBORS
    log.info(f"LOF: fitting on ALL {len(X_fit):,} training rows (k={k})")

    knn_query, index = _make_knn(X_fit)
    k_dist_fit, lrd_fit, dist_fit, idx_fit = _fit_lof_stats(knn_query, X_fit, k)
    raw_train = _lof_from_neighbours(dist_fit, idx_fit, k_dist_fit, lrd_fit) - 1.0

    if index is not None:
        import faiss
        model = ApproximateLOFModel(k, lrd_fit, k_dist_fit,
                                    faiss_index_bytes=np.asarray(faiss.serialize_index(index)))
    else:
        model = ApproximateLOFModel(k, lrd_fit, k_dist_fit, X_fit=X_fit)

    scaler = MinMaxScaler().fit(raw_train.reshape(-1, 1))
    raw_test = -model.decision_function(X_q)          # = lof - 1
    scores_norm = _minmax_apply(scaler, raw_test)

    save_lof_model(model, LOF_MODEL_PATH)
    with open(LOF_SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    log.info(f"Saved LOF model  -> {LOF_MODEL_PATH}")
    log.info(f"Saved LOF scaler -> {LOF_SCALER_PATH}")
    return scores_norm


# ─── Autoencoder ─────────────────────────────────────────────────────────────

def build_autoencoder(input_dim, hidden_dims):
    """
    torch nn.Module: encoder input->32->16->8, decoder 8->16->32->input.
    No output activation: targets are MinMax-scaled and can leave [0,1] on new data;
    inputs and reconstructions are clipped to [0,1] before the MSE instead.
    """
    import torch.nn as nn

    class Autoencoder(nn.Module):
        def __init__(self):
            super().__init__()
            mid = len(hidden_dims) // 2
            enc, prev = [], input_dim
            for h in hidden_dims[:mid + 1]:
                enc += [nn.Linear(prev, h), nn.ReLU()]
                prev = h
            self.encoder = nn.Sequential(*enc)
            dec = []
            for h in hidden_dims[mid + 1:]:
                dec += [nn.Linear(prev, h), nn.ReLU()]
                prev = h
            dec += [nn.Linear(prev, input_dim)]
            self.decoder = nn.Sequential(*dec)

        def forward(self, x):
            return self.decoder(self.encoder(x))

    return Autoencoder()


def export_ae_weights(model, path=AE_WEIGHTS_PATH):
    """Save Linear layers in forward order as W0,b0,W1,b1,... for the NumPy forward pass."""
    import torch.nn as nn
    arrays = {}
    linears = [m for m in list(model.encoder) + list(model.decoder) if isinstance(m, nn.Linear)]
    for i, lin in enumerate(linears):
        arrays[f'W{i}'] = lin.weight.detach().cpu().numpy().astype(np.float32)
        arrays[f'b{i}'] = lin.bias.detach().cpu().numpy().astype(np.float32)
    np.savez(path, n_layers=len(linears), **arrays)


def ae_forward_numpy(weights, X):
    """ReLU after every layer except the last -- identical to build_autoencoder()."""
    h = np.asarray(X, dtype=np.float32)
    n = int(weights['n_layers'])
    for i in range(n):
        h = h @ weights[f'W{i}'].T + weights[f'b{i}']
        if i < n - 1:
            h = np.maximum(h, 0.0)
    return h


def ae_reconstruction_mse(weights, X_norm):
    recon = ae_forward_numpy(weights, X_norm)
    return np.mean((np.clip(X_norm, 0, 1) - np.clip(recon, 0, 1)) ** 2, axis=1)


def train_autoencoder(X_train, X_test, y_train=None):
    """
    Trained on NORMAL rows only when y_train is given (semi-supervised anomaly
    detection: the AE learns normal behaviour; fraud reconstructs poorly).
    Early stopping on a held-out 10% of the (shuffled) training rows.
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    log.info("=== Autoencoder ===")
    torch.manual_seed(RANDOM_SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"Device: {device}")

    X_train = np.asarray(X_train, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)
    if y_train is not None:
        normal_mask = np.asarray(y_train) == 0
        X_train_ae = X_train[normal_mask]
        log.info(f"AE training on {len(X_train_ae):,} normal rows (excluded {(~normal_mask).sum():,} fraud)")
    else:
        X_train_ae = X_train

    input_dim = X_train_ae.shape[1]
    scaler = MinMaxScaler()
    X_train_norm = scaler.fit_transform(X_train_ae).astype(np.float32)
    X_test_norm = scaler.transform(X_test).astype(np.float32)
    with open(AE_SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)

    rng = np.random.default_rng(RANDOM_SEED)
    perm = rng.permutation(len(X_train_norm))
    n_val = max(1, int(0.1 * len(perm)))
    X_val_norm = X_train_norm[perm[:n_val]]
    X_fit_norm = X_train_norm[perm[n_val:]]

    loader = DataLoader(TensorDataset(torch.tensor(X_fit_norm)), batch_size=AE_BATCH_SIZE, shuffle=True)
    model = build_autoencoder(input_dim, AE_HIDDEN).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=AE_LR)
    criterion = nn.MSELoss()
    val_t = torch.tensor(X_val_norm).to(device)

    best_val, best_state, no_improve = float('inf'), None, 0
    for epoch in range(AE_EPOCHS):
        model.train()
        total = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            loss = criterion(model(batch), batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(val_t), val_t).item()
        if (epoch + 1) % 5 == 0:
            log.info(f"  Epoch {epoch+1:3d}/{AE_EPOCHS}  train={total/len(loader):.6f}  val={val_loss:.6f}")
        if val_loss < best_val:
            best_val, no_improve = val_loss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= AE_EARLY_STOP_PATIENCE:
                log.info(f"  Early stopping at epoch {epoch+1}")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval().cpu()

    torch.save(model.state_dict(), AE_MODEL_PATH)
    export_ae_weights(model, AE_WEIGHTS_PATH)
    weights = dict(np.load(AE_WEIGHTS_PATH))

    mse_train = ae_reconstruction_mse(weights, X_fit_norm)
    mse_scaler = MinMaxScaler().fit(mse_train.reshape(-1, 1))
    with open(AE_MSE_SCALER_PATH, 'wb') as f:
        pickle.dump(mse_scaler, f)

    mse_test = ae_reconstruction_mse(weights, X_test_norm)
    log.info(f"Saved AE -> {AE_MODEL_PATH}, {AE_WEIGHTS_PATH}  (best val_loss={best_val:.6f})")
    return _minmax_apply(mse_scaler, mse_test)


# ─── Cached loaders (inference) ──────────────────────────────────────────────

@functools.lru_cache(maxsize=1)
def _load_if_cached():
    with open(IF_MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(IF_SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler


@functools.lru_cache(maxsize=1)
def _load_ae_cached():
    if not os.path.exists(AE_WEIGHTS_PATH) and os.path.exists(AE_MODEL_PATH):
        # One-off conversion from an older torch checkpoint (needs torch locally).
        import torch
        with open(AE_SCALER_PATH, 'rb') as f:
            input_dim = pickle.load(f).n_features_in_
        m = build_autoencoder(input_dim, AE_HIDDEN)
        m.load_state_dict(torch.load(AE_MODEL_PATH, map_location='cpu', weights_only=True))
        export_ae_weights(m, AE_WEIGHTS_PATH)
    weights = dict(np.load(AE_WEIGHTS_PATH))
    with open(AE_SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    with open(AE_MSE_SCALER_PATH, 'rb') as f:
        mse_scaler = pickle.load(f)
    return weights, scaler, mse_scaler


@functools.lru_cache(maxsize=1)
def _load_lof_cached():
    lof_model = load_lof_model(LOF_MODEL_PATH)
    with open(LOF_SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    lof_model._build_query_fn()
    return lof_model, scaler


# ─── Inference ───────────────────────────────────────────────────────────────

def score_if(X):
    model, scaler = _load_if_cached()
    return _minmax_apply(scaler, -model.decision_function(X))


def score_ae(X):
    weights, scaler, mse_scaler = _load_ae_cached()
    X_norm = scaler.transform(X).astype(np.float32)
    return _minmax_apply(mse_scaler, ae_reconstruction_mse(weights, X_norm))


def score_lof(X):
    lof_model, scaler = _load_lof_cached()
    return _minmax_apply(scaler, -lof_model.decision_function(X))


def score_all(X, weights=None):
    """Component scores + ensemble for a feature matrix (n, 23)."""
    weights = weights or ENSEMBLE_WEIGHTS
    X = np.asarray(X, dtype=np.float32)
    s = {'if': score_if(X), 'ae': score_ae(X), 'lof': score_lof(X)}
    s['ensemble'] = weights['if'] * s['if'] + weights['ae'] * s['ae'] + weights['lof'] * s['lof']
    return s


def run_inference(df: pd.DataFrame, model_name: str = 'Isolation Forest', threshold: float = 0.85):
    """Return a copy of df with 'anomaly_score' and 'is_fraud' columns."""
    X = df.values.astype(np.float32)
    if model_name == 'Isolation Forest':
        scores = score_if(X)
    elif model_name == 'Autoencoder':
        scores = score_ae(X)
    elif model_name == 'LOF':
        scores = score_lof(X)
    elif model_name == 'Ensemble':
        scores = score_all(X)['ensemble']
    else:
        raise ValueError(f"Unknown model: {model_name}")
    result = df.copy()
    result['anomaly_score'] = scores
    result['is_fraud'] = scores >= threshold
    return result


# ─── Train all + save combined test scores ───────────────────────────────────

def train_all(X_train, X_test, y_train=None):
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    if_scores = train_isolation_forest(X_train, X_test)
    lof_scores = train_lof(X_train, X_test)
    ae_scores = train_autoencoder(X_train, X_test, y_train=y_train)
    _load_if_cached.cache_clear(); _load_ae_cached.cache_clear(); _load_lof_cached.cache_clear()

    ensemble = (ENSEMBLE_WEIGHTS['if'] * if_scores + ENSEMBLE_WEIGHTS['ae'] * ae_scores
                + ENSEMBLE_WEIGHTS['lof'] * lof_scores)
    scores_df = pd.DataFrame({'if_score': if_scores, 'lof_score': lof_scores,
                              'ae_score': ae_scores, 'ensemble_score': ensemble})
    scores_df.to_csv(ANOMALY_SCORES_PATH, index=False)
    log.info(f"Saved all scores -> {ANOMALY_SCORES_PATH} (ensemble re-weighted by tune.py)")
    return scores_df


if __name__ == '__main__':
    from data_processing import load_and_clean
    X_train, X_test, y_train, y_test, X_val, y_val = load_and_clean()
    scores = train_all(X_train, X_test, y_train=y_train)
    print("\nAll models trained.")
    print(scores.describe())
