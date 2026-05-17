"""
models.py
Three unsupervised anomaly detectors + weighted ensemble:
  1. Isolation Forest  -- tree-based outlier scoring
  2. LOF               -- Local Outlier Factor (C-02: replaced DBSCAN)
  3. Autoencoder       -- PyTorch; high reconstruction error = anomaly
  4. Ensemble          -- weighted combination (FIX 9.5)

C-01: AE_HIDDEN=[32,16,8,16,32] -- bottleneck=8 (42% of 19 dims, was 4/21%).
      Wider bottleneck -> better recall for subtle fraud.
      REQUIRED: delete outputs/autoencoder.pt + autoencoder_mse_scaler.pkl before retrain.
      REQUIRED: recalibrate AE_THRESHOLD after retrain (MSE distribution shifts).

C-02: LOF replaces DBSCAN.
      DBSCAN AUC=0.7888, precision=0.073 -- fraud geometrically spread across clusters.
      LOF compares local density to neighbours -- correct algorithm for this geometry.
      novelty=True required for inference on unseen data.
      Score direction: decision_function() returns (−∞,1], negated before scaling.
      Speed: FAISS approximate kNN when available; no 50k training cap.

Phase 0/2/Audit fixes carried forward.
"""

import os
import sys
import pickle
import logging
import functools
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    CONTAMINATION, RANDOM_SEED, OUTPUTS_DIR,
    IF_MODEL_PATH, IF_SCALER_PATH,
    AE_MODEL_PATH, AE_SCALER_PATH, AE_MSE_SCALER_PATH,
    ANOMALY_SCORES_PATH,
    AE_EPOCHS, AE_BATCH_SIZE, AE_LR, AE_HIDDEN,
    AE_EARLY_STOP_PATIENCE,
    ENSEMBLE_WEIGHTS,
    LOF_N_NEIGHBORS, LOF_ALGORITHM, LOF_MODEL_PATH, LOF_SCALER_PATH,
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
log = logging.getLogger(__name__)


# ─── Isolation Forest ────────────────────────────────────────────────────────

def train_isolation_forest(X_train, X_test):
    log.info("=== Isolation Forest ===")
    model = IsolationForest(
        contamination=CONTAMINATION,
        random_state=RANDOM_SEED,
        n_estimators=100,
        n_jobs=-1,
    )
    model.fit(X_train)

    raw_scores = -model.decision_function(X_test)

    scaler = MinMaxScaler()
    scores_norm = scaler.fit_transform(raw_scores.reshape(-1, 1)).ravel()

    with open(IF_MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    with open(IF_SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)

    log.info(f"Saved IF model  -> {IF_MODEL_PATH}")
    log.info(f"Saved IF scaler -> {IF_SCALER_PATH}")
    return scores_norm


# ─── LOF (C-02: replaces DBSCAN) ─────────────────────────────────────────────

# ---------------------------------------------------------------------------
# ApproximateLOF — FAISS-backed LOF that scales to 200k+ rows.
#
# Why not vanilla sklearn LOF on full data?
#   sklearn LOF builds an exact ball-tree; kneighbors queries at inference are
#   O(n * k * log n) over ALL stored training points. At 200k rows and k=20
#   that is ~800k distance evaluations per query row — slow at inference time.
#
# Strategy: FAISS IVFFlat index (inverted-file approximate NN).
#   - Fits on ALL training rows — no subsampling, no quality loss.
#   - Builds a compressed index (nlist centroids) so query is O(nprobe * k)
#     instead of O(n * k). 100-1000x faster at inference for 200k+ datasets.
#   - LOF scores computed manually: lrd(x) / mean(lrd(neighbours)).
#     Same math as sklearn LOF, approximate neighbours only.
#   - Falls back to sklearn LOF on full data if faiss unavailable.
#     sklearn on 200k still works — takes ~5 min to train, ~2s per inference
#     call (acceptable for batch scoring; slow for live single-row scoring).
# ---------------------------------------------------------------------------

def _lof_scores_from_knn(X_fit: np.ndarray, X_query: np.ndarray, k: int):
    """
    Compute LOF scores given a pre-built kNN structure.
    Works with any kNN backend that returns (distances, indices).
    Returns raw LOF scores (higher = more anomalous, >1 means outlier).
    """
    try:
        import faiss
        _use_faiss = True
    except ImportError:
        _use_faiss = False

    X_fit32   = X_fit.astype(np.float32)
    X_query32 = X_query.astype(np.float32)
    n, d = X_fit32.shape

    if _use_faiss:
        # Normalize for cosine-style distance; L2 on unit sphere = 1 - cosine
        # For tabular fraud data we keep L2 (no normalization needed).
        nlist = max(4, int(np.sqrt(n)))          # number of Voronoi cells
        nprobe = max(1, nlist // 10)              # cells to search at query time
        quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
        index.train(X_fit32)
        index.add(X_fit32)
        index.nprobe = nprobe
        log.info(f"FAISS IVFFlat: nlist={nlist}, nprobe={nprobe}, n={n:,}, d={d}")

        def knn_query(X_q, n_neighbors):
            # k+1 because a point is its own neighbour in the training set
            D, I = index.search(X_q, n_neighbors + 1)
            # Distances are squared L2; convert to L2
            return np.sqrt(np.clip(D[:, 1:], 0, None)), I[:, 1:]

    else:
        # Pure sklearn fallback — exact kNN, no FAISS
        log.info("faiss not available — using sklearn BallTree (exact, slower)")
        from sklearn.neighbors import BallTree
        tree = BallTree(X_fit32, leaf_size=40)

        def knn_query(X_q, n_neighbors):
            D, I = tree.query(X_q, k=n_neighbors + 1)
            return D[:, 1:], I[:, 1:]

    # ── reach-distance & lrd on training set (needed for lrd of query pts) ──
    dist_fit, idx_fit = knn_query(X_fit32, k)
    k_dist_fit = dist_fit[:, -1]                  # k-th NN distance per point
    # reach_dist[i, j] = max(k_dist(neighbour_j), dist(i, neighbour_j))
    reach_dist_fit = np.maximum(dist_fit, k_dist_fit[idx_fit])
    lrd_fit = 1.0 / (np.mean(reach_dist_fit, axis=1) + 1e-10)

    def _lof(X_q):
        dist_q, idx_q = knn_query(X_q, k)
        k_dist_q = dist_q[:, -1]
        reach_dist_q = np.maximum(dist_q, k_dist_fit[idx_q])
        lrd_q = 1.0 / (np.mean(reach_dist_q, axis=1) + 1e-10)
        lof_q = np.mean(lrd_fit[idx_q] / lrd_q[:, np.newaxis], axis=1)
        return lof_q

    # Chunk queries to avoid OOM on very large sets
    CHUNK = 10_000
    lof_train = np.concatenate([_lof(X_fit32[i:i+CHUNK]) for i in range(0, n, CHUNK)])
    lof_test  = np.concatenate([_lof(X_query32[i:i+CHUNK]) for i in range(0, len(X_query32), CHUNK)])

    return lof_train, lof_test, index if _use_faiss else None, knn_query


class ApproximateLOFModel:
    """
    Pickle-able wrapper that stores the FAISS index (or sklearn BallTree)
    + training lrd values so inference can be run on new data.
    Matches the sklearn LOF decision_function() interface:
        decision_function(X) -> array of shape (n,)
        Values in (-inf, 1]. More negative = more anomalous.
    """
    def __init__(self, lrd_fit, idx_fit_knn_query_fn, k, X_fit, use_faiss):
        self.k = k
        self.lrd_fit = lrd_fit          # shape (n_train,)
        self.X_fit   = X_fit            # float32 array, stored for BallTree fallback
        self.use_faiss = use_faiss
        # Store raw index bytes for FAISS (faiss objects aren't picklable directly)
        if use_faiss:
            import faiss, io
            buf = faiss.serialize_index(idx_fit_knn_query_fn)
            self.faiss_index_bytes = np.frombuffer(buf, dtype=np.uint8)
        else:
            self.faiss_index_bytes = None
        self._knn_query = None          # rebuilt lazily on first call

    def _build_query_fn(self):
        if self.use_faiss:
            import faiss
            buf = self.faiss_index_bytes.tobytes()
            index = faiss.deserialize_index(np.frombuffer(buf, dtype=np.uint8))
            index.nprobe = max(1, index.nlist // 10)
            k = self.k
            def knn_query(X_q, n_neighbors):
                D, I = index.search(X_q.astype(np.float32), n_neighbors + 1)
                return np.sqrt(np.clip(D[:, 1:], 0, None)), I[:, 1:]
        else:
            from sklearn.neighbors import BallTree
            tree = BallTree(self.X_fit, leaf_size=40)
            def knn_query(X_q, n_neighbors):
                D, I = tree.query(X_q.astype(np.float32), k=n_neighbors + 1)
                return D[:, 1:], I[:, 1:]
        self._knn_query = knn_query

    def decision_function(self, X):
        """sklearn-compatible: returns values in (-inf, 1]. Negate for anomaly score."""
        if self._knn_query is None:
            self._build_query_fn()
        X32 = np.asarray(X, dtype=np.float32)
        k = self.k
        lrd_fit = self.lrd_fit
        CHUNK = 10_000
        lof_scores = []
        for i in range(0, len(X32), CHUNK):
            Xc = X32[i:i+CHUNK]
            dist_q, idx_q = self._knn_query(Xc, k)
            reach_dist_q = np.maximum(dist_q, self._k_dist_fit[idx_q])
            lrd_q = 1.0 / (np.mean(reach_dist_q, axis=1) + 1e-10)
            lof_q = np.mean(lrd_fit[idx_q] / lrd_q[:, np.newaxis], axis=1)
            lof_scores.append(lof_q)
        lof = np.concatenate(lof_scores)
        # Convert to sklearn decision_function convention: 1 - lof (>0 = inlier)
        return 1.0 - lof

    def fit_store_kdist(self, k_dist_fit):
        """Store k-distances of training points (needed for reach-distance at inference)."""
        self._k_dist_fit = k_dist_fit


def train_lof(X_train, X_test):
    """
    C-02: Local Outlier Factor — full-data training via FAISS approximate kNN.

    Key implementation notes:
    - novelty=True equivalent: ApproximateLOFModel.decision_function() works on unseen data.
    - Score direction: decision_function() returns (−∞, 1].
      More negative = more anomalous. Negate before scaling so higher = more anomalous.
    - Speed: FAISS IVFFlat index trained on ALL rows. Query is O(nprobe * k) not O(n * k).
      200k rows, k=20, nprobe=sqrt(n)/10: ~0.3s per 1k queries vs ~5s with sklearn exact kNN.
    - Falls back to sklearn LOF on full data if faiss not installed.
    - Scaler fitted on training scores (same pattern as IF).
    - NO subsampling: all training rows used.
    """
    log.info("=== LOF (full-data, approximate kNN) ===")

    X_fit = X_train.values if hasattr(X_train, 'values') else X_train
    X_q   = X_test.values  if hasattr(X_test,  'values') else X_test
    X_fit = X_fit.astype(np.float32)
    X_q   = X_q.astype(np.float32)
    k     = LOF_N_NEIGHBORS

    log.info(f"LOF: fitting on ALL {len(X_fit):,} training rows (k={k})")

    try:
        import faiss
        _use_faiss = True
        log.info("LOF backend: FAISS IVFFlat (approximate, fast)")
    except ImportError:
        _use_faiss = False
        log.info("LOF backend: sklearn BallTree (exact, slower — pip install faiss-cpu for speedup)")

    lof_train_scores, lof_test_scores, faiss_index_or_none, knn_query_fn = \
        _lof_scores_from_knn(X_fit, X_q, k)

    # Raw LOF values: >1 means outlier. Negate so higher = more anomalous.
    raw_train = lof_train_scores - 1.0   # centre at 0 for inliers
    raw_test  = lof_test_scores  - 1.0

    scaler = MinMaxScaler()
    scaler.fit(raw_train.reshape(-1, 1))
    scores_norm = np.clip(scaler.transform(raw_test.reshape(-1, 1)).ravel(), 0.0, 1.0)

    # ── Build and save ApproximateLOFModel ──
    # knn_query needed to get k_dist_fit for reach-distance at inference
    X_fit32 = X_fit.astype(np.float32)
    dist_fit_kdist, idx_fit = knn_query_fn(X_fit32, k)
    k_dist_fit = dist_fit_kdist[:, -1]

    # lrd_fit needed at inference
    reach_dist_fit = np.maximum(dist_fit_kdist, k_dist_fit[idx_fit])
    lrd_fit = 1.0 / (np.mean(reach_dist_fit, axis=1) + 1e-10)

    if _use_faiss:
        lof_model = ApproximateLOFModel(
            lrd_fit=lrd_fit,
            idx_fit_knn_query_fn=faiss_index_or_none,
            k=k,
            X_fit=X_fit32,
            use_faiss=True,
        )
    else:
        # For BallTree fallback, store X_fit so it can be rebuilt
        lof_model = ApproximateLOFModel(
            lrd_fit=lrd_fit,
            idx_fit_knn_query_fn=None,  # not used for BallTree path
            k=k,
            X_fit=X_fit32,
            use_faiss=False,
        )
    lof_model.fit_store_kdist(k_dist_fit)

    with open(LOF_MODEL_PATH, 'wb') as f:
        pickle.dump(lof_model, f)
    with open(LOF_SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)

    log.info(f"LOF trained on {len(X_fit):,} rows (was capped at 50k previously)")
    log.info(f"Saved LOF model  -> {LOF_MODEL_PATH}")
    log.info(f"Saved LOF scaler -> {LOF_SCALER_PATH}")
    return scores_norm


# ─── Autoencoder (PyTorch) ───────────────────────────────────────────────────

class Autoencoder(nn.Module):
    """
    Symmetric autoencoder. C-01: AE_HIDDEN=[32,16,8,16,32]:
      encoder: input -> 32 -> 16 -> 8  (bottleneck=8, 42% of 19 input dims)
      decoder: 8 -> 16 -> 32 -> input
    Was [16,8,4,8,16]: bottleneck=4 (21%) -> recall=0.262, missed 74% of fraud.
    """
    def __init__(self, input_dim, hidden_dims):
        super().__init__()
        mid = len(hidden_dims) // 2

        enc_layers = []
        prev = input_dim
        for h in hidden_dims[:mid + 1]:
            enc_layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        self.encoder = nn.Sequential(*enc_layers)

        dec_layers = []
        for h in hidden_dims[mid + 1:]:
            dec_layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        # L-11: Removed Sigmoid -- it clamped output to [0,1] but MinMaxScaler
        # can produce values outside [0,1] for test data. Sigmoid made those
        # unreconstructable -> artificially inflated MSE -> false positives.
        # Instead we clip inputs to [0,1] before MSE (see train_autoencoder/run_inference).
        dec_layers += [nn.Linear(prev, input_dim)]
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x):
        return self.decoder(self.encoder(x))


def train_autoencoder(X_train, X_test, y_train=None):
    """
    FIX 9.9: Validation split + early stopping. FIX-02: MSE scaler. C-01: wider bottleneck.
    C-05: If y_train provided, trains on NORMAL data only (y_train==0).
          This is correct for anomaly detection -- AE learns to reconstruct normal patterns.
          Fraud has high reconstruction error because model never saw it.
    """
    log.info("=== Autoencoder ===")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"Device: {device}")

    # C-05: Filter to normal-only training data
    if y_train is not None:
        normal_mask = (y_train == 0).values if hasattr(y_train, 'values') else (y_train == 0)
        X_train_ae = X_train[normal_mask]
        log.info(f"C-05: AE training on {len(X_train_ae):,} normal rows (excluded {(~normal_mask).sum():,} fraud)")
    else:
        X_train_ae = X_train
        log.info(f"AE training on all {len(X_train_ae):,} rows (no labels provided)")

    input_dim = X_train_ae.shape[1]

    scaler = MinMaxScaler()
    X_train_norm = scaler.fit_transform(X_train_ae).astype(np.float32)
    X_test_norm  = scaler.transform(X_test).astype(np.float32)

    with open(AE_SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    log.info(f"Saved AE scaler -> {AE_SCALER_PATH}")

    n_val = max(1, int(0.1 * len(X_train_norm)))
    X_val_norm = X_train_norm[:n_val]
    X_fit_norm = X_train_norm[n_val:]

    train_tensor = torch.tensor(X_fit_norm)
    val_tensor   = torch.tensor(X_val_norm)
    loader = DataLoader(TensorDataset(train_tensor), batch_size=AE_BATCH_SIZE, shuffle=True)

    model = Autoencoder(input_dim, AE_HIDDEN).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=AE_LR)
    criterion = nn.MSELoss()

    best_val_loss   = float('inf')
    best_state_dict = None
    patience        = AE_EARLY_STOP_PATIENCE
    no_improve      = 0

    for epoch in range(AE_EPOCHS):
        model.train()
        total_loss = 0
        for (batch,) in loader:
            batch = batch.to(device)
            recon = model(batch)
            loss  = criterion(recon, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        with torch.no_grad():
            val_recon = model(val_tensor.to(device)).cpu()
            val_loss  = criterion(val_recon, torch.tensor(X_val_norm)).item()

        if (epoch + 1) % 5 == 0:
            log.info(f"  Epoch {epoch+1:3d}/{AE_EPOCHS}  train={total_loss/len(loader):.6f}  val={val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss   = val_loss
            best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve      = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                log.info(f"  Early stopping at epoch {epoch+1} (no val improvement for {patience} epochs)")
                break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        log.info(f"  Restored best checkpoint (val_loss={best_val_loss:.6f})")

    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X_test_norm).to(device)
        recon = model(X_t).cpu().numpy()
    # L-11: Clip both target and reconstruction to [0,1] for fair MSE comparison.
    # Without Sigmoid the decoder can output any range; clipping aligns them.
    X_test_clipped = np.clip(X_test_norm, 0.0, 1.0)
    recon_clipped  = np.clip(recon, 0.0, 1.0)
    mse = np.mean((X_test_clipped - recon_clipped) ** 2, axis=1)

    with torch.no_grad():
        X_train_t = torch.tensor(X_fit_norm).to(device)
        recon_train = model(X_train_t).cpu().numpy()
    X_fit_clipped    = np.clip(X_fit_norm, 0.0, 1.0)
    recon_tr_clipped = np.clip(recon_train, 0.0, 1.0)
    mse_train = np.mean((X_fit_clipped - recon_tr_clipped) ** 2, axis=1)

    mse_scaler = MinMaxScaler()
    mse_scaler.fit(mse_train.reshape(-1, 1))

    with open(AE_MSE_SCALER_PATH, 'wb') as f:
        pickle.dump(mse_scaler, f)
    log.info(f"Saved AE MSE scaler -> {AE_MSE_SCALER_PATH}")

    mse_norm = np.clip(mse_scaler.transform(mse.reshape(-1, 1)).ravel(), 0.0, 1.0)

    torch.save(model.state_dict(), AE_MODEL_PATH)
    log.info(f"Saved AE model -> {AE_MODEL_PATH}  (best val_loss={best_val_loss:.6f})")
    return mse_norm


# ─── L-06: Module-level model cache -- avoids re-loading from disk on every call ──

@functools.lru_cache(maxsize=1)
def _load_if_cached():
    with open(IF_MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(IF_SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

@functools.lru_cache(maxsize=1)
def _load_ae_cached(input_dim):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ae = Autoencoder(input_dim, AE_HIDDEN).to(device)
    ae.load_state_dict(torch.load(AE_MODEL_PATH, map_location=device, weights_only=True))
    ae.eval()
    with open(AE_SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    with open(AE_MSE_SCALER_PATH, 'rb') as f:
        mse_scaler = pickle.load(f)
    return ae, scaler, mse_scaler, device

@functools.lru_cache(maxsize=1)
def _load_lof_cached():
    # FIX: pickle stored ApproximateLOFModel as __main__.ApproximateLOFModel
    # when models.py was run directly. Inject into __main__ so unpickling works
    # regardless of entry point (streamlit, script, etc.)
    import sys as _sys
    if not hasattr(_sys.modules.get('__main__', None), 'ApproximateLOFModel'):
        import types as _types
        _main = _sys.modules.setdefault('__main__', _types.ModuleType('__main__'))
        _main.ApproximateLOFModel = ApproximateLOFModel
    with open(LOF_MODEL_PATH, 'rb') as f:
        lof_model = pickle.load(f)
    with open(LOF_SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    # Rebuild the kNN query function (FAISS index or BallTree) after unpickling.
    if hasattr(lof_model, '_build_query_fn'):
        lof_model._build_query_fn()
    return lof_model, scaler


# ─── Run inference on new data (used by app.py) ──────────────────────────────

def run_inference(df: pd.DataFrame, model_name: str = 'Isolation Forest', threshold: float = 0.85):
    """
    Given a preprocessed DataFrame, return a copy with 'anomaly_score'
    and 'is_fraud' columns.
    L-06: Uses module-level LRU-cached model loaders to avoid redundant disk I/O.
    """
    X = df.values.astype(np.float32)

    if model_name == 'Isolation Forest':
        model, scaler = _load_if_cached()
        raw = -model.decision_function(X)
        scores = np.clip(scaler.transform(raw.reshape(-1, 1)).ravel(), 0.0, 1.0)

    elif model_name == 'Autoencoder':
        ae, scaler, mse_scaler, device = _load_ae_cached(X.shape[1])
        X_norm = scaler.transform(X).astype(np.float32)
        with torch.no_grad():
            recon = ae(torch.tensor(X_norm).to(device)).cpu().numpy()
        # L-11: Clip to [0,1] for fair MSE -- decoder no longer has Sigmoid
        X_clipped = np.clip(X_norm, 0.0, 1.0)
        recon_clipped = np.clip(recon, 0.0, 1.0)
        mse = np.mean((X_clipped - recon_clipped) ** 2, axis=1)
        scores = np.clip(mse_scaler.transform(mse.reshape(-1, 1)).ravel(), 0.0, 1.0)

    elif model_name == 'LOF':
        lof_model, scaler = _load_lof_cached()
        raw = -lof_model.decision_function(X)
        scores = np.clip(scaler.transform(raw.reshape(-1, 1)).ravel(), 0.0, 1.0)

    elif model_name == 'Ensemble':
        if_model, if_scaler = _load_if_cached()
        raw_if = -if_model.decision_function(X)
        if_sc  = np.clip(if_scaler.transform(raw_if.reshape(-1, 1)).ravel(), 0, 1)

        ae, ae_scaler, ae_mse_scaler, device = _load_ae_cached(X.shape[1])
        X_norm = ae_scaler.transform(X).astype(np.float32)
        with torch.no_grad():
            recon = ae(torch.tensor(X_norm).to(device)).cpu().numpy()
        X_clipped = np.clip(X_norm, 0.0, 1.0)
        recon_clipped = np.clip(recon, 0.0, 1.0)
        mse = np.mean((X_clipped - recon_clipped) ** 2, axis=1)
        ae_sc = np.clip(ae_mse_scaler.transform(mse.reshape(-1, 1)).ravel(), 0.0, 1.0)

        lof_model, lof_scaler = _load_lof_cached()
        raw_lof = -lof_model.decision_function(X)
        lof_sc  = np.clip(lof_scaler.transform(raw_lof.reshape(-1, 1)).ravel(), 0, 1)

        scores = (
            ENSEMBLE_WEIGHTS['if']  * if_sc +
            ENSEMBLE_WEIGHTS['ae']  * ae_sc +
            ENSEMBLE_WEIGHTS['lof'] * lof_sc
        )

    else:
        raise ValueError(f"Unknown model: {model_name}")

    result = df.copy()
    result['anomaly_score'] = scores
    result['is_fraud'] = scores >= threshold
    return result


# ─── Train all + save combined scores ────────────────────────────────────────

def train_all(X_train, X_test, y_train=None):
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    if_scores  = train_isolation_forest(X_train, X_test)
    lof_scores = train_lof(X_train, X_test)
    ae_scores  = train_autoencoder(X_train, X_test, y_train=y_train)  # C-05: normal-only

    ensemble_scores = (
        ENSEMBLE_WEIGHTS['if']  * if_scores +
        ENSEMBLE_WEIGHTS['ae']  * ae_scores +
        ENSEMBLE_WEIGHTS['lof'] * lof_scores
    )

    scores_df = pd.DataFrame({
        'if_score':       if_scores,
        'lof_score':      lof_scores,
        'ae_score':       ae_scores,
        'ensemble_score': ensemble_scores,
    })
    scores_df.to_csv(ANOMALY_SCORES_PATH, index=False)
    log.info(f"Saved all scores -> {ANOMALY_SCORES_PATH}")
    return scores_df


if __name__ == '__main__':
    from data_processing import load_and_clean
    X_train, X_test, y_train, y_test, X_val, y_val = load_and_clean()
    scores = train_all(X_train, X_test, y_train=y_train)  # C-05: pass y_train
    print("\nAll models trained.")
    print(scores.describe())
