# PayGuard — Real-Time UPI Fraud Detection

Anomaly-detection system for UPI payments: an ensemble of **Isolation Forest**, an **Autoencoder** and **LOF (FAISS approximate kNN)**, with explainability (SHAP, gradient saliency, permutation importance), a velocity store, OCR receipt scoring and a Streamlit dashboard with a tamper-evident audit log.

> Data is **synthetic**: 293,000 UPI transactions (1.02% fraud, 8 fraud patterns) produced by `src/generate_upi_data.py`. All results below are on that synthetic data.

## Results (held-out TEST split: 43,950 rows, 450 fraud)

Thresholds and ensemble weights are chosen on the **validation** split only (`src/tune.py`). The test split is scored once. 95% CIs come from 300 bootstrap resamples.

| Model | Threshold | ROC-AUC (95% CI) | PR-AUC | Precision | Recall | F1 (95% CI) | Flag rate |
|---|---|---|---|---|---|---|---|
| **Ensemble** (0.05 IF + 0.95 AE) | 0.28 | **0.980** (0.974–0.985) | **0.758** | **0.849** | **0.727** | **0.783** (0.753–0.810) | 0.88% |
| Autoencoder | 0.27 | 0.975 (0.968–0.981) | 0.754 | 0.864 | 0.722 | 0.787 (0.757–0.812) | 0.86% |
| Isolation Forest | 0.69 | 0.946 (0.939–0.953) | 0.258 | 0.288 | 0.367 | 0.323 (0.290–0.361) | 1.30% |
| LOF | 0.31 | 0.793 (0.769–0.815) | 0.048 | 0.075 | 0.182 | 0.106 (0.089–0.125) | 2.49% |

5-fold stratified CV on train+val (scalers fitted inside each fold): Isolation Forest ROC-AUC 0.959 ± 0.007, LOF 0.624 ± 0.020.

Notes:
- The Autoencoder is trained on **normal rows only**, so it uses labels to filter its training data (semi-supervised). Isolation Forest and LOF never see labels.
- On validation, the weight search gave almost all ensemble weight to the Autoencoder. IF and LOF add little F1, and that is reported as-is.
- LOF is weak here because ordinal-encoded categoricals (bank, state) dominate its distance metric.

## Pipeline

```
generate_upi_data.py   synthetic transactions, 8 fraud patterns, time-window features
data_processing.py     23 features, 70/15/15 stratified split, train-only sender/receiver stats, velocity store, 5-fold CV
models.py              Isolation Forest, LOF (FAISS IVFFlat), Autoencoder (PyTorch; exported to NumPy)
tune.py                per-model thresholds + ensemble weights on VAL; TEST metrics -> outputs/metrics.json
analysis.py            SHAP (IF), gradient saliency (AE), permutation importance (LOF)
visualization.py       PCA, t-SNE, score distribution, confusion matrix, SHAP chart
app.py                 Streamlit dashboard: Overview · Live Score (+OCR) · Scan CSV · Analytics · About
```

## Run locally

```bash
pip install -r requirements.txt          # dashboard only (no torch)
streamlit run app.py

pip install -r requirements-train.txt    # full retrain (~6 min on CPU)
bash retrain.sh
pytest -q                                # 28 tests
```

## Deploy on Streamlit Community Cloud

1. share.streamlit.io → **Create app** → repo `guruansh03/PayGaurd`, branch `main`, main file `app.py`.
2. **Advanced settings → Python version: 3.12**. Streamlit Cloud ignores `runtime.txt`, and the pinned numpy/pandas need Python 3.11 or newer.
3. Deploy. `packages.txt` installs `tesseract-ocr` for the OCR receipt feature.

The dashboard needs no torch: the Autoencoder runs in NumPy from `outputs/autoencoder_weights.npz`, so the install stays small.

## Audit fixes (Nov 2026)

- **Timestamp unit bug (critical).** Under pandas 3, `datetime.astype(int64) // 10**9` returned values 1000× too small. Every "1h" velocity window covered about 41 days and every "24h" window about 2.7 years. Replaced with a resolution-safe conversion and regenerated the data.
- **Velocity fraud pattern** never produced bursts (each row had an independent random time). It now generates real 5-transaction bursts within about 30 minutes.
- **Live/raw-CSV feature order**: `encode_raw_df()` emitted the 23 features in a different order from training, so the models read them by position into the wrong slots. Fixed and covered by a test.
- **Isolation Forest score scaler** was fitted on test scores. It is now fitted on training scores, like LOF and AE.
- **LOF** dropped the true nearest neighbour for every new point. It is now exact-equivalent to sklearn's novelty LOF (tested). The model is saved as a plain dict, which loads from any entry point and is about half the size.
- **Thresholds and ensemble weights** were hand-set placeholders. They are now tuned on validation and loaded automatically by `config.py`.
- **Evaluation**: each model uses its own threshold. CV runs on train+val only, with no pre-scaling leak. Results include PR-AUC and bootstrap CIs.
- **Deploy**: removed the requirement conflicts (scipy/shap pins vs numpy 2.x), removed torch from runtime requirements, and pinned the tested versions.
- **Dashboard**: fixed the pandas-3 `read_json` crash in Scan, the "Amount at Risk" figure (it was summing standardised values), mixed-format timestamp parsing, and the hard-coded AUC cards.
