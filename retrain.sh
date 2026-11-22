#!/usr/bin/env bash
# retrain.sh — full pipeline from scratch. Run from repo root: bash retrain.sh
# Needs requirements-train.txt (includes torch). The dashboard itself only needs requirements.txt.
set -e

echo "=== 1/6 Generate synthetic UPI data ==="
python src/generate_upi_data.py

echo "=== 2/6 Features, 70/15/15 split, velocity store, 5-fold CV (train+val) ==="
python src/data_processing.py

echo "=== 3/6 Train Isolation Forest + LOF + Autoencoder ==="
python src/models.py

echo "=== 4/6 Tune thresholds + ensemble weights on VAL, report TEST metrics ==="
python src/tune.py

echo "=== 5/6 Explainability: SHAP (IF), gradient saliency (AE), permutation (LOF) ==="
python src/analysis.py

echo "=== 6/6 Charts ==="
python src/visualization.py

echo "Done. Launch: streamlit run app.py"
