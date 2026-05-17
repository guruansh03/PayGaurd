#!/usr/bin/env bash
# retrain.sh — Delete stale artifacts and retrain all models after IMP-01/02/03
# Run from repo root: bash retrain.sh

set -e

echo "=== Deleting stale artifacts ==="
rm -f outputs/autoencoder.pt
rm -f outputs/autoencoder_scaler.pkl
rm -f outputs/feature_scaler.pkl
rm -f outputs/anomaly_scores.csv
rm -f outputs/lof_model.pkl
rm -f outputs/lof_scaler.pkl
rm -f outputs/velocity.db
rm -f outputs/receiver_stats.csv
rm -f data/upi_transactions.csv
echo "Done."

echo "=== Step 1: Generate data (IMP-01 + IMP-02 + IMP-03) ==="
python src/generate_upi_data.py

echo "=== Step 2: Process features + rebuild velocity store ==="
python src/data_processing.py

echo "=== Step 3: Train IF + AE (input_dim=23) + LOF ==="
python src/models.py

echo "=== Step 4: SHAP analysis with new features ==="
python src/analysis.py

echo ""
echo "=== DONE. Next steps: ==="
echo "  1. Run threshold sweep on outputs/anomaly_scores.csv"
echo "  2. Update AE_THRESHOLD, LOF_THRESHOLD, ENSEMBLE_WEIGHTS in config.py"
echo "  3. streamlit run app.py"
