#!/bin/bash
# Reproduce all experiments and analyses of the paper (one GPU; about 1-2 days with cached volumes).
set -e
cd "$(dirname "$0")/.."
MODELS="amage concat cross gate g d braingnn gcn bc_gcn resnet efficientnet cnn_orig cnn mfformer ctransformer setgfrn"
for cohort in camcan oasis; do
  for m in $MODELS; do
    python train.py --model "$m" --cohort "$cohort"
  done
done
python analysis/compare.py
python analysis/calibration.py
python analysis/saliency.py
python analysis/saliency_stats.py
python analysis/figures.py

python analysis/figure4_publication.py
python analysis/figure5_publication.py
python analysis/figure10_publication.py
