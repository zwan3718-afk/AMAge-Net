# AMAge-Net: Attention-Guided Multimodal Fusion of Structural and Functional MRI for Brain Age Prediction

Code for *"Attention-Guided Multimodal Fusion of Structural and Functional MRI Reveals Interpretable Brain Aging Patterns"* (Human Brain Mapping, under revision).

AMAge-Net combines a 3D DenseNet121 branch for T1-weighted sMRI with a graph attention network (GAT) branch for resting-state functional connectivity, fused by multi-head cross-attention (functional query, structural keys/values) followed by a scalar gated fusion. This repository contains the model, all competing methods and ablation variants as used in the paper, the cross-validation protocol, and the scripts that produce the statistics and data figures.

## Repository structure

```
amagenet/            AMAge-Net
  gat.py             fMRI branch (GAT + attention pooling)
  models.py          AMAge-Net and fusion variants (Cross-only, Gate-only, Concat)
  data.py            participant lists, T1w preprocessing (cached), graph construction
baselines/           re-implementations of the competing methods and single-modality variants
registry.py          architecture, inputs, loss and optimizer of every method
train.py             five-fold cross-validation with an internal validation set (all methods)
analysis/
  compare.py         paired comparisons with AMAge-Net (Table 1, Table 3, Tables S2, S8)
  calibration.py     age bias and calibration (Table 2)
  saliency.py        per-participant fMRI and sMRI regional saliency
  saliency_stats.py  rankings, stability, cross-cohort replication, sex comparison, rendering data (Figs 6-9)
  figures.py         data figures (Figs 4, 5, 10, S1-S4)
  figure4_publication.py / figure5_publication.py / figure10_publication.py
                     final manuscript layouts (larger fonts; no Regression/Identical legend)
  check_gat_oversmoothing.py  diagnostic for the GAT node representations
config.py            data locations (override in config_local.py)
data/aal90_abbreviations.csv
scripts/run_all.sh   reproduce all experiments and analyses
```

## Installation

Tested with Python 3.12, CUDA 11.8 and one NVIDIA RTX 4090 (24 GB).

```bash
pip install -r requirements.txt
```

Install `torch` and `torch_geometric` builds matching your CUDA version (see their installation pages).

## Data

The experiments use the Cam-CAN (https://camcan-archive.mrc-cbu.cam.ac.uk) and OASIS-3 (https://sites.wustl.edu/oasisbrains/) datasets, which are available from the data providers upon application. No imaging or participant-level data are distributed with this repository.

For each participant, the code expects:

- a preprocessed T1-weighted volume in MNI space (Cam-CAN: DARTEL; OASIS-3: skull-stripped volume from the official OASIS FreeSurfer pipeline), which is linearly resampled to 128³ and min-max scaled (`amagenet/data.py`);
- a 90 × 90 functional connectivity matrix (Pearson correlation between the time series of the 90 cerebral AAL regions), stored as CSV with region labels in the first row and column;
- an age table (`Subject`, `Age`) and, for the sex analysis, a sex table (`Subject`, `Sex` with values `MALE`/`FEMALE`).

Set the paths in `config.py`, or copy the `DATASETS` dictionary and the AAL paths into `config_local.py` (ignored by git). Preprocessing details are given in the paper and its Supplementary Material.

## Cross-validation protocol (all methods)

- Outer folds: `KFold(5, shuffle=True, random_state=42)` over the sorted participant list; each outer fold is the held-out **test** set once.
- Internal **validation** set: 10% of each training portion, stratified by age quintile (`random_state = 42 + fold`).
- Training: at most 50 epochs, batch size 8, learning rate 1e-4, no early stopping. `ReduceLROnPlateau` (factor 0.5, patience 5, minimum 1e-6) monitors the internal-validation loss; the checkpoint with the lowest internal-validation MAE is kept.
- The test fold is evaluated **once**, with that checkpoint. The min-max scaling of age is fitted on the training set of each fold.
- The splits are written to `results/splits/<cohort>/fold<k>.json` by the first run and shared by all methods, so that every method is evaluated on the same test participants.

## Running

```bash
python train.py --model amage --cohort camcan          # one model, one cohort
python train.py --model amage --cohort camcan --smoke  # quick check of the setup
bash scripts/run_all.sh                                # everything (about 1-2 days on one RTX 4090)
```

| `--model` | Method | Optimizer / loss |
|---|---|---|
| `amage` | AMAge-Net (Cross-Attention + Gated Fusion) | Adam, wd 1e-4 / MSE |
| `concat`, `cross`, `gate` | fusion ablations: concatenation, Cross-Attention only, Gated Fusion only | Adam, wd 1e-4 / MSE |
| `g`, `d` | single-modality ablations: fMRI (GAT) only, sMRI (DenseNet121) only | Adam / MSE |
| `braingnn`, `gcn`, `bc_gcn` | fMRI baselines | Adam / MSE |
| `resnet`, `efficientnet`, `cnn_orig` | sMRI baselines (`cnn_orig` = 3D-CNN, original configuration) | Adam / MSE |
| `cnn` | 3D-CNN with weight decay and all layers trained (Table S14) | Adam, wd 1e-4 / MSE |
| `mfformer`, `ctransformer`, `setgfrn` | multimodal baselines | AdamW, wd 1e-4 / L1 |

Each run writes, per fold, the test predictions, metrics, training history and best checkpoint to `results/<model>_<cohort>/`, and a `summary.json` (mean and SD over folds).

Then:

```bash
python analysis/compare.py         # -> analysis_out/comparison.csv
python analysis/calibration.py     # -> analysis_out/calibration.csv, age_strata.csv
python analysis/saliency.py        # -> analysis_out/saliency_{fmri,smri}_<cohort>.csv
python analysis/saliency_stats.py  # -> analysis_out/saliency_*.csv, saliency_summary.json, rendering/
python analysis/figures.py         # -> analysis_out/fig*.png (general plots and supplementary figures)
python analysis/figure4_publication.py
python analysis/figure5_publication.py
python analysis/figure10_publication.py  # -> analysis_out/publication_figures/ (final manuscript layouts)
```

## Notes

- **Regional saliency.** fMRI saliency is the gradient × input attribution of the predicted age with respect to the FC node features; sMRI saliency is the absolute gradient of the loss with respect to the input voxels, averaged within AAL regions. Both are computed on each participant's held-out test fold. The attention-pooling scores of the GAT are *not* used as regional importance: with the fully connected functional graph, the GAT node representations become identical across regions (`analysis/check_gat_oversmoothing.py`).
- **Graph construction.** |FC| defines the topology (all region pairs are connected) and is not passed to the GAT as an edge attribute; FC information enters the GAT through the node features (rows of the FC matrix).
- **Baselines** are our re-implementations for brain age regression (classification heads replaced by regression heads where needed). Please cite the original publications of these methods when using them. As in our experiments, SET-GFRN receives the band-pass-filtered FC matrix as its functional input, and the 3D-CNN (`cnn_orig`) builds its first fully connected layer lazily after optimizer construction, as in its original configuration.
- **Third-party code.** `baselines/efficientnet_utils.py` and `baselines/efficientnet3d.py` are adapted (3D) from EfficientNet-PyTorch (Luke Melas-Kyriazi, Apache License 2.0); please keep the corresponding notice when redistributing.
- **Memory.** MFFormer needs about 23.5 GB of GPU memory at batch size 8; `train.py` enables `expandable_segments` to avoid fragmentation.
- Results may differ slightly between GPUs and library versions because of non-deterministic CUDA kernels.

## Reproduction scope

This package contains training and downstream analysis code. It does not contain raw MRI, participant-level tables or predictions, trained checkpoints, the AAL atlas, or the external SPM/FreeSurfer preprocessing tools. Obtain the datasets and atlas separately and configure their paths before running. Brain renderings for Figures 6–9 require external rendering software; the analysis exports their regional data. The final layout scripts require results for both cohorts (Figures 4–5) or all Cam-CAN ablations (Figure 10).

## Citation

If you use this code, please cite the paper (reference to be added upon publication).

## License

To be added by the authors.
