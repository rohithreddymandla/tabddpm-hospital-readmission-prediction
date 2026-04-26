# TabDDPM for Hospital Readmission Prediction

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-TabDDPM-red)
![Task](https://img.shields.io/badge/Task-Tabular%20Binary%20Classification-green)
![Status](https://img.shields.io/badge/Status-Academic%20Project-orange)

This repository evaluates whether diffusion-generated synthetic tabular data can improve prediction of 30-day hospital readmission (`<30`) in an imbalanced clinical dataset.

## Objective

The target class (`readmitted = <30`) is a minority class.  
The project compares data balancing and generation strategies to improve minority-class detection:

- Baseline (no augmentation)
- SMOTE
- CTGAN
- TabDDPM (class-conditioned denoising diffusion model)

## Dataset

- Source: Diabetes 130-US hospitals dataset (`data/diabetic_data.csv`)
- Size: 101,766 records, 50 columns
- Target: `readmitted` (`NO`, `>30`, `<30`)
- Binary setup for modeling: `<30` vs all others

## Method Overview

1. Data preprocessing and feature handling
2. Baseline XGBoost model on imbalanced data
3. Synthetic minority generation with SMOTE / CTGAN / TabDDPM
4. Train downstream classifier on augmented data
5. Compare metrics on held-out test data

## Repository Structure

```text
tabddpm-hospital-readmission-prediction/
├── anum-forward-process/
│   ├── data/
│   ├── models/
│   ├── notebooks/
│   │   └── forward_process.ipynb
│   └── results/
├── data/
│   ├── diabetic_data.csv
│   ├── description.pdf
│   └── processed/
├── models/
│   ├── diffusion_config.pkl
│   ├── label_encoders.pkl
│   ├── scaler.pkl
│   └── xgboost_baseline.pkl
├── notebook/
│   ├── main.ipynb
│   └── tabddpm_model.py
├── results/
│   ├── figures/
│   └── metrics/
├── BUG_REPORT.md
├── MSML612_Project_Proposal.pdf
├── TabDDPM-proposal.pdf
├── requirements.txt
└── README.md
```

## Key Files

- `notebook/main.ipynb`: end-to-end workflow and experiments
- `notebook/tabddpm_model.py`: TabDDPM denoiser architecture module
- `models/diffusion_config.pkl`: saved diffusion metadata/config
- `results/metrics/`: experiment outputs
- `results/figures/`: charts and visual diagnostics

## Setup

```bash
pip install -r requirements.txt
```

Then run the notebook:

- `notebook/main.ipynb`

## Evaluation Metrics

- F1-score (minority class `<30`) - primary metric
- ROC-AUC
- Confusion matrix
- Distribution and fidelity checks for synthetic data

## Results Snapshot

Current baseline outputs are available in:

- `results/metrics/baseline_results.json`
- `results/figures/baseline_auc_roc.png`
- `results/figures/baseline_confusion_matrix.png`
- `results/figures/baseline_feature_importance.png`

Add final comparison tables/plots here when SMOTE, CTGAN, and TabDDPM runs are finalized.
