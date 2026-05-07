# TabDDPM for Hospital Readmission Prediction

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-TabDDPM-red)
![Task](https://img.shields.io/badge/Task-Tabular%20Binary%20Classification-green)
![Status](https://img.shields.io/badge/Status-Academic%20Project-orange)



This project tests whether a diffusion model (TabDDPM) can generate realistic fake patient records to fix class imbalance in hospital readmission prediction. Only 11% of patients in the dataset were readmitted within 30 days, so a normal classifier just says "not readmitted" for everyone and misses the patients that actually matter. We built TabDDPM from scratch in PyTorch, generated synthetic readmitted patients, and compared it against SMOTE and CTGAN.

## Objective

The target class (`readmitted = <30`) makes up only 11% of the dataset. A classifier trained on this imbalanced data gets 89% accuracy while catching almost zero actual readmissions.

This project compares four approaches to fix that:

- **Baseline** -- train on the original imbalanced data, no fix
- **SMOTE** -- create fake patients by averaging between real nearest neighbors
- **CTGAN** -- use a GAN to generate fake patient rows
- **TabDDPM** -- use a diffusion model to generate fake patients (our method, built from scratch)

## Dataset

- **Source:** Diabetes 130-US Hospitals dataset ([UCI / Kaggle](https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008))
- **Size:** 101,766 patients from 130 US hospitals
- **Features:** 42 after preprocessing (9 numeric, 33 categorical)
- **Target:** `readmitted` -- binary setup: `<30` vs everything else
- **Class split:** 11% readmitted (minority), 89% not readmitted (majority)
- **Raw file:** `data/diabetic_data.csv`

## How TabDDPM Works

1. **Forward process (Anum):** Take a real patient row and slowly destroy it with noise over 1000 steps. Numeric columns get Gaussian noise, categorical columns get randomly flipped (multinomial diffusion). By step 999 the original patient is completely gone.
2. **Neural network (Jiten):** An MLP that looks at a noisy patient row + timestep and predicts what noise was added. Uses residual blocks with adaptive layer norm.
3. **Class conditioning (Simi):** The readmission label (0 or 1) is converted to a 128-dim vector and fed into the network so it learns what readmitted vs non-readmitted patients look like separately.
4. **Training (Rohith):** Pick random patients, pick random timesteps, run forward process, feed noisy data to the network, compute loss (MSE for numeric + cross-entropy for categorical), update weights. 100 epochs.
5. **Generation (Simi):** Start from pure random noise, set class = readmitted, run the trained model backward 1000 steps, out comes a new fake patient.
6. **Evaluation (Taneir):** Train XGBoost on augmented data, test on real held-out patients. Compare F1 and ROC-AUC across all methods. Repeat 5 times with different seeds.

## Results

| Method | F1 Score | ROC-AUC |
|--------|----------|---------|
| Baseline (no augmentation) | 0.0548 ± 0.0000 | 0.6581 |
| SMOTE | 0.0524 ± 0.0052 | 0.6525 |
| CTGAN | 0.0480 ± 0.0000 | 0.6599 |
| TabDDPM 50% | 0.0405 ± 0.0000 | 0.6588 |
| **TabDDPM 100%** | **0.0477 ± 0.0000** | **0.6616** |
| TabDDPM 200% | 0.0421 ± 0.0000 | 0.6584 |

**Key findings:**
- TabDDPM 100% got the best ROC-AUC (0.6616) -- best at ranking high-risk patients
- No method improved F1 over baseline -- the bottleneck is the classification threshold, not data quality
- 100% augmentation is the sweet spot -- 50% is too little, 200% adds too much noise
- Training on only fake patients gave F1 = 0 for both CTGAN and TabDDPM -- synthetic data can support real data but can't replace it
- SMOTE was the only method with variance across seeds (std = 0.0052) -- TabDDPM gives the same result every run

## Repository Structure

```text
tabddpm-hospital-readmission-prediction/
├── anum-forward-process/
│   ├── data/                          # copy of dataset + processed splits
│   ├── models/                        # diffusion config + saved models
│   ├── notebooks/
│   │   └── forward_process.ipynb      # forward process + noise scheduling (Anum)
│   └── results/figures/               # forward process visualizations
├── data/
│   ├── diabetic_data.csv              # raw dataset
│   ├── description.pdf                # dataset documentation
│   ├── processed/                     # cleaned train/test splits (X_train, X_test, y_train, y_test)
│   └── synthetic/                     # generated fake patients at 50%, 100%, 200%
├── models/
│   ├── tabddpm_best.pt               # trained TabDDPM model checkpoint
│   ├── diffusion_config.pkl           # noise schedule, column indices, category counts
│   ├── xgboost_baseline.pkl           # baseline XGBoost classifier
│   ├── scaler.pkl                     # feature scaler from preprocessing
│   └── label_encoders.pkl             # encoders for categorical columns
├── notebook/
│   ├── main.ipynb                     # data prep, baseline, training loop (Rohith)
│   ├── tabddpm_model.py              # MLP network architecture (Jiten)
│   ├── class_conditioning_generation.ipynb  # conditioning + generation (Simi)
│   ├── generate.py                    # generation script (Simi)
│   └── taneir_evaluation.ipynb        # SMOTE, CTGAN, all experiments (Taneir)
├── results/
│   ├── figures/                       # all plots (distributions, t-SNE, comparisons)
│   └── metrics/                       # experiment results as JSON files
├── BUG_REPORT.md
├── MSML612_Project_Proposal.pdf
├── TabDDPM-proposal.pdf
├── requirements.txt
└── README.md
```

## Key Files

| File | What it does |
|------|-------------|
| `notebook/main.ipynb` | End-to-end pipeline -- preprocessing, baseline, training |
| `notebook/tabddpm_model.py` | The TabDDPM neural network (MLP + timestep embedding + AdaLN) |
| `notebook/class_conditioning_generation.ipynb` | Class conditioning setup + synthetic patient generation |
| `notebook/taneir_evaluation.ipynb` | All evaluation experiments (SMOTE, CTGAN, TabDDPM, synthetic-only test) |
| `anum-forward-process/notebooks/forward_process.ipynb` | Noise schedule, Gaussian forward process, multinomial diffusion |
| `models/diffusion_config.pkl` | Precomputed betas, alphas, alpha_bars, column info -- used by training and generation |
| `models/tabddpm_best.pt` | Trained model weights |
| `data/synthetic/` | Generated synthetic patients at three augmentation levels |

## Note on `anum-forward-process/`

This folder is a self-contained copy of the forward process work. It has its own copy of the dataset, processed splits, and model files. The main pipeline uses the files in `data/`, `models/`, and `notebook/`. Both work independently.

## Setup

```bash
git clone https://github.com/Jiten-Bhalavat/tabddpm-hospital-readmission-prediction.git
cd tabddpm-hospital-readmission-prediction
pip install -r requirements.txt
```

Then open and run `notebook/main.ipynb`.

## Evaluation Metrics

- **F1 score on the readmitted class** -- primary metric. Measures how well the classifier catches the rare high-risk patients.
- **ROC-AUC** -- secondary metric. Measures how well the classifier ranks high-risk patients above low-risk ones overall, regardless of threshold.
- **Jensen-Shannon divergence** -- measures how similar the generated patient distributions are to real ones. Lower = more realistic.
- **t-SNE** -- 2D visualization to check if real and fake patients overlap or separate into clusters.


## References

- Kotelnikov et al. (2023). TabDDPM: Modelling Tabular Data with Diffusion Models. ICML 2023.
- Ho et al. (2020). Denoising Diffusion Probabilistic Models. NeurIPS 2020.
- Xu et al. (2019). Modeling Tabular Data using Conditional GAN. NeurIPS 2019.
- Chawla et al. (2002). SMOTE: Synthetic Minority Over-sampling Technique. JAIR 2002.
- Hoogeboom et al. (2021). Argmax Flows and Multinomial Diffusion. NeurIPS 2021.
- Strack et al. (2014). Impact of HbA1c Measurement on Hospital Readmission Rates. BioMed Research International.
