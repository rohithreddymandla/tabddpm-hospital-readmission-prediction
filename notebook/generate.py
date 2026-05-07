"""
generate.py — TabDDPM Generation Script
Starts from random noise, runs 1000 denoising steps, outputs synthetic readmission patients.

Needs:
    - models/tabddpm.pt           (trained model weights)
    - models/diffusion_config.pkl (noise schedule and column metadata)
    - tabddpm_model.py            (model architecture)
    - data/processed/X_train.csv  (used to sample categorical columns)
"""

import torch
import pickle
import numpy as np
import pandas as pd
from tabddpm_model import TabDDPMDenoiser


def load_model_and_config(model_path: str, config_path: str) -> tuple:
    """
    Load the trained model and the diffusion config.
    The config tells us which columns are numeric, which are categorical,
    and what the noise schedule looks like.
    """
    with open(config_path, 'rb') as f:
        cfg = pickle.load(f)

    model = TabDDPMDenoiser(
        num_numeric       = len(cfg['numeric_indices']),
        cat_cardinalities = cfg['cat_num_classes'],
        numeric_indices   = cfg['numeric_indices'],
        cat_indices       = cfg['cat_indices'],
    )
    checkpoint = torch.load(model_path, map_location='cpu')
    state_dict = checkpoint['model_state'] if 'model_state' in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    print(f"Model loaded. Numeric cols: {len(cfg['numeric_indices'])}, "
          f"Categorical cols: {len(cfg['cat_indices'])}")
    return model, cfg


@torch.no_grad()
def generate_numeric(model: TabDDPMDenoiser, cfg: dict, n_samples: int, target_class: int = 1, device: str = 'cpu',) -> np.ndarray:
    """
    Reverse diffusion loop for the 9 numeric columns.

    How it works:
    - Start with a completely random row of numbers (pure noise)
    - Run 1000 steps in reverse, from step 999 down to step 0
    - At each step, the model guesses what noise was added and we remove a bit of it
    - By step 0, the noise has been fully removed and we have a realistic synthetic patient

    The class label (readmitted = 1) is passed in at every step so the model
    knows what kind of patient to generate.
    """
    betas      = torch.tensor(cfg['betas'],      dtype=torch.float32, device=device)
    alphas     = torch.tensor(cfg['alphas'],     dtype=torch.float32, device=device)
    alpha_bars = torch.tensor(cfg['alpha_bars'], dtype=torch.float32, device=device)

    # Step 1: start from pure random noise
    x = torch.randn(n_samples, len(cfg['numeric_indices']), device=device)

    # Set the class label to readmitted (1) for all samples
    y = torch.full((n_samples,), target_class, dtype=torch.long, device=device)

    T = len(betas)  # 1000 total steps

    for t in reversed(range(T)):
        t_batch = torch.full((n_samples,), t, dtype=torch.long, device=device)

        # The model expects a full 42-column row.
        # We only have the 9 numeric columns at this point,
        # so we place them in the right positions and leave the rest as zero.
        x_full = torch.zeros(n_samples, 42, device=device)
        for col_pos, feat_idx in enumerate(cfg['numeric_indices']):
            x_full[:, feat_idx] = x[:, col_pos]

        # Ask the model to predict what noise was added at this step
        eps_pred = model(x_full, t_batch, y)

        # Use the predicted noise to estimate what the clean patient looks like
        alpha_t     = alphas[t]
        alpha_bar_t = alpha_bars[t]
        beta_t      = betas[t]

        x0_pred = (x - (1 - alpha_bar_t).sqrt() * eps_pred) / alpha_bar_t.sqrt()
        x0_pred = x0_pred.clamp(-3, 3)  # stop values going to extreme ranges

        if t > 0:
            # Not the last step yet — compute the denoised value and
            # add a small amount of fresh noise to keep generation diverse
            alpha_bar_prev = alpha_bars[t - 1]
            mean = (
                beta_t * alpha_bar_prev.sqrt() / (1 - alpha_bar_t) * x0_pred
                + (1 - alpha_bar_prev) * alpha_t.sqrt() / (1 - alpha_bar_t) * x
            )
            posterior_var = beta_t * (1 - alpha_bar_prev) / (1 - alpha_bar_t)
            x = mean + posterior_var.sqrt() * torch.randn_like(x)
        else:
            # Final step — no noise added, this is our finished synthetic patient
            x = x0_pred

    return x.cpu().numpy()


def sample_categoricals( X_train: pd.DataFrame, cat_cols: list, n_samples: int, y_train: pd.Series = None, target_class: int = 1,) -> pd.DataFrame:
    """
    Get the categorical columns for the synthetic patients.

    The model only generates numeric columns. For the 33 categorical columns
    (things like race, diagnosis category, medication flags) we sample
    directly from real readmitted patients in the training set.
    This preserves realistic categorical patterns for the minority class.
    """
    pool = X_train[y_train == target_class][cat_cols] if y_train is not None else X_train[cat_cols]
    sampled = pool.sample(n=n_samples, replace=True).reset_index(drop=True)
    print(f"Sampled {n_samples} categorical rows from {len(pool)} real readmitted patients.")
    return sampled


def generate_patients( model_path: str, config_path: str, X_train_path: str, y_train_path: str, n_samples: int = 100, target_class: int = 1,) -> pd.DataFrame:
    """
    Full generation pipeline — returns a DataFrame of synthetic patients.

    Three steps:
    1. Run reverse diffusion to generate the 9 numeric features
    2. Sample the 33 categorical features from real minority class patients
    3. Combine into one DataFrame that matches the original dataset structure
    """
    model, cfg = load_model_and_config(model_path, config_path)

    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path).squeeze()

    print(f"\nGenerating {n_samples} synthetic patients (class={target_class})")

    # Generate the 9 numeric columns via reverse diffusion
    numeric_array = generate_numeric(model, cfg, n_samples, target_class)
    numeric_cols  = [X_train.columns[i] for i in cfg['numeric_indices']]
    numeric_df    = pd.DataFrame(numeric_array, columns=numeric_cols)

    # Sample the 33 categorical columns from real training data
    cat_cols = [X_train.columns[i] for i in cfg['cat_indices']]
    cat_df   = sample_categoricals(X_train, cat_cols, n_samples, y_train, target_class)

    # Combine and reorder columns to match the original dataset
    synthetic = pd.concat([numeric_df, cat_df], axis=1)
    synthetic = synthetic[X_train.columns]

    print(f"\nSynthetic.shape = {synthetic.shape}")
    return synthetic


if __name__ == '__main__':
    synthetic_df = generate_patients(model_path   = 'models/tabddpm_best.pt', config_path  = 'models/diffusion_config.pkl',
        X_train_path = 'data/processed/X_train.csv', y_train_path = 'data/processed/y_train.csv', n_samples = 50,)
    synthetic_df.to_csv('data/synthetic/synthetic_patients_50.csv', index=False)

