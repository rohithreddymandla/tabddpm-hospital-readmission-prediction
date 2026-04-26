"""
tabddpm_model.py — TabDDPM Denoiser Architecture
Owner: Jiten Bhalavat

Usage (Rohith's training loop):
    from tabddpm_model import TabDDPMDenoiser
    import pickle

    with open('../models/diffusion_config.pkl', 'rb') as f:
        cfg = pickle.load(f)

    model = TabDDPMDenoiser(
        num_numeric       = len(cfg['numeric_indices']),
        cat_cardinalities = cfg['cat_num_classes'],
        numeric_indices   = cfg['numeric_indices'],
        cat_indices       = cfg['cat_indices'],
    )

    # Forward pass: model(x_noisy, t, y) -> noise_pred  shape (batch, 9)
    # Loss:         F.mse_loss(noise_pred, actual_noise)
"""

import math
import torch
import torch.nn as nn


class SinusoidalTimestepEmbedding(nn.Module):
    """
    Sinusoidal positional encoding for diffusion timesteps.
    Produces a unique d-dimensional vector for each integer timestep 0..T-1.
    Formulation from Ho et al. (DDPM, 2020).
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (batch,) integer timesteps
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000)
            * torch.arange(half, dtype=torch.float32, device=t.device)
            / (half - 1)
        )
        args = t[:, None].float() * freqs[None]   # (batch, half)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (batch, dim)


class ResidualBlock(nn.Module):
    """
    Feed-forward residual block with Adaptive Layer Normalisation (AdaLN).

    The conditioning vector (timestep + class) modulates every hidden unit
    via a learned scale and shift, then the result is added back to the input
    as a residual connection.

    Near-identity init: linear2 is zero-initialised so the block starts as a
    pure skip connection.  cond_proj keeps random weights so gradients flow to
    linear2 immediately and conditioning activates from the first training step.
    """
    def __init__(self, dim: int, cond_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm      = nn.LayerNorm(dim)
        self.linear1   = nn.Linear(dim, dim * 2)
        self.act       = nn.SiLU()
        self.linear2   = nn.Linear(dim * 2, dim)
        self.dropout   = nn.Dropout(dropout)
        self.cond_proj = nn.Linear(cond_dim, dim * 2)
        nn.init.zeros_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        scale, shift = self.cond_proj(cond).chunk(2, dim=-1)
        h = h * (1.0 + scale) + shift   # AdaLN modulation
        h = self.act(self.linear1(h))
        h = self.dropout(h)
        h = self.linear2(h)
        return x + h                     # residual skip


class TabDDPMDenoiser(nn.Module):
    """
    Class-conditioned MLP denoiser for TabDDPM.

    Given a noisy patient row x_noisy, a diffusion timestep t, and a class
    label y, predicts the Gaussian noise that was added to the numeric columns.
    Rohith's training loop should minimise F.mse_loss(noise_pred, actual_noise).

    Architecture:
        x_noisy (batch, 42)
            numeric slice  -> as-is
            cat slice      -> per-column nn.Embedding tables -> concat
            concat         -> Linear -> SiLU  (input_proj, dim=256)

        t (batch,) -> SinusoidalTimestepEmbedding(128) -> 2-layer MLP -> t_emb (256)
        y (batch,) -> nn.Embedding(2, 64) -> Linear -> SiLU           -> c_emb (256)

        cond = t_emb + c_emb

        h -> ResidualBlock x4 (AdaLN with cond) -> LayerNorm -> Linear
        output: noise_pred (batch, 9)

    Args:
        num_numeric       : number of continuous columns (9)
        cat_cardinalities : list of vocab sizes K for each categorical column
        numeric_indices   : int positions of numeric columns in the 42-col row
        cat_indices       : int positions of categorical columns
        hidden_dim        : MLP width (default 256)
        depth             : number of residual blocks (default 4)
        time_emb_dim      : sinusoidal embedding size (default 128)
        class_emb_dim     : class label embedding size (default 64)
        dropout           : dropout rate in residual blocks (default 0.1)
    """

    def __init__(
        self,
        num_numeric: int,
        cat_cardinalities: list,
        numeric_indices: list,
        cat_indices: list,
        hidden_dim: int = 256,
        depth: int = 4,
        time_emb_dim: int = 128,
        class_emb_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.numeric_indices = numeric_indices
        self.cat_indices     = cat_indices
        self.num_numeric     = num_numeric

        # Categorical embeddings — one table per column
        # dim heuristic: max(2, min(16, (K+1)//2))
        cat_emb_dims = [max(2, min(16, (K + 1) // 2)) for K in cat_cardinalities]
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(K, d) for K, d in zip(cat_cardinalities, cat_emb_dims)
        ])
        total_cat_emb = sum(cat_emb_dims)

        # Input projection: numeric + cat_emb -> hidden_dim
        self.input_proj = nn.Sequential(
            nn.Linear(num_numeric + total_cat_emb, hidden_dim),
            nn.SiLU(),
        )

        # Timestep embedding: sinusoidal -> 2-layer MLP -> hidden_dim
        self.time_sin = SinusoidalTimestepEmbedding(time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Class embedding: binary label -> hidden_dim
        self.class_emb  = nn.Embedding(2, class_emb_dim)
        self.class_proj = nn.Sequential(
            nn.Linear(class_emb_dim, hidden_dim),
            nn.SiLU(),
        )

        # Residual blocks (AdaLN conditioning)
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, hidden_dim, dropout)
            for _ in range(depth)
        ])

        # Output head: predict noise for the 9 numeric columns
        self.output_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_numeric),
        )

    def forward(
        self,
        x_noisy: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x_noisy : (batch, 42)  full noisy patient row (numeric + categorical)
            t       : (batch,)     integer diffusion timesteps 0..T-1
            y       : (batch,)     class labels  0 = not readmitted, 1 = <30 days

        Returns:
            noise_pred : (batch, 9)  predicted Gaussian noise for numeric columns
        """
        # Encode numeric and categorical inputs
        x_num = x_noisy[:, self.numeric_indices].float()   # (batch, 9)
        x_cat = x_noisy[:, self.cat_indices].long()        # (batch, 33)

        cat_emb = torch.cat(
            [emb(x_cat[:, i]) for i, emb in enumerate(self.cat_embeddings)],
            dim=-1,
        )                                                   # (batch, total_cat_emb)

        h = self.input_proj(torch.cat([x_num, cat_emb], dim=-1))  # (batch, hidden_dim)

        # Build conditioning vector: timestep + class
        t_emb = self.time_mlp(self.time_sin(t))            # (batch, hidden_dim)
        c_emb = self.class_proj(self.class_emb(y.long()))  # (batch, hidden_dim)
        cond  = t_emb + c_emb                              # (batch, hidden_dim)

        # Residual blocks with AdaLN conditioning
        for block in self.blocks:
            h = block(h, cond)

        return self.output_head(h)                         # (batch, num_numeric)
