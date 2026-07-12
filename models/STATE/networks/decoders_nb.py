# models/decoders_nb.py (ported)
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import NegativeBinomial


class NBDecoder(nn.Module):
    """
    scVI-style decoder that maps a latent embedding (optionally with batch covariates)
    to parameters of a Negative Binomial distribution over counts.
    """

    def __init__(
        self,
        latent_dim: int,
        gene_dim: int,
        hidden_dims=list([1024, 256, 256]),
        dropout: float = 0.0,
        use_zero_inflation: bool = False,
    ):
        super().__init__()
        modules = []
        in_features = latent_dim
        for h in hidden_dims:
            modules += [
                nn.Linear(in_features, h),
                nn.LayerNorm(h),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            in_features = h
        self.encoder = nn.Sequential(*modules)

        self.skip = nn.Identity() if in_features == latent_dim else nn.Linear(latent_dim, in_features, bias=False)
        self.post_norm = nn.LayerNorm(in_features)

        self.px_scale = nn.Linear(in_features, gene_dim)
        self.l_encoder = nn.Linear(in_features, 1)

        self.log_theta = nn.Parameter(torch.randn(gene_dim))

        self.use_zero_inflation = use_zero_inflation
        if use_zero_inflation:
            self.px_dropout = nn.Linear(in_features, gene_dim)

    @property
    def theta(self):
        return F.softplus(self.log_theta)

    def forward(self, z: torch.Tensor, log_library: torch.Tensor | None = None):
        flat = False
        if z.dim() == 3:
            B, S, D = z.shape
            z = z.reshape(-1, D)
            flat = True

        h = self.encoder(z)
        h = self.post_norm(h + self.skip(z))

        if log_library is None:
            log_library = self.l_encoder(h)
        px_scale = F.softplus(self.px_scale(h))
        mu = torch.exp(log_library) * px_scale

        if self.use_zero_inflation:
            pi = torch.sigmoid(self.px_dropout(h))
            if flat:
                mu = mu.reshape(B, S, -1)
                pi = pi.reshape(B, S, -1)
                return mu, self.theta, pi
            return mu, self.theta, pi

        if flat:
            mu = mu.reshape(B, S, -1)
            return mu, self.theta
        return mu, self.theta

    def gene_dim(self) -> int:
        return self.px_scale.out_features


def nb_nll(x, mu, theta, eps: float = 1e-6):
    logits = (mu + eps).log() - (theta + eps).log()
    dist = NegativeBinomial(total_count=theta, logits=logits)
    return -dist.log_prob(x).mean()

