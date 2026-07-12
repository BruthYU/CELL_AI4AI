from __future__ import annotations

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class FinetuneVCICountsDecoder(nn.Module):
    """
    Optional dependency from the original state codebase.
    We keep the symbol to preserve the original module layout, but avoid hard-importing
    missing VCI/finetune dependencies at import time.
    """

    def __init__(self, *args, **kwargs):
        raise ImportError(
            "FinetuneVCICountsDecoder requires the original VCI finetune dependencies which are not "
            "vendored into nemo_cellflow. Set `finetune_vci_decoder: false` (recommended) or port "
            "the VCI decoder stack into `nemo_cellflow/models/STATE`."
        )


class LatentToGeneDecoder(nn.Module):
    """
    A simple MLP decoder to map latent embeddings back to gene space.
    """

    def __init__(
        self,
        latent_dim: int,
        gene_dim: int,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.1,
        residual_decoder: bool = False,
    ):
        super().__init__()
        hidden_dims = hidden_dims or [512, 1024]
        self.residual_decoder = residual_decoder

        if residual_decoder:
            self.blocks = nn.ModuleList()
            input_dim = latent_dim
            for h in hidden_dims:
                self.blocks.append(
                    nn.Sequential(
                        nn.Linear(input_dim, h),
                        nn.LayerNorm(h),
                        nn.GELU(),
                        nn.Dropout(dropout),
                    )
                )
                input_dim = h
            self.final_layer = nn.Sequential(nn.Linear(input_dim, gene_dim), nn.ReLU())
        else:
            layers: list[nn.Module] = []
            input_dim = latent_dim
            for h in hidden_dims:
                layers.extend([nn.Linear(input_dim, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)])
                input_dim = h
            layers.extend([nn.Linear(input_dim, gene_dim), nn.ReLU()])
            self.decoder = nn.Sequential(*layers)

    def gene_dim(self):
        if self.residual_decoder:
            return self.final_layer[0].out_features
        for m in reversed(self.decoder):
            if isinstance(m, nn.Linear):
                return m.out_features
        return None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.residual_decoder:
            block_outputs = []
            current = x
            for i, block in enumerate(self.blocks):
                out = block(current)
                if i >= 1 and i % 2 == 1:
                    out = out + block_outputs[i - 1]
                block_outputs.append(out)
                current = out
            return self.final_layer(current)
        return self.decoder(x)

