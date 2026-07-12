import torch
import torch.nn as nn
import torch.nn.functional as F
from geomloss import SamplesLoss
from scvi.distributions import NegativeBinomial
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from geomloss import SamplesLoss
from torchmetrics.functional import pearson_corrcoef, r2_score
from dataclasses import dataclass




class Classification_Loss(nn.Module):
    def __init__(self, reduction: str = "mean", scale: float = 1.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(reduction=reduction)
        self.scale = scale
    def forward(self, logits, labels) -> tuple[torch.Tensor, torch.Tensor]:
        cls_loss = self.ce(logits, labels)
        return cls_loss * self.scale

@dataclass
class SlicedWassersteinDistance:
    """Callable helper that computes the sliced Wasserstein distance."""

    n_proj: int = 64

    def __call__(self, x: torch.Tensor, y: torch.Tensor, n_proj: Optional[int] = None) -> torch.Tensor:
        if x.shape != y.shape:
            raise ValueError("Input tensors must have identical shapes")

        num_proj = n_proj or self.n_proj
        batch_size, _, latent_dim = x.shape
        projections = torch.randn(num_proj, latent_dim, device=x.device)
        projections = projections / projections.norm(dim=1, keepdim=True)

        x_proj = x @ projections.t()
        y_proj = y @ projections.t()

        x_sorted, _ = torch.sort(x_proj, dim=1)
        y_sorted, _ = torch.sort(y_proj, dim=1)
        return ((x_sorted - y_sorted) ** 2).mean()



class Reconstruction_Loss_NB(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, nb_mean, nb_dispersion, targets, mask):
        nb_dist = NegativeBinomial(mu=nb_mean, theta=nb_dispersion)
        recon_loss_all = -nb_dist.log_prob(targets)
        if mask is not None:
            mask_f = mask.float()
            masked_count = mask_f.sum()
            if masked_count > 0:
                recon_loss = (recon_loss_all * mask_f).sum() / masked_count
            else:
                recon_loss = torch.tensor(0.0, device=targets.device, dtype=recon_loss_all.dtype)
        else:
            recon_loss = recon_loss_all.mean()
        return recon_loss, recon_loss_all



class SW_Loss_NB(nn.Module):
    def __init__(self, n_proj):
        super().__init__()
        self.sw_distance = SlicedWassersteinDistance(n_proj=n_proj)

    def forward(
        self,
        masked_embeddings: torch.Tensor,
        *,
        subsample_size: Optional[int] = None,
        min_size: int = 32,
        max_size: int = 128,
        n_proj: Optional[int] = None,
    ) -> torch.Tensor:
        n_cells = masked_embeddings.shape[1]
        device = masked_embeddings.device

        if subsample_size is None:
            upper = min(max_size, n_cells)
            lower = min(min_size, max(upper, 1))
            if lower == 0:
                return torch.tensor(0.0, device=device, dtype=masked_embeddings.dtype)
            if lower == upper:
                k = lower
            else:
                k = torch.randint(lower, upper + 1, (), device=device).item()
        else:
            k = max(1, min(subsample_size, n_cells))

        idx = torch.randperm(n_cells, device=device)[:k]
        masked_embeddings_subsampled = masked_embeddings[:, idx]
        prior_samples = torch.randn_like(masked_embeddings_subsampled)
        centered_embeddings = masked_embeddings_subsampled - masked_embeddings_subsampled.mean(
            dim=1, keepdim=True
        )

        if n_proj is None:
            return self.sw_distance(centered_embeddings, prior_samples)
        return self.sw_distance(centered_embeddings, prior_samples, n_proj=n_proj)


class WassersteinLoss(nn.Module):
    """
    Implements Wasserstein distance loss for distributions represented by logits.
    This implementation supports both 1D and 2D Wasserstein distance calculations.
    """

    def __init__(self, p=1, reduction="mean"):
        """
        Args:
            p (int): Order of Wasserstein distance (1 or 2)
            reduction (str): 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.p = p
        self.reduction = reduction

    def forward(self, p, q):
        """
        Compute Wasserstein distance between predicted and target distributions.

        Args:
            logits (torch.Tensor): Predicted logits of shape (batch_size, num_classes)
            target (torch.Tensor): Target probabilities of shape (batch_size, num_classes)
                                 or class indices of shape (batch_size,)

        Returns:
            torch.Tensor: Computed Wasserstein distance
        """

        q = torch.nan_to_num(q, nan=0.0)
        # Convert logits to probabilities
        pred_probs = F.softmax(p, dim=-1)
        q = F.softmax(q, dim=-1)

        # Compute cumulative distribution functions (CDFs)
        pred_cdf = torch.cumsum(pred_probs, dim=-1)
        target_cdf = torch.cumsum(q, dim=-1)

        max_len = max(pred_cdf.size(1), target_cdf.size(1))
        if pred_cdf.size(1) < max_len:
            pred_cdf = F.pad(pred_cdf, (0, max_len - pred_cdf.size(1)), "constant", 0)
        if target_cdf.size(1) < max_len:
            target_cdf = F.pad(target_cdf, (0, max_len - target_cdf.size(1)), "constant", 0)

        # Compute Wasserstein distance
        wasserstein_dist = torch.abs(pred_cdf - target_cdf).pow(self.p)
        wasserstein_dist = wasserstein_dist.sum(dim=-1)

        # Apply reduction if specified
        if self.reduction == "mean":
            return wasserstein_dist.mean()
        elif self.reduction == "sum":
            return wasserstein_dist.sum()
        return wasserstein_dist


class KLDivergenceLoss(nn.Module):
    def __init__(self, apply_normalization=False, epsilon=1e-10):
        super().__init__()
        self.apply_normalization = apply_normalization
        self.epsilon = epsilon

    def forward(self, p, q):
        q = torch.nan_to_num(q, nan=0.0)
        p = torch.nan_to_num(p, nan=0.0)

        max_len = max(p.size(1), q.size(1))
        if p.size(1) < max_len:
            p = F.pad(p, (0, max_len - p.size(1)), "constant", 0)
        if q.size(1) < max_len:
            q = F.pad(q, (0, max_len - q.size(1)), "constant", 0)

        if self.apply_normalization:
            p = F.softmax(p, dim=-1)
            q = F.softmax(q, dim=-1)

        return torch.sum(p * torch.log(p / q))

from geomloss import SamplesLoss
class MMDLoss(nn.Module):
    def __init__(self, kernel="energy", blur=0.05, scaling=0.5, downsample=1, scale=1):
        super().__init__()
        self.mmd_loss = SamplesLoss(loss=kernel, blur=blur, scaling=scaling)
        self.downsample = downsample
        self.scale = scale

    def forward(self, input, target):
        # input = input.reshape(-1, self.downsample, input.shape[-1])
        # target = target.reshape(-1, self.downsample, target.shape[-1])

        loss = self.mmd_loss(input, target)
        return loss.mean()*self.scale

class Transpose_MMDLoss(nn.Module):
    def __init__(self, kernel="energy", blur=0.05, scaling=0.5, scale=1.0):
        super().__init__()
        self.mmd_loss = SamplesLoss(loss=kernel, blur=blur, scaling=scaling)
        self.scale = scale

    def forward(self, input, target):
        """
        input:  [B, N, D]
        target: [B, N, D] or [B, M, D]
        """

        # 1. 转置最后两维
        input_t = input.transpose(-1, -2)   # [B, D, N]
        target_t = target.transpose(-1, -2) # [B, D, M]

        # 2. reshape 成 SamplesLoss 需要的 [batch, samples, dim]
        #    此时“samples = D（feature维）”
        loss = self.mmd_loss(input_t, target_t)

        return loss.mean() * self.scale


class MSELoss(nn.Module):
    def __init__(self, batch=True, reduction="mean", scale=100.0):
        super().__init__()
        self.mse_loss = nn.MSELoss(reduction=reduction)
        self.batch = batch
        self.scale = scale

    def forward(self, input, target):
        if self.batch:
            input = input.mean(dim=1)
            target = target.mean(dim=1)
        loss = self.mse_loss(input, target)
        return loss * self.scale

class PairMSELoss(nn.Module):
    def __init__(self, reduction="mean", scale=100.0):
        super().__init__()
        self.mse_loss = nn.MSELoss(reduction=reduction)
        self.scale = scale

    def forward(self, input, target):
        loss = self.mse_loss(input, target)
        return loss * self.scale

class MAELoss(nn.Module):
    def __init__(self, batch=True, reduction="mean", scale=100.0):
        super().__init__()
        self.mae_loss = nn.L1Loss(reduction=reduction)
        self.batch = batch
        self.scale = scale
    def forward(self, input, target):
        if self.batch:
            input = input.mean(dim=1)
            target = target.mean(dim=1)
        loss = self.mae_loss(input, target)
        return loss * self.scale


class TabularLoss(nn.Module):
    def __init__(self, shared=128, downsample=1):
        super().__init__()
        self.shared = shared
        self.downsample = downsample

        self.gene_loss = SamplesLoss(loss="energy")
        self.cell_loss = SamplesLoss(loss="energy")

    def forward(self, input, target):
        input = input.reshape(-1, self.downsample, input.shape[-1])
        target = target.reshape(-1, self.downsample, target.shape[-1])
        gene_mmd = self.gene_loss(input, target).nanmean()

        # cell_mmd should only be on the shared genes, and match scale to mse loss
        cell_inputs = input[:, :, -self.shared :]
        cell_targets = target[:, :, -self.shared :]

        # need to reshape each from (B, self.downsample, F) to (F, self.downsample, B)
        cell_inputs = cell_inputs.transpose(2, 0)
        cell_targets = cell_targets.transpose(2, 0)
        cell_mmd = self.cell_loss(cell_inputs, cell_targets).nanmean()

        final_loss = torch.tensor(0.0).to(cell_mmd.device)
        if not gene_mmd.isnan():
            final_loss += gene_mmd
        if not cell_mmd.isnan():
            final_loss += cell_mmd

        return final_loss


class Delta_Pearson(nn.Module):
    def __init__(self, scale=1.0):
        super().__init__()
        self.scale = scale

    def forward(self, input, pred, target):
        B = input.shape[0]
        input = input.double()
        pred = pred.double()
        target = target.double()

        change_pred = pred - input
        change_target = target - input
        pearson_list = []
        for batch_id in range(B):
            x0 = change_target[batch_id].mean(0)
            x1 = change_pred[batch_id].mean(0)
            pearson_list.append(pearson_corrcoef(x0, x1))
        pearson_mean_lfc = torch.stack(pearson_list).mean()
        return pearson_mean_lfc * self.scale

class Pearson(nn.Module):
    def __init__(self, scale=1.0):
        super().__init__()
        self.scale = scale

    def forward(self, pred, target):
        B = pred.shape[0]
        pred = pred.double()
        target = target.double()

        

        pearson_list = []
        for batch_id in range(B):
            x0 = pred[batch_id].mean(0)
            x1 = target[batch_id].mean(0)
            pearson_list.append(pearson_corrcoef(x0, x1))
        pearson_mean_lfc = torch.stack(pearson_list).mean()
        return pearson_mean_lfc * self.scale

class R2_Score:
    def __init__(self, scale=1.0):
        super().__init__()
        self.scale = scale
    def forward(self, input, pred, target):
        B = input.shape[0]
        change_pred = pred - input
        change_target = target - input
        score_list = []
        for batch_id in range(B):
            x0 = change_target[batch_id].mean(0)
            x1 = change_pred[batch_id].mean(0)
            score_list.append(r2_score(x0, x1))     
        mean_score = torch.stack(score_list).mean()
        return mean_score * self.scale