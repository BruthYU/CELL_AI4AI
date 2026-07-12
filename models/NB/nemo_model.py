import math
import torch
from tqdm import tqdm

from models.NB.flow.interpolant import Interpolant
from models.NB.networks._nemo_vf import ConditionalVelocityField

from bionemo.llm.model.biobert.lightning import get_batch_on_this_context_parallel_rank
from bionemo.llm.api import MegatronLossType, MegatronModelType
from models.nemo_model_base import ModelInterfaceBase
from typing import Iterator, Dict, Tuple
from torch import Tensor
from omegaconf import OmegaConf
from bionemo.llm.model.loss import _Nemo2CompatibleLossReduceMixin
from nemo.collections.nlp.modules.common.megatron.utils import average_losses_across_data_parallel_group
from nemo.lightning.megatron_parallel import (
    MegatronLossReduction,
)
from megatron.core.transformer.transformer_config import TransformerConfig
default_transformer_config = TransformerConfig(
    num_layers=2,
    hidden_size=12,
    num_attention_heads=4,
    use_cpu_initialization=True,
    pipeline_dtype=torch.float32,
)
from _metrics import (
    MSELoss, 
    MAELoss, 
    PairMSELoss,
    Delta_Pearson,
)

class metrics:
    def __init__(self):
        self.mse = PairMSELoss(reduction="mean", scale=10.0)
        self.mae = MAELoss(batch=True, reduction="mean", scale=10.0)
        self.delta_pearson = Delta_Pearson(scale=1.0)
        self.flow_mse = MSELoss(batch=False, reduction="mean", scale=10.0)     

class FM_LossWithReduction(_Nemo2CompatibleLossReduceMixin, MegatronLossReduction):
    """Custom loss reduction class for Nemo."""
    
    def __init__(self, validation_step: bool = False, val_drop_last: bool = True, add_sop_loss: bool = True) -> None:
        super().__init__()
        self.validation_step = validation_step
        self.val_drop_last = val_drop_last
        self.add_sop_loss = add_sop_loss

    def forward(
        self,
        batch: Dict[str, Tensor],
        forward_out: Dict[str, Tensor],
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        unreduced_loss = forward_out['loss']
        # Normal case: average loss across data parallel group
        reduced = average_losses_across_data_parallel_group([unreduced_loss])
        return unreduced_loss, {'avg': reduced}


class NB_Nemo_Model(ModelInterfaceBase[MegatronModelType, MegatronLossType]):
    def __init__(self, conf, model_transform=None, configure_init_model_parallel=False):
        super().__init__(
            model_transform=model_transform,
            configure_init_model_parallel=configure_init_model_parallel,
        )
        if OmegaConf.is_config(conf):
            conf = OmegaConf.to_container(conf, resolve=True)
        
        self.conf = conf
        self.config = default_transformer_config
        
        self.model_conf = conf["prepare_model"]

        self.configure_model()

        # flow utils
        self.flow_utils = Interpolant(conf["Interpolant"])
        self.metric_utils = metrics()

        
       


    def data_step(self, dataloader_iter: Iterator) -> Dict:
        batch = next(dataloader_iter)
        if isinstance(batch, tuple) and len(batch) == 3:
            _batch = batch[0]
        else:
            _batch = batch
        def to_cuda(x):
            if isinstance(x, torch.Tensor):
                return x.cuda(non_blocking=True)
            else:
                return x
        _batch = {k: to_cuda(v) for k, v in _batch.items()}
        return get_batch_on_this_context_parallel_rank(_batch)  
    
    def configure_model(self):
        # Velocity Field Model
        self.module = ConditionalVelocityField(self.model_conf, self.config)

        # dict-access version
        if self.model_conf["megatron_ckpt"] is not None:
            self.load_from_megatron_ckpt(self.model_conf["megatron_ckpt"])

    def load_from_megatron_ckpt(self, ckpt_dir: str) -> None:
        """Load the model from a Megatron checkpoint directory."""
        # Initialize process group if needed
        self.init_process_group_if_needed(backend="gloo")
        
        # Load the Megatron checkpoint
        from megatron.core.dist_checkpointing import load_plain_tensors
        full_state_dict = load_plain_tensors(ckpt_dir)
        
        # Extract only the model state dict from the Lightning checkpoint
        model_state_dict = {}
        for key, value in full_state_dict.items():
            if key.startswith('module.') and not key.startswith('optimizer.'):
                model_state_dict[key] = value
        
        # Load the model state dict into the model
        self.load_state_dict(model_state_dict, strict=True)
        print(f"✅ Megatron checkpoint loaded successfully: {ckpt_dir}")
    
    def init_process_group_if_needed(self, backend="gloo"):
            import torch.distributed as dist
            import os
            
            # 动态查找可用端口
            available_port = self.find_available_port()
            
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = str(available_port)
            os.environ['RANK'] = '0'
            os.environ['WORLD_SIZE'] = '1'

            if not dist.is_initialized():
                # 单进程用 gloo 足够
                dist.init_process_group(backend=backend, init_method="env://", world_size=1, rank=0)
                print(f"✅ 分布式进程组初始化成功，使用端口: {available_port}")
    
    def find_available_port(self, start_port=12355, max_attempts=100):
        """
        查找可用端口，从start_port开始尝试
        """
        import socket
        import random
        
        for _ in range(max_attempts):
            try:
                # 随机选择一个端口范围，避免冲突
                port = start_port + random.randint(0, 1000)
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('localhost', port))
                    return port
            except OSError:
                continue
        
        # 如果都不可用，使用系统分配的端口
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('localhost', 0))
            return s.getsockname()[1]

    # Megatron Required
    def loss_reduction(self, *args, **kwargs):
        return FM_LossWithReduction(**kwargs)

    def forward_step(self, batch, mode="train"):
        if mode == "train":
            return self.loss_fn(batch)
        else:
            return self.eval_fn(batch)
    

    def training_step(self, batch, batch_idx=None):
        forward_out = self.forward_step(batch, mode="train")
        log_info = self.add_mode_prefix(forward_out['log_info'], "train")
        self.log_dict(log_info, on_step=True, on_epoch=True, prog_bar=True)
        return forward_out
    
    def validation_step(self, batch, batch_idx=None):
        
        forward_out = self.forward_step(batch, mode="val")
        log_info = self.add_mode_prefix(forward_out['log_info'], "val")
        print(log_info)
        self.log_dict(log_info, on_step=True, on_epoch=True, prog_bar=False)
        return forward_out

    def predict_step(self, batch, batch_idx=None):
        return self.eval_fn(batch, test_mode=True)



    def save_to_torch_ckpt(self, ckpt_dir: str, out_path: str) -> None:
        """Save the model to a torch checkpoint."""
        '''
        self.save_to_torch_ckpt('/mnt/shared-storage-user/gaozhangyang/workspace/FoldCompression/results/struct_compress/eval_nn10_in_str_out_str_lr1e5_10Xsteps_droptoken/checkpoints/epoch=0-step=4012999-consumed_samples=64208000.0/weights', '/mnt/shared-storage-user/gaozhangyang/workspace/FoldCompression/results/struct_compress/eval_nn10_in_str_out_str_lr1e5_10Xsteps_droptoken/checkpoints/10Xsteps_droptoken_step_4012999_weights.pt')
        '''
        
        # 1. 生成 sharded_state_dict
        sharded_sd = self.module.sharded_state_dict()  # <— 一定要在 parallel 初始化后调用
        ckpt = self.trainer.strategy.checkpoint_io.load_checkpoint(
                str(ckpt_dir),
                sharded_state_dict=sharded_sd,            # <<< 关键：这里不能省略
            )  # 底层会调用 dist_checkpointing.load(sharded_state_dict=…, …)
        torch.save(ckpt, out_path)
        print(f"✅ 转换成功：{out_path}")





    
    def reshape_t(self, t, n):
        if t.dim()==0:
            t = t[None].expand(n) 
        return t
    
    def add_mode_prefix(self, data_dict: Dict, mode: str) -> Dict:
        assert mode in ["train", "val", "test"], f"mode must be 'train', 'val', or 'test', got {mode}"
        return {f"{mode}_{key}": value for key, value in data_dict.items()}
    
    def forward(self, x_t, cond_batch):
        return self.module(x_t, cond_batch)

    def _reshape_expression_batch(self, batch, require_target: bool = True):
        seq_len = int(self.conf["dataset"]["cell_sentence_len"])
        cell_dim = int(self.model_conf["cell_dim"])
        ctrl = batch["ctrl_cell_emb"].float()
        pert = batch.get("pert_cell_emb")
        if pert is not None:
            pert = pert.float()

        if ctrl.dim() == 3:
            batch_size = ctrl.shape[0]
            seq_len = ctrl.shape[1]
            src_batch = ctrl
        else:
            total = ctrl.shape[0]
            if total % seq_len != 0:
                raise ValueError(
                    f"ctrl_cell_emb first dim ({total}) must be divisible by seq_len ({seq_len})."
                )
            batch_size = total // seq_len
            src_batch = ctrl.reshape(batch_size, seq_len, cell_dim)

        if pert is None:
            if require_target:
                raise KeyError("NB training/validation requires pert_cell_emb in batch.")
            return src_batch, None, batch_size, seq_len

        if pert.dim() == 3:
            tgt_batch = pert
        else:
            tgt_batch = pert.reshape(batch_size, seq_len, cell_dim)
        return src_batch, tgt_batch, batch_size, seq_len

    def _pseudo_counts(self, x_log1p: torch.Tensor) -> torch.Tensor:
        return torch.expm1(x_log1p.float()).clamp_min(0.0)

    def _library_size(self, x_log1p: torch.Tensor) -> torch.Tensor:
        mode = self.model_conf.get("library_size_mode", "pseudo_count_sum")
        pseudo_count = self._pseudo_counts(x_log1p)
        if mode == "pseudo_count_sum":
            return pseudo_count.sum(dim=-1, keepdim=True).clamp_min(1.0)
        if mode == "constant_gene_count":
            return torch.full_like(pseudo_count[..., :1], float(pseudo_count.shape[-1]))
        raise ValueError(f"Unknown library_size_mode: {mode}")

    def _prediction_lib_size(self, src_batch: torch.Tensor, tgt_batch: torch.Tensor | None) -> torch.Tensor:
        source = self.model_conf.get("prediction_lib_size_source", "control")
        if source == "target":
            if tgt_batch is not None:
                return self._library_size(tgt_batch)
            return self._library_size(src_batch)
        if source == "control":
            return self._library_size(src_batch)
        raise ValueError(f"Unknown prediction_lib_size_source: {source}")

    def _rectangular_gene_mask(self, x: torch.Tensor) -> torch.Tensor | None:
        mask_rate_min = float(self.model_conf.get("rect_mask_rate_min", 0.15))
        mask_rate_max = float(self.model_conf.get("rect_mask_rate_max", 0.30))
        if mask_rate_max <= 0.0:
            return None
        mask_rate_min = max(0.0, min(mask_rate_min, mask_rate_max))
        b, s, g = x.shape
        rate = torch.empty((), device=x.device).uniform_(mask_rate_min, mask_rate_max).item()
        n_mask = min(g, max(1, int(round(g * rate))))
        gene_idx = torch.randperm(g, device=x.device)[:n_mask]
        mask = torch.zeros((b, s, g), dtype=torch.bool, device=x.device)
        mask[:, :, gene_idx] = True
        return mask

    def _nb_nll(
        self,
        mean: torch.Tensor,
        dispersion: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        eps = float(self.model_conf.get("nb_eps", 1e-8))
        mean = mean.clamp_min(eps)
        dispersion = dispersion.clamp_min(eps)
        target = target.to(dtype=mean.dtype).clamp_min(0.0)
        log_theta_mu = torch.log(dispersion + mean + eps)
        log_prob = (
            torch.lgamma(target + dispersion)
            - torch.lgamma(dispersion)
            - torch.lgamma(target + 1.0)
            + dispersion * (torch.log(dispersion + eps) - log_theta_mu)
            + target * (torch.log(mean + eps) - log_theta_mu)
        )
        nll = -log_prob
        if mask is None:
            return nll.mean()
        mask = mask.to(dtype=nll.dtype)
        return (nll * mask).sum() / mask.sum().clamp_min(1.0)

    def _endpoint_sliced_wasserstein_loss(
        self,
        pred_endpoint: torch.Tensor,
        target_endpoint: torch.Tensor,
    ) -> torch.Tensor:
        if pred_endpoint.shape != target_endpoint.shape:
            raise ValueError(
                "endpoint SW expects matching shapes, got "
                f"{tuple(pred_endpoint.shape)} and {tuple(target_endpoint.shape)}."
            )

        n_cells = pred_endpoint.shape[1]
        latent_dim = pred_endpoint.shape[-1]
        subsample_size = int(self.model_conf["endpoint_sw_subsample_size"])
        n_proj = int(self.model_conf["endpoint_sw_n_proj"])
        k = min(max(1, subsample_size), n_cells)

        if k < n_cells:
            cell_idx = torch.randperm(n_cells, device=pred_endpoint.device)[:k]
            pred_endpoint = pred_endpoint[:, cell_idx]
            target_endpoint = target_endpoint[:, cell_idx]

        pred_endpoint = pred_endpoint.float()
        target_endpoint = target_endpoint.float()
        projections = torch.randn(
            n_proj,
            latent_dim,
            device=pred_endpoint.device,
            dtype=pred_endpoint.dtype,
        )
        projections = projections / projections.norm(dim=1, keepdim=True).clamp_min(1e-8)

        pred_proj = pred_endpoint @ projections.t()
        target_proj = target_endpoint @ projections.t()
        pred_sorted, _ = torch.sort(pred_proj, dim=1)
        target_sorted, _ = torch.sort(target_proj, dim=1)
        return ((pred_sorted - target_sorted) ** 2).mean()

    def _decode_nb(self, z: torch.Tensor, lib_size: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.module.tgt_cellmap_decoder(z, lib_size=lib_size)

    def _sample_nb_counts(self, mean: torch.Tensor, dispersion: torch.Tensor) -> torch.Tensor:
        eps = float(self.model_conf.get("nb_eps", 1e-8))
        mean = mean.float().clamp_min(eps)
        dispersion = dispersion.float().clamp_min(eps)
        gamma_rate = (dispersion / mean).clamp_min(eps)
        poisson_rate = torch.distributions.Gamma(
            concentration=dispersion,
            rate=gamma_rate,
        ).sample()
        return torch.poisson(poisson_rate).to(dtype=mean.dtype)

    def _decode_log1p_prediction(
        self,
        z: torch.Tensor,
        pred_lib_size: torch.Tensor,
        src_encoded_map: torch.Tensor,
        src_batch: torch.Tensor,
        tgt_batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        nb_out = self._decode_nb(z, pred_lib_size)
        eval_mode = self.model_conf.get("eval_nb_mode", "sample")
        if eval_mode == "mean":
            return torch.log1p(nb_out["nb_mean"].clamp_min(0.0))
        if eval_mode != "sample":
            raise ValueError(f"Unknown eval_nb_mode: {eval_mode}. Expected 'sample' or 'mean'.")

        dispersion_source = self.model_conf.get("eval_nb_dispersion_source", "control_median")
        if dispersion_source == "predicted":
            sample_dispersion = nb_out["nb_dispersion"]
        elif dispersion_source == "control_median":
            src_nb = self._decode_nb(src_encoded_map, self._library_size(src_batch))
            sample_dispersion = src_nb["nb_dispersion"].median(dim=1, keepdim=True).values
        elif dispersion_source == "target_median" and tgt_batch is not None:
            tgt_encoded_map = self.module.tgt_cellmap_encoder(tgt_batch)
            tgt_nb = self._decode_nb(tgt_encoded_map, self._library_size(tgt_batch))
            sample_dispersion = tgt_nb["nb_dispersion"].median(dim=1, keepdim=True).values
        else:
            raise ValueError(
                f"Unknown eval_nb_dispersion_source: {dispersion_source}. "
                "Expected 'control_median', 'predicted', or validation-only 'target_median'."
            )

        sampled_counts = self._sample_nb_counts(nb_out["nb_mean"], sample_dispersion)
        return torch.log1p(sampled_counts.clamp_min(0.0))

    def _build_cond_batch(self, batch, batch_size: int, seq_len: int, device, t=None) -> Dict[str, torch.Tensor]:
        cond_batch = {}
        if t is not None:
            cond_batch["t"] = t
        cond_keys = self.conf["layers_before_pool"].keys()
        for k in cond_keys:
            cov = batch[k].to(device)
            if cov.dim() == 3:
                cond_batch[k] = cov
            elif cov.shape[0] == batch_size:
                if cov.dim() == 1:
                    cond_batch[k] = cov.unsqueeze(1).expand(-1, seq_len)
                else:
                    cond_batch[k] = cov.unsqueeze(1).expand(-1, seq_len, -1)
            elif cov.shape[0] == batch_size * seq_len:
                if cov.dim() == 1:
                    cond_batch[k] = cov.reshape(batch_size, seq_len)
                else:
                    cond_batch[k] = cov.reshape(batch_size, seq_len, -1)
            else:
                raise ValueError(
                    f"Unexpected shape for condition '{k}': {tuple(cov.shape)} "
                    f"(batch_size={batch_size}, seq_len={seq_len})."
                )
        return cond_batch

    def _flow_prediction(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        src_encoded_map: torch.Tensor,
        cond_batch: Dict[str, torch.Tensor],
    ):
        pred_type = self.model_conf.get("pred_type", "v")
        pred = self.module(x_t, cond_batch)
        if pred_type == "x":
            x1_pred = pred
            v_pred = x1_pred - src_encoded_map
        elif pred_type == "v":
            v_pred = pred
            one_minus_t = (1.0 - t).reshape(-1, 1, 1).to(dtype=x_t.dtype, device=x_t.device)
            x1_pred = x_t + one_minus_t * v_pred
        else:
            raise ValueError(f"Unknown pred_type: {pred_type}. Expected 'x' or 'v'.")
        return pred, x1_pred, v_pred

    def _tower_lr_scale(self) -> float:
        start = float(self.model_conf.get("tower_lr_scale_start", 1.0))
        end = float(self.model_conf.get("tower_lr_scale_end", 1.0))
        decay_epochs = float(self.model_conf.get("tower_lr_decay_epochs", 0.0))
        if decay_epochs <= 0.0:
            return end
        epoch = float(getattr(self, "current_epoch", 0))
        progress = min(1.0, max(0.0, epoch / decay_epochs))
        return end + 0.5 * (start - end) * (1.0 + math.cos(math.pi * progress))

    def on_before_optimizer_step(self, optimizer) -> None:
        scale = self._tower_lr_scale()
        for param in self.module.tower_parameters():
            if param.grad is not None:
                param.grad.mul_(scale)
        self.log("train_tower_lr_scale", scale, on_step=True, on_epoch=False, prog_bar=False)

    def loss_fn(self, batch):
        src_batch, tgt_batch, batch_size, seq_len = self._reshape_expression_batch(batch, require_target=True)
        device = src_batch.device

        src_encoded_map = self.module.tgt_cellmap_encoder(src_batch)
        tgt_encoded_map = self.module.tgt_cellmap_encoder(tgt_batch)

        x_t, v_t, t = self.flow_utils.interpolate(src_encoded_map, tgt_encoded_map, device)
        cond_batch = self._build_cond_batch(batch, batch_size, seq_len, device, t=t)
        _, x1_pred, v_pred = self._flow_prediction(x_t, t, src_encoded_map, cond_batch)

        x_loss = self.metric_utils.flow_mse(x1_pred, tgt_encoded_map)
        v_loss = self.metric_utils.flow_mse(v_pred, v_t)
        loss_type = self.model_conf.get("loss_type", "v")
        if loss_type == "x":
            flow_loss = x_loss
        elif loss_type == "v":
            flow_loss = v_loss
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}. Expected 'x' or 'v'.")

        rect_mask = self._rectangular_gene_mask(src_batch)
        src_masked = src_batch.masked_fill(rect_mask, 0.0) if rect_mask is not None else src_batch
        tgt_masked = tgt_batch.masked_fill(rect_mask, 0.0) if rect_mask is not None else tgt_batch

        src_recon_z = self.module.tgt_cellmap_encoder(src_masked)
        tgt_recon_z = self.module.tgt_cellmap_encoder(tgt_masked)

        src_counts = self._pseudo_counts(src_batch)
        tgt_counts = self._pseudo_counts(tgt_batch)
        src_lib_size = self._library_size(src_batch)
        tgt_lib_size = self._library_size(tgt_batch)

        src_nb = self._decode_nb(src_recon_z, src_lib_size)
        tgt_nb = self._decode_nb(tgt_recon_z, tgt_lib_size)
        src_recon_loss = self._nb_nll(src_nb["nb_mean"], src_nb["nb_dispersion"], src_counts, rect_mask)
        tgt_recon_loss = self._nb_nll(tgt_nb["nb_mean"], tgt_nb["nb_dispersion"], tgt_counts, rect_mask)

        endpoint_nb_weight = float(self.model_conf.get("endpoint_nb_weight", 0.1))
        if endpoint_nb_weight > 0.0:
            endpoint_nb = self._decode_nb(x1_pred, tgt_lib_size)
            endpoint_nb_loss = self._nb_nll(
                endpoint_nb["nb_mean"],
                endpoint_nb["nb_dispersion"],
                tgt_counts,
                None,
            )
        else:
            endpoint_nb_loss = torch.zeros((), device=device, dtype=flow_loss.dtype)

        endpoint_sw_weight = float(self.model_conf["endpoint_sw_weight"])
        if endpoint_sw_weight > 0.0:
            endpoint_sw_loss = self._endpoint_sliced_wasserstein_loss(x1_pred, tgt_encoded_map)
        else:
            endpoint_sw_loss = torch.zeros((), device=device, dtype=flow_loss.dtype)

        src_recon_weight = float(self.model_conf.get("src_recon_weight", 1.0))
        tgt_recon_weight = float(self.model_conf.get("tgt_recon_weight", 1.0))
        nb_loss_weight = float(self.model_conf.get("nb_loss_weight", 1.0))
        flow_loss_weight = float(self.model_conf.get("flow_loss_weight", 1.0))
        recon_loss = src_recon_weight * src_recon_loss + tgt_recon_weight * tgt_recon_loss
        final_loss = (
            flow_loss_weight * flow_loss
            + nb_loss_weight * recon_loss
            + endpoint_nb_weight * endpoint_nb_loss
            + endpoint_sw_weight * endpoint_sw_loss
        )

        mask_rate = 0.0 if rect_mask is None else rect_mask.float().mean().item()
        log_info = {
            "src_nb_nll": src_recon_loss.item(),
            "tgt_nb_nll": tgt_recon_loss.item(),
            "endpoint_nb_nll": endpoint_nb_loss.item(),
            "endpoint_sw_loss": endpoint_sw_loss.item(),
            "recon_loss": recon_loss.item(),
            "x_loss": x_loss.item(),
            "v_loss": v_loss.item(),
            "flow_loss": flow_loss.item(),
            "rect_mask_rate": mask_rate,
            "tower_lr_scale": self._tower_lr_scale(),
            "final_loss": final_loss.item(),
        }
        return {"loss": final_loss, "log_info": log_info}

    def eval_fn(self, batch, test_mode=False):
        src_batch, tgt_batch, batch_size, seq_len = self._reshape_expression_batch(
            batch,
            require_target=not test_mode,
        )
        device = src_batch.device
        src_encoded_map = self.module.tgt_cellmap_encoder(src_batch)
        prior = src_encoded_map
        cond_batch = self._build_cond_batch(batch, batch_size, seq_len, device)

        ts = torch.linspace(0.0, 1.0, self.conf["experiment"]["time_step"], device=device)
        x_t = prior
        pred_type = self.model_conf.get("pred_type", "v")
        for t0, t1 in tqdm(zip(ts[:-1], ts[1:]), desc=f"[Rank {self.global_rank}] Entering Evaluation"):
            cond_batch["t"] = t0.expand(batch_size)
            dt = (t1 - t0).expand(batch_size)
            pred = self.module(x_t, cond_batch)
            if pred_type == "x":
                denom = (1.0 - t0).clamp_min(float(self.model_conf.get("eval_x_denom_eps", 1e-4)))
                exp_vf = (pred - x_t) / denom
            elif pred_type == "v":
                exp_vf = pred
            else:
                raise ValueError(f"Unknown pred_type: {pred_type}. Expected 'x' or 'v'.")
            x_t = self.flow_utils.denoise(x_t, exp_vf, dt)

        pred_lib_size = self._prediction_lib_size(src_batch, tgt_batch)
        tgt_pred = self._decode_log1p_prediction(
            x_t,
            pred_lib_size,
            src_encoded_map,
            src_batch,
            tgt_batch,
        )

        if test_mode:
            return tgt_pred

        delta_pearson, mae, mse, gt_delta_pearson = self.compute_metrics(src_batch, tgt_batch, tgt_pred)
        log_info = {"delta_pearson": delta_pearson, "mse": mse, "mae": mae}
        return {"loss": torch.zeros((), device=device), "log_info": log_info}

    def compute_metrics(self, src_batch, tgt_batch, tgt_pred):
        delta_pearson = self.metric_utils.delta_pearson(src_batch, tgt_pred, tgt_batch)
        mae = self.metric_utils.mae(tgt_batch, tgt_pred)
        mse = self.metric_utils.mse(tgt_batch, tgt_pred)
        gt_delta_pearson = self.metric_utils.delta_pearson(src_batch, tgt_batch, tgt_batch)
        return delta_pearson, mae, mse, gt_delta_pearson
    
    
