from __future__ import annotations

import inspect
import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, Iterator, Optional, Tuple

from bionemo.llm.model.biobert.lightning import get_batch_on_this_context_parallel_rank
from bionemo.llm.model.loss import _Nemo2CompatibleLossReduceMixin
from bionemo.llm.api import MegatronLossType, MegatronModelType
from nemo.collections.nlp.modules.common.megatron.utils import average_losses_across_data_parallel_group
from nemo.lightning.megatron_parallel import MegatronLossReduction

from models.nemo_model_base import ModelInterfaceBase
import models.PRIOR.networks._basic as nn_basic

# Import all migrated models
from .networks import (
    StateTransitionPerturbationModel,
    ContextMeanPerturbationModel,
    EmbedSumPerturbationModel,
    PerturbMeanPerturbationModel,
    OldNeuralOTPerturbationModel,
    DecoderOnlyPerturbationModel,
    PseudobulkPerturbationModel,
)

MODEL_CLASSES = {
    "state_transition": StateTransitionPerturbationModel,
    "context_mean": ContextMeanPerturbationModel,
    "embed_sum": EmbedSumPerturbationModel,
    "perturb_mean": PerturbMeanPerturbationModel,
    "old_neural_ot": OldNeuralOTPerturbationModel,
    "decoder_only": DecoderOnlyPerturbationModel,
    "pseudobulk": PseudobulkPerturbationModel,
}


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
        unreduced_loss = forward_out["loss"]
        reduced = average_losses_across_data_parallel_group([unreduced_loss])
        return unreduced_loss, {"avg": reduced}


class _PertOneHotEncoder(nn.Module):
    """Encode `pert_onehot` into a dense `pert_emb` using the same config style as other nemo_cellflow models."""

    def __init__(self, pert_layer_conf: dict):
        super().__init__()
        # Reuse the shared helper that supports onehot -> argmax for embedding layers.
        modules = nn_basic.get_customized_layer(dict(pert_layer_conf))
        self.modules_ = nn.ModuleList(modules)
        # infer output dim
        last_dim = pert_layer_conf["dims"][-1]
        self.out_dim = int(last_dim)

    def forward(self, pert_onehot: torch.Tensor) -> torch.Tensor:
        x = pert_onehot
        for m in self.modules_:
            x = m(x)
        return x


class STATE_Nemo_Model(ModelInterfaceBase[MegatronModelType, MegatronLossType]):
    """
    Nemo adapter for state models (default: StateTransitionPerturbationModel).
    Supports all models migrated to `nemo_cellflow/models/STATE/networks/`.

    Expects the `cell_group` style batch:
      - ctrl_cell_emb: [B, S, D]
      - pert_cell_emb: [B, S, D]
      - pert_onehot:   [B, vocab] (one-hot)
    Produces:
      - pert_emb:      [B, S, pert_dim] to match st model input contract
    """

    def __init__(self, conf, model_transform=None, configure_init_model_parallel: bool = False):
        super().__init__(model_transform=model_transform, configure_init_model_parallel=configure_init_model_parallel)
        self.conf = conf
        self.model_conf = conf.get("prepare_model", {})

        # sentence length (shared top-level field in accelerate.yaml)
        self.cell_sentence_len = int(conf.get("cell_sentence_len") or conf.get("dataset", {}).get("cell_sentence_len") or 256)

        # Whether to use raw one-hot as pert_emb (matches original state training pipeline).
        self.use_raw_pert_onehot = bool(self.model_conf.get("use_raw_pert_onehot", True))

        # Optional perturbation encoder from config (kept for compatibility, but not used when
        # use_raw_pert_onehot=True).
        layers_before_pool = conf.get("layers_before_pool", {})
        if self.use_raw_pert_onehot:
            # No encoder needed; use raw one-hot directly.
            self.pert_encoder = None
            self.pert_dim = int(self.model_conf.get("pert_dim", 0)) or 0
            if self.pert_dim <= 0 and "pert_onehot" in layers_before_pool:
                # Best-effort fallback when dims are available
                self.pert_dim = int(layers_before_pool["pert_onehot"]["dims"][-1])
        else:
            if "pert_onehot" not in layers_before_pool:
                raise KeyError("STATE model expects config `layers_before_pool.pert_onehot` to encode perturbations.")
            self.pert_encoder = _PertOneHotEncoder(layers_before_pool["pert_onehot"])
            self.pert_dim = int(self.pert_encoder.out_dim)

        # Common dimensions
        self.cell_dim = int(self.model_conf.get("cell_dim", 2000))
        self.hidden_dim = int(self.model_conf.get("hidden_dim", 256))

        # Determine model class
        self.model_type = self.model_conf.get("model_type", "state_transition")
        if self.model_type not in MODEL_CLASSES:
             raise ValueError(f"Unknown model_type: {self.model_type}. Available: {list(MODEL_CLASSES.keys())}")
        self.model_class = MODEL_CLASSES[self.model_type]

        self.configure_model()

    # Megatron required
    def loss_reduction(self, *args, **kwargs):
        return FM_LossWithReduction(**kwargs)

    def data_step(self, dataloader_iter: Iterator) -> Dict:
        batch = next(dataloader_iter)
        if isinstance(batch, tuple) and len(batch) == 3:
            batch = batch[0]

        # IMPORTANT: keep batch tensors on the same device as this LightningModule.
        # Nemo/Lightning may move the model later; using `.cuda()` here can create device mismatches
        # (e.g., embedding weights on CPU while inputs are on CUDA).
        device = next(self.parameters()).device

        def to_device(x):
            if isinstance(x, torch.Tensor):
                return x.to(device, non_blocking=True)
            return x

        batch = {k: to_device(v) for k, v in batch.items()}
        batch = get_batch_on_this_context_parallel_rank(batch)

        # Build pert_emb from pert_onehot (raw one-hot by default), then expand to per-cell tokens
        if "pert_onehot" not in batch:
            raise KeyError("STATE model requires `pert_onehot` in batch.")

        if self.use_raw_pert_onehot:
            pert_vec = batch["pert_onehot"].float()
        else:
            pert_vec = self.pert_encoder(batch["pert_onehot"])  # [B, pert_dim] or [B,1,pert_dim] depending on config
        if pert_vec.dim() == 3 and pert_vec.size(1) == 1:
            pert_vec = pert_vec.squeeze(1)
        if pert_vec.dim() != 2:
            pert_vec = pert_vec.reshape(pert_vec.size(0), -1)

        # Determine seq_len from ctrl_cell_emb (preferred) else config
        if "ctrl_cell_emb" in batch and batch["ctrl_cell_emb"].dim() >= 2:
            seq_len = batch["ctrl_cell_emb"].shape[1]
        else:
            seq_len = self.cell_sentence_len

        batch["pert_emb"] = pert_vec.unsqueeze(1).expand(-1, seq_len, -1).contiguous()
        return batch

    def configure_model(self) -> None:
        # Prepare kwargs for model init
        kwargs = {
            "input_dim": self.cell_dim,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.cell_dim,
            "pert_dim": self.pert_dim,
            # Common defaults
            "output_space": "gene",
            "gene_decoder_bool": False,
            "embed_key": None,
        }

        # Add StateTransition specific args if needed, or just dump model_conf
        if self.model_type == "state_transition":
             kwargs.update({
                "transformer_backbone_key": str(self.model_conf.get("transformer_backbone_key", "GPT2")),
                "cell_set_len": self.cell_sentence_len,
             })
             # backbone kwargs
             backbone_kwargs = dict(self.model_conf.get("transformer_backbone_kwargs") or {})
             backbone_kwargs.setdefault("n_embd", self.hidden_dim)
             backbone_kwargs.setdefault("n_layer", int(self.model_conf.get("n_layer", 4)))
             backbone_kwargs.setdefault("n_head", int(self.model_conf.get("n_head", 8)))
             kwargs["transformer_backbone_kwargs"] = backbone_kwargs

        # Allow config to override anything
        kwargs.update(self.model_conf)

        # Remove keys that might confuse models if they strictly check kwargs (unlikely but safe)
        # For now, we trust **kwargs in models handle extra keys gracefully.

        self.module = self.model_class(**kwargs)
    
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

    def forward_step(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        # Check signature to see if padded is supported
        sig = inspect.signature(self.module.forward)
        fwd_kwargs = {}
        if "padded" in sig.parameters:
            fwd_kwargs["padded"] = True

        pred = self.module.forward(batch, **fwd_kwargs)
        target = batch["pert_cell_emb"]

        # Reshape if necessary (handling sequence dimension)
        if pred.dim() == 3 and target.dim() == 3:
             # Both are [B, S, D], flatten to [B*S, D] for loss
             pred = pred.reshape(-1, self.cell_dim)
             target = target.reshape(-1, self.cell_dim)
        elif pred.dim() == 2 and target.dim() == 3:
             # Model produced [B, D] but target is [B, S, D].
             # This happens if a cell-level model (like ContextMean) is fed a sequence batch.
             # We should probably flatten target or broadcast pred?
             # For now, let's assume we flatten target to [B*S, D] and expand pred to [B*S, D]?
             # Or maybe the user set cell_sentence_len=1.
             target = target.reshape(-1, self.cell_dim)
             if pred.shape[0] * self.cell_sentence_len == target.shape[0]:
                 # Expand pred: [B, D] -> [B, S, D] -> [B*S, D]
                 pred = pred.unsqueeze(1).expand(-1, self.cell_sentence_len, -1).reshape(-1, self.cell_dim)

        loss = self.module.loss_fn(pred, target).nanmean()
        return {"loss": loss, "log_info": {"loss": loss.detach()}}

    def training_step(self, batch: Dict[str, Tensor], batch_idx: Optional[int] = None) -> Dict[str, Tensor]:
        out = self.forward_step(batch)
        self.log_dict({"train_loss": out["loss"]}, on_step=True, on_epoch=True, prog_bar=True)
        return out

    def validation_step(self, batch: Dict[str, Tensor], batch_idx: Optional[int] = None) -> Dict[str, Tensor]:
        out = self.forward_step(batch)
        self.log_dict({"val_loss": out["loss"]}, on_step=True, on_epoch=True, prog_bar=False)
        return out
