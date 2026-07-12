import dataclasses
from collections.abc import Callable, Sequence
from dataclasses import field as dc_field
from typing import Any, Literal
from megatron.core.transformer.enums import ModelType
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor

from models.JIT_LLM_REPLOGLE_V3_STATEALIGN.networks._condition_encoders import ConditionEncoder

from models.JIT_LLM_REPLOGLE_V3_STATEALIGN.networks._basic import DiTBlock
from models.JIT_LLM_REPLOGLE_V3_STATEALIGN.networks._llm_plus import CellMapEncoder, CellMapClassifier, CellMapDecoder

from omegaconf import OmegaConf
from megatron.core.models.common.language_module.language_module import LanguageModule
from megatron.core.transformer.transformer_config import TransformerConfig

from nemo.core.classes import ModelPT


__all__ = ["ConditionalVelocityField"]


    

class ConditionalVelocityField(LanguageModule):

    def __init__(self, conf, base_config: TransformerConfig):
        super().__init__(base_config)

        self.config = base_config
        self.model_type = ModelType.encoder_and_decoder
        self.pre_process = True
        self.post_process = True
        self.share_embeddings_and_output_weights = True
        self.conf = conf
        # 全部 dict-access
        self.cell_dim = conf["cell_dim"]
        self.cellmap_width = conf["cellmap_width"]
        self.dit_depth = conf["dit_depth"]
        self.cell_sentence_len = conf["cell_sentence_len"]


        self._setup()

    # ----------------------------------------------------------------------

    def _setup(self):
        
        # ---------- CellMap Encoder ----------
        self.cellmap_encoder = CellMapEncoder(self.conf)
        self.cellmap_classifier = CellMapClassifier(self.conf)
        self.cellmap_decoder = CellMapDecoder(self.conf)
        # ---------- Condition Encoder ----------
        self.condition_encoder = ConditionEncoder(self.conf)
        self.use_support_delta_adapter = bool(self.conf.get("use_support_delta_adapter", False))
        if self.use_support_delta_adapter:
            hidden_dim = int(self.conf.get("support_delta_hidden_dim", 1024))
            self.support_other_encoder = nn.Sequential(
                nn.Linear(self.cell_dim, self.cellmap_width),
                nn.LayerNorm(self.cellmap_width),
                nn.SiLU(),
            )
            self.support_same_encoder = nn.Sequential(
                nn.Linear(self.cell_dim, self.cellmap_width),
                nn.LayerNorm(self.cellmap_width),
                nn.SiLU(),
            )
            self.support_fusion = nn.Sequential(
                nn.Linear(self.cellmap_width * 3 + 3, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, self.cellmap_width),
            )
            self.support_delta_projector = nn.Sequential(
                nn.Linear(self.cell_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, self.cell_dim),
            )
            self.support_gate = nn.Sequential(
                nn.Linear(self.cellmap_width + 3, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1),
            )
            if bool(self.conf.get("support_delta_projector_zero_init", True)):
                final = self.support_delta_projector[-1]
                nn.init.zeros_(final.weight)
                nn.init.zeros_(final.bias)
            gate_bias = self.conf.get("support_gate_init_bias")
            if gate_bias is not None:
                final_gate = self.support_gate[-1]
                nn.init.zeros_(final_gate.weight)
                nn.init.constant_(final_gate.bias, float(gate_bias))
        else:
            self.support_other_encoder = None
            self.support_same_encoder = None
            self.support_fusion = None
            self.support_delta_projector = None
            self.support_gate = None

        self.use_gene_delta_head = bool(self.conf.get("use_gene_delta_head", False))
        if self.use_gene_delta_head:
            head_dims = list(self.conf.get("gene_delta_head_dims", [1024]))
            layers = []
            in_dim = self.cellmap_width
            for hidden_dim in head_dims:
                layers.extend([nn.Linear(in_dim, hidden_dim), nn.SiLU()])
                in_dim = hidden_dim
            layers.append(nn.Linear(in_dim, self.cell_dim))
            self.gene_delta_head = nn.Sequential(*layers)
            if bool(self.conf.get("gene_delta_head_zero_init", True)):
                final = self.gene_delta_head[-1]
                nn.init.zeros_(final.weight)
                nn.init.zeros_(final.bias)
        else:
            self.gene_delta_head = None

        # ---------- DiT Velocity Field ----------
        # num_heads/mlp_ratio 可通过 dit_kwargs 配置（为了与 PRIOR 对齐时可设 num_heads=16）
        dit_kwargs = self.conf.get("dit_kwargs", {})
        self.dit_block_conf = {
            "hidden_size": self.cellmap_width,
            "num_heads": dit_kwargs.get("num_heads", 8),
            "mlp_ratio": dit_kwargs.get("mlp_ratio", 4.0),
        }
        self.dit_blocks = nn.ModuleList([DiTBlock(self.dit_block_conf) for _ in range(self.dit_depth)])


    def set_input_tensor(self, input_tensor: Tensor):
        self.input_tensor = input_tensor

    def forward(self, x_t, cond_batch):
        # xt: [B, 512, 512]
        aggregated_condition = self.condition_embedding(cond_batch) # [B, 512]
        pred_vf = x_t
        for block in self.dit_blocks:
            pred_vf = block(pred_vf, aggregated_condition)
        return pred_vf

    def condition_embedding(self, cond_batch):
        aggregated_condition = self.condition_encoder(cond_batch)
        if not self.use_support_delta_adapter:
            return aggregated_condition
        support = self.support_delta_components(
            cond_batch,
            aggregated_condition=aggregated_condition,
            apply_dropout=self.training,
        )
        support_context = torch.cat(
            [
                aggregated_condition,
                self.support_other_encoder(support["other_delta"]),
                self.support_same_encoder(support["same_delta"]),
                support["other_mask"],
                support["same_mask"],
                support["count_norm"],
            ],
            dim=-1,
        )
        support_update = self.support_fusion(support_context)
        scale = float(self.conf.get("support_condition_scale", 1.0))
        return aggregated_condition + scale * support_update

    def support_delta_components(self, cond_batch, aggregated_condition=None, apply_dropout=False):
        if not self.use_support_delta_adapter:
            return None
        if "support_delta_other_cells" not in cond_batch:
            if aggregated_condition is None:
                aggregated_condition = self.condition_encoder(cond_batch)
            batch_size = aggregated_condition.shape[0]
            device = aggregated_condition.device
            dtype = aggregated_condition.dtype
            zero_delta = torch.zeros(batch_size, self.cell_dim, device=device, dtype=dtype)
            zero_mask = torch.zeros(batch_size, 1, device=device, dtype=dtype)
            return {
                "other_delta": zero_delta,
                "same_delta": zero_delta,
                "other_mask": zero_mask,
                "same_mask": zero_mask,
                "source_delta": zero_delta,
                "source_mask": zero_mask,
                "count_norm": zero_mask,
                "adapted_delta": zero_delta,
                "gate": zero_mask,
            }

        other_delta = cond_batch["support_delta_other_cells"].float()
        same_delta = cond_batch["support_delta_same_cell"].float()
        other_mask = cond_batch["support_other_mask"].float()
        same_mask = cond_batch["support_same_mask"].float()
        count = cond_batch["support_count"].float()
        if other_mask.dim() == 1:
            other_mask = other_mask.unsqueeze(-1)
        if same_mask.dim() == 1:
            same_mask = same_mask.unsqueeze(-1)
        if count.dim() == 1:
            count = count.unsqueeze(-1)

        if apply_dropout:
            other_dropout = float(self.conf.get("support_other_train_dropout", 0.0))
            same_dropout = float(self.conf.get("support_same_train_dropout", 1.0))
            if other_dropout > 0.0:
                keep = (torch.rand_like(other_mask) >= other_dropout).float()
                other_mask = other_mask * keep
            if same_dropout > 0.0:
                keep = (torch.rand_like(same_mask) >= same_dropout).float()
                same_mask = same_mask * keep

        if not bool(self.conf.get("support_same_condition_enabled", True)):
            same_mask = torch.zeros_like(same_mask)

        source_mode = str(self.conf.get("support_delta_source", "other_fallback_same")).lower()
        if source_mode == "other":
            source_delta = other_delta * other_mask
            source_mask = other_mask
        elif source_mode == "same":
            source_delta = same_delta * same_mask
            source_mask = same_mask
        elif source_mode == "blend":
            denom = torch.clamp(other_mask + same_mask, min=1.0)
            source_delta = (other_delta * other_mask + same_delta * same_mask) / denom
            source_mask = torch.clamp(other_mask + same_mask, max=1.0)
        else:
            use_same = (1.0 - other_mask) * same_mask
            source_delta = other_delta * other_mask + same_delta * use_same
            source_mask = torch.clamp(other_mask + use_same, max=1.0)

        count_norm = torch.log1p(torch.clamp(count, min=0.0)) / float(self.conf.get("support_count_norm", 1000.0))
        count_norm = torch.clamp(count_norm, max=1.0)
        if aggregated_condition is None:
            aggregated_condition = self.condition_encoder(cond_batch)
        gate_in = torch.cat([aggregated_condition, other_mask, same_mask, count_norm], dim=-1)
        gate = torch.sigmoid(self.support_gate(gate_in)) * source_mask
        projected = self.support_delta_projector(source_delta)
        adapted_delta = gate * (source_delta + projected)
        return {
            "other_delta": other_delta * other_mask,
            "same_delta": same_delta * same_mask,
            "other_mask": other_mask,
            "same_mask": same_mask,
            "source_delta": source_delta,
            "source_mask": source_mask,
            "count_norm": count_norm,
            "adapted_delta": adapted_delta,
            "gate": gate,
        }

    def predict_gene_delta(self, cond_batch):
        cond_for_delta = dict(cond_batch)
        if "t" in cond_for_delta:
            head_t = float(self.conf.get("gene_delta_head_t", 1.0))
            cond_for_delta["t"] = torch.full_like(cond_for_delta["t"], head_t)
        aggregated_condition = self.condition_embedding(cond_for_delta)
        gene_delta = None
        if self.gene_delta_head is not None:
            gene_delta = self.gene_delta_head(aggregated_condition)
        if self.use_support_delta_adapter:
            support = self.support_delta_components(
                cond_for_delta,
                aggregated_condition=aggregated_condition,
                apply_dropout=self.training,
            )
            support_scale = float(self.conf.get("support_delta_scale", 1.0))
            support_delta = support_scale * support["adapted_delta"]
            gene_delta = support_delta if gene_delta is None else gene_delta + support_delta
        return gene_delta






