import dataclasses
from collections.abc import Callable, Sequence
from dataclasses import field as dc_field
from typing import Any, Literal
from megatron.core.transformer.enums import ModelType
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor

from models.JIT_LLM.networks._condition_encoders import ConditionEncoder

from models.JIT_LLM.networks._basic import DiTBlock
from models.JIT_LLM.networks._llm_plus import CellMapEncoder, CellMapClassifier, CellMapDecoder

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
        aggregated_condition = self.condition_encoder(cond_batch) # [B, 512]
        pred_vf = x_t
        for block in self.dit_blocks:
            pred_vf = block(pred_vf, aggregated_condition)
        return pred_vf












