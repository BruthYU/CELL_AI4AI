import dataclasses
from collections.abc import Callable, Sequence
from dataclasses import field as dc_field
from typing import Any, Literal
from megatron.core.transformer.enums import ModelType
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor

from models.ENDPOINT.networks._condition_encoders import ConditionEncoder

from models.ENDPOINT.networks._basic import DiTBlock

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
        self.cell_dim = int(conf["cell_dim"])
        self.dit_hidden_dim = int(conf.get("dit_hidden_dim", self.cell_dim))
        self.dit_depth = int(conf["dit_depth"])
        self.cell_sentence_len = int(conf["cell_sentence_len"])


        self._setup()

    # ----------------------------------------------------------------------

    def _setup(self):
        if self.cell_dim == self.dit_hidden_dim:
            self.input_projection = nn.Identity()
            self.output_projection = nn.Identity()
        else:
            self.input_projection = nn.Linear(self.cell_dim, self.dit_hidden_dim)
            self.output_projection = nn.Linear(self.dit_hidden_dim, self.cell_dim)

        # ---------- Condition Encoder ----------
        self.condition_encoder = ConditionEncoder(self.conf)
        condition_dim = int(self.condition_encoder.output_dim)
        if condition_dim == self.dit_hidden_dim:
            self.condition_projection = nn.Identity()
        else:
            self.condition_projection = nn.Linear(condition_dim, self.dit_hidden_dim)

        # ---------- Raw-cell DiT endpoint field ----------
        self.dit_block_conf = dict(self.conf.get("dit_kwargs") or {})
        self.dit_block_conf["hidden_size"] = self.dit_hidden_dim
        self.dit_block_conf.setdefault("num_heads", 16)
        self.dit_block_conf.setdefault("mlp_ratio", 4.0)
        self.dit_blocks = nn.ModuleList([DiTBlock(self.dit_block_conf) for _ in range(self.dit_depth)])
        


    def set_input_tensor(self, input_tensor: Tensor):
        self.input_tensor = input_tensor

    def forward(self, x_t, cond_batch):
        aggregated_condition = self.condition_encoder(cond_batch)
        aggregated_condition = self.condition_projection(aggregated_condition)
        pred_endpoint = self.input_projection(x_t)
        for block in self.dit_blocks:
            pred_endpoint = block(pred_endpoint, aggregated_condition)
        
        pred_endpoint = self.output_projection(pred_endpoint)
        return torch.relu(pred_endpoint)







