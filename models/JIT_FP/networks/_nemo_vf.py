import dataclasses
from collections.abc import Callable, Sequence
from dataclasses import field as dc_field
from typing import Any, Literal
from megatron.core.transformer.enums import ModelType
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor

from models.JIT_FP.networks._condition_encoders import ConditionEncoder
from models.JIT_FP.networks._basic import DiTBlock, CellMapEncoder, CellMapDecoder
from models.JIT_FP.networks._llm_plus import CellMapClassifier

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
        self.cell_dim = conf["cell_dim"]
        self.cellmap_width = conf["cellmap_width"]
        self.dit_depth = conf["dit_depth"]
        self.cell_sentence_len = conf["cell_sentence_len"]

        self._setup()

    def _setup(self):
        enc_kwargs = self.conf.get("cellmap_encoder_kwargs", {})
        enc_conf = {
            "input_dim": self.cell_dim,
            "input_seq_len": self.cell_sentence_len,
            "output_dim": self.cellmap_width,
            "output_seq_len": self.cellmap_width,
            "dims": enc_kwargs.get("dims", [self.cell_dim, 1024, self.cellmap_width]),
            "add_pos_emd": enc_kwargs.get("add_pos_emd", False),
            "gene_encoder_type": enc_kwargs.get("gene_encoder_type", "mlp"),
        }
        for k in ["dropout_rate", "act_fn", "cellmap_pooling", "cellmap_pooling_kwargs"]:
            if k in enc_kwargs:
                enc_conf[k] = enc_kwargs[k]
        self.cellmap_encoder = CellMapEncoder(enc_conf)

        self.cellmap_classifier = CellMapClassifier(self.conf)

        dec_kwargs = self.conf.get("cellmap_decoder_kwargs", {})
        dec_conf = {
            "input_dim": self.cellmap_width,
            "output_dim": self.cell_dim,
            "output_seq_len": self.cell_sentence_len,
            "dims": dec_kwargs.get("dims", [self.cellmap_width, 1024, self.cell_dim]),
            "gene_decoder_type": dec_kwargs.get("gene_decoder_type", "mlp"),
        }
        for k in ["dropout_rate", "act_fn", "cellmap_pooling", "cellmap_pooling_kwargs"]:
            if k in dec_kwargs:
                dec_conf[k] = dec_kwargs[k]
        self.cellmap_decoder = CellMapDecoder(dec_conf)

        self.condition_encoder = ConditionEncoder(self.conf)

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
        aggregated_condition = self.condition_encoder(cond_batch)
        pred_vf = x_t
        for block in self.dit_blocks:
            pred_vf = block(pred_vf, aggregated_condition)
        return pred_vf
