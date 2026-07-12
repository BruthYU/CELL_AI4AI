from collections.abc import Sequence
from dataclasses import field as dc_field
from typing import Any, Literal
import logging
import copy

import torch
import torch.nn as nn
import torch.optim as optim


import models.JIT_FP.networks._basic as nn_basic

__all__ = ["ConditionEncoder"]

logger = logging.getLogger(__name__)



    

class ConditionEncoder(nn.Module):
    def __init__(self, conf):
        super().__init__()
        # --------- set configuration ----------
        self.conf = nn_basic.get_module_conf(self.__class__.__name__, conf)
        
        self.layers_before_pool = self.conf["layers_before_pool"]
        self.layers_after_pool = self.conf["layers_after_pool"]
        self.onthot_lastdim = 0
        for key, layer_info in self.layers_before_pool.items():
            self.onthot_lastdim += layer_info["dims"][-1]
        self.layers_after_pool["input_dim"] = self.onthot_lastdim


        self.layers_for_aggregation = self.conf["layers_for_aggregation"]
        self.cell_dim = self.conf["cell_dim"]
        self.cellmap_width = self.conf["cellmap_width"]
        self.cell_sentence_len = self.conf["cell_sentence_len"]
        self.time_freqs = self.conf["time_freqs"]
        self.time_max_period = self.conf["time_max_period"]
        self.time_encoder_dims = self.conf["time_encoder_dims"]
        self.time_encoder_dropout = self.conf["time_encoder_dropout"]



        
        # --------- time encoder ---------
        self.time_encoder_conf = {
            "input_dim": self.time_freqs,
            "dims": self.time_encoder_dims,
            "dropout_rate": self.time_encoder_dropout,
            "act_last_layer": False,
        }
        self.time_encoder = nn_basic.MLPBlock(self.time_encoder_conf)

        # --------- modules before pooling ---------
        self.before_pool_modules = nn.ModuleDict({key: nn.ModuleList(nn_basic.get_customized_layer(layer_info))
            for key, layer_info in self.layers_before_pool.items()})

        # pooling module
        self.pooling_type = self.conf.get("pooling", "attention_token")
        self.pooling_kwargs = self.conf.get("pooling_kwargs", {})
        
        if self.pooling_type == "attention_token":
            self.pool_module = nn_basic.TokenAttentionPooling({'qkv_dim': self.onthot_lastdim, **self.pooling_kwargs})
        elif self.pooling_type == "attention_seed":
            # 这里的 multi-head attn 聚合通常指 SeedAttention
            seed_conf = {
                'v_dim': self.onthot_lastdim, 
                'seed_dim': self.onthot_lastdim,
                'num_heads': 8,
                **self.pooling_kwargs
            }
            self.pool_module = nn_basic.SeedAttentionPooling(seed_conf)
        elif self.pooling_type == "attention_basic":
            # 最基础的 multi-head self-attention + mean pooling
            basic_attn_conf = {
                'embed_dim': self.onthot_lastdim,
                'num_heads': 8,
                'pool_method': 'mean',  # "mean" or "first"
                **self.pooling_kwargs
            }
            self.pool_module = nn_basic.BasicMultiHeadAttentionPooling(basic_attn_conf)
        elif self.pooling_type == "mean":
            self.pool_module = nn_basic.MeanPooling()
        else:
            raise ValueError(f"Unknown pooling type: {self.pooling_type}")

        # ------------- modules after pooling ------------
        self.after_pool_modules = nn.ModuleList(nn_basic.get_customized_layer(self.layers_after_pool))

        # ------------- modules for aggregation ------------
        self.aggregation_modules = nn.ModuleList(nn_basic.get_customized_layer(self.layers_for_aggregation))





        # --------- control cellmap fusion  ---------
        # self.cellmap_encoder_conf = {
        #     "add_pos_emd": True,
        #     "input_dim": self.cell_dim,
        #     "output_dim": self.cellmap_width,
        #     "input_seq_len": self.cell_sentence_len,
        #     "output_seq_len": self.cellmap_width,
        # }
        # self.cellmap_encoder = nn_basic.CellMapEncoder(self.cellmap_encoder_conf)
        # cellmap_pool_conf = {"qkv_dim": self.cellmap_width}
        # self.src_pool_module = nn_basic.TokenAttentionPooling(cellmap_pool_conf)

        

    # ============================================================

    def forward(self, conditions):
        # ---------- Encode Time ----------
        t = conditions['t']
        t_encoded = nn_basic.sinusoidal_time_encoder(t, time_freqs=self.time_freqs, time_max_period=self.time_max_period)
        t_encoded = self.time_encoder(t_encoded) # [B, 512]
        
        # ---------- Encode Cellmap ----------
        # src_batch = conditions["src_batch"]
        # src_encoded_map = self.cellmap_encoder(src_batch) # [B, 512, 512]
        # src_pooled_map = self.src_pool_module(src_encoded_map) # [B, 512]

        # ---------- Encode One-Hot Condition----------
        processed_onehot = []
        for pert_cov in self.before_pool_modules.keys():
            x = conditions[pert_cov]
            x = nn_basic.apply_modules(self.before_pool_modules[pert_cov], x) # [B, N, 256]
            
            processed_onehot.append(x)
        onehot_pooling_arr = torch.cat(processed_onehot, dim=-1) # [B, N, m * 256]
        onehot_embedding = self.pool_module(onehot_pooling_arr) # [B, 1, m * 256] or [B, m * 256]
        
        # 确保 pooling 输出是 2D [B, D]（某些 pooling 模块对 3D 输入可能返回 [B, 1, D]）
        if onehot_embedding.ndim == 3:
            onehot_embedding = onehot_embedding.squeeze(1)
        
        onehot_embedding = nn_basic.apply_modules(self.after_pool_modules, onehot_embedding)  # [B, 512]


        # ---------- Aggregate ----------
        # aggregated_condition= torch.cat([src_pooled_map, onehot_embedding, t_encoded], dim=-1) # [B, 3 * 512]
        aggregated_condition= torch.cat([onehot_embedding, t_encoded], dim=-1) # [B, 2 * 512]
        aggregated_condition = nn_basic.apply_modules(self.aggregation_modules, aggregated_condition) # [B, 512]

        return aggregated_condition # [B, 512]