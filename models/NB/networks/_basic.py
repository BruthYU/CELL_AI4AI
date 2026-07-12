import math
from collections.abc import Sequence
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
import copy
from timm.models.vision_transformer import Attention, Mlp
def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, conf):
        super().__init__()
        self.conf = get_module_conf(self.__class__.__name__, conf)
        self.hidden_size = self.conf["hidden_size"]
        self.num_heads = self.conf["num_heads"]
        self.mlp_ratio = self.conf["mlp_ratio"]
        
        self.norm1 = nn.LayerNorm(self.hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(self.hidden_size, num_heads=self.num_heads, qkv_bias=True)
        self.norm2 = nn.LayerNorm(self.hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(self.hidden_size * self.mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=self.hidden_size, 
                       hidden_features=mlp_hidden_dim, 
                       act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.hidden_size, 6 * self.hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x




class SetAttentionPooling(nn.Module):
    """
    Permutation-invariant set pooling:
    x: [B, N, D]  ->  pooled: [B, K, D]
    """
    # def __init__(self, d_model: int, K: int, hidden: int = 256):
    def __init__(self, conf):
        super().__init__()
        self.conf = get_module_conf(self.__class__.__name__, conf)
        self.K = self.conf["K"]
        self.d_model = self.conf["d_model"]
        self.hidden = self.conf["hidden"]

        self.phi = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.hidden),
            nn.GELU(),
            nn.Linear(self.hidden, self.d_model),
        )
        # 每个 head 产生一个标量 score：K 个 head => scores [B, N, K]
        self.score = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.hidden),
            nn.GELU(),
            nn.Linear(self.hidden, self.K),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, D]
        v = self.phi(x)                 # [B, N, D]
        s = self.score(x)               # [B, N, K]
        a = F.softmax(s, dim=1)         # softmax over N (elements), [B, N, K]

        # 对 N 做加权求和：输出 [B, K, D]
        pooled = torch.einsum("bnk,bnd->bkd", a, v)
        return pooled

class BasicAttentionPooling(nn.Module):
    def __init__(self, conf):
        super().__init__()
        # NOTE:
        # - This module is used as a set-pooling layer in CellMapEncoder/Decoder.
        # - The previous implementation was incomplete and also attempted to read
        #   default_params.yaml via get_module_conf("BasicAttentionPooling", ...),
        #   but that key does not exist, causing ConfigKeyError.
        # Here we implement a standard attention pooling:
        # learnable queries (K, D) attend to tokens (N, D) => pooled (K, D).
        self.conf = conf or {}
        self.d_model = int(self.conf.get("d_model", self.conf.get("embed_dim", 512)))
        self.K = int(self.conf.get("K", 1))
        dropout_rate = float(self.conf.get("dropout_rate", 0.0))
        num_heads = int(self.conf.get("num_heads", 8))
        if num_heads <= 0:
            num_heads = 1
        if self.d_model % num_heads != 0:
            # fall back to a safe head count
            num_heads = 1

        # Learnable queries (seed tokens)
        self.query = nn.Parameter(torch.randn(1, self.K, self.d_model) / math.sqrt(self.d_model))
        self.attn = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(self.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, D]
        b, _, d = x.shape
        q = self.query.expand(b, -1, -1)  # [B, K, D]
        pooled, _ = self.attn(q, x, x, need_weights=False)  # [B, K, D]
        pooled = self.norm(pooled)
        return pooled


def _safe_num_heads(embed_dim: int, requested_heads: int) -> int:
    requested_heads = max(1, int(requested_heads))
    if embed_dim % requested_heads == 0:
        return requested_heads
    for heads in range(requested_heads, 0, -1):
        if embed_dim % heads == 0:
            return heads
    return 1


def _activation_from_name(name: str) -> nn.Module:
    if not hasattr(nn, name):
        raise ValueError(f"Activation {name} not found in torch.nn")
    return getattr(nn, name)()


class TabularAttentionLayer(nn.Module):
    """
    Stack-style cell-by-gene tabular attention.
    x: [B, S, H, T] -> [B, S, H, T]
    """

    def __init__(
        self,
        n_hidden: int,
        token_dim: int,
        num_heads: int,
        mlp_ratio: float,
        dropout_rate: float,
        act_fn: str,
    ):
        super().__init__()
        self.n_hidden = int(n_hidden)
        self.token_dim = int(token_dim)
        self.cell_dim = self.n_hidden * self.token_dim
        token_heads = _safe_num_heads(self.token_dim, num_heads)
        cell_heads = _safe_num_heads(self.cell_dim, num_heads)

        self.token_attn = nn.MultiheadAttention(
            embed_dim=self.token_dim,
            num_heads=token_heads,
            dropout=dropout_rate,
            batch_first=True,
        )
        self.token_norm = nn.LayerNorm(self.token_dim)
        self.cell_attn = nn.MultiheadAttention(
            embed_dim=self.cell_dim,
            num_heads=cell_heads,
            dropout=dropout_rate,
            batch_first=True,
        )
        self.cell_norm = nn.LayerNorm(self.cell_dim)

        mlp_hidden = max(self.token_dim, int(self.token_dim * mlp_ratio))
        self.mlp = nn.Sequential(
            nn.Linear(self.token_dim, mlp_hidden),
            _activation_from_name(act_fn),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_hidden, self.token_dim),
            nn.Dropout(dropout_rate),
        )
        self.mlp_norm = nn.LayerNorm(self.token_dim)

    def forward(self, x: torch.Tensor, gene_pos_embedding: torch.Tensor) -> torch.Tensor:
        b, s, h, t = x.shape
        if h != self.n_hidden or t != self.token_dim:
            raise ValueError(
                f"Expected [B,S,{self.n_hidden},{self.token_dim}], got {tuple(x.shape)}."
            )

        token_x = x.reshape(b * s, h, t)
        token_in = token_x + gene_pos_embedding.unsqueeze(0)
        token_attn, _ = self.token_attn(token_in, token_in, token_in, need_weights=False)
        token_x = self.token_norm(token_x + token_attn)
        x = token_x.reshape(b, s, h, t)

        cell_x = x.reshape(b, s, h * t)
        cell_attn, _ = self.cell_attn(cell_x, cell_x, cell_x, need_weights=False)
        cell_x = self.cell_norm(cell_x + cell_attn)
        x = cell_x.reshape(b, s, h, t)

        mlp_x = x.reshape(b * s * h, t)
        mlp_out = self.mlp(mlp_x)
        x = self.mlp_norm(mlp_x + mlp_out).reshape(b, s, h, t)
        return x


class CellMapEncoder(nn.Module):
    def __init__(self, conf: dict):
        super().__init__()
        self.conf = get_module_conf(self.__class__.__name__, conf)
        self.input_dim = int(self.conf["input_dim"])
        self.input_seq_len = int(self.conf["input_seq_len"])
        self.dims = self.conf["dims"]
        self.output_dim = int(self.conf["output_dim"])
        self.output_seq_len = int(self.conf["output_seq_len"])
        self.add_pos_emd = bool(self.conf["add_pos_emd"])
        self.gene_encoder_type = self.conf.get("gene_encoder_type", "mlp")
        if self.add_pos_emd:
            self.pos_emd = nn.Embedding(self.input_seq_len, self.input_dim)

        if self.gene_encoder_type == "tabular_attention":
            self.n_hidden = int(self.conf.get("n_hidden", 64))
            self.token_dim = int(self.conf.get("token_dim", self.output_dim // self.n_hidden))
            if self.n_hidden * self.token_dim != self.output_dim:
                raise ValueError(
                    "CellMapEncoder requires n_hidden * token_dim == output_dim for "
                    f"tabular_attention, got {self.n_hidden} * {self.token_dim} != {self.output_dim}."
                )
            self.gene_reduction = nn.Sequential(
                nn.Linear(self.input_dim, self.output_dim),
                _activation_from_name(self.conf.get("act_fn", "SiLU")),
                nn.Dropout(float(self.conf.get("dropout_rate", 0.0))),
            )
            self.gene_pos_embedding = nn.Parameter(
                torch.randn(self.n_hidden, self.token_dim) * 0.02
            )
            self.tabular_layers = nn.ModuleList(
                [
                    TabularAttentionLayer(
                        n_hidden=self.n_hidden,
                        token_dim=self.token_dim,
                        num_heads=int(self.conf.get("tabular_n_heads", self.conf.get("num_heads", 8))),
                        mlp_ratio=float(self.conf.get("tabular_mlp_ratio", self.conf.get("mlp_ratio", 2.0))),
                        dropout_rate=float(self.conf.get("dropout_rate", 0.0)),
                        act_fn=self.conf.get("act_fn", "SiLU"),
                    )
                    for _ in range(int(self.conf.get("n_tabular_layers", 2)))
                ]
            )

        elif self.gene_encoder_type == "transformer":
            gene_transformer_conf = {
                "input_dim": self.input_dim,
                "d_model": self.dims[-1],
            }
            # Optional passthrough of config
            for k in ["num_heads", "num_layers", "dim_feedforward", "dropout_rate", "act_fn", "norm_first"]:
                if k in self.conf:
                    gene_transformer_conf[k] = self.conf[k]

            self.gene_project = TransformerEncoderBlock(gene_transformer_conf)

        elif self.gene_encoder_type == "mlp":
            mlp_conf = {
                "input_dim": self.input_dim,
                "dims": self.dims,
                "act_last_layer": True,
            }
            for k in ["dropout_rate", "act_fn"]:
                if k in self.conf:
                    mlp_conf[k] = self.conf[k]
            
            self.gene_project = MLPBlock(mlp_conf)
        
        else:
            raise ValueError(f"Unknown gene_encoder_type: {self.gene_encoder_type}")

        if self.gene_encoder_type != "tabular_attention":
            set_attention_pooling_conf = {
                "d_model": self.dims[-1],
                "K": self.output_seq_len,
                "hidden": self.dims[-1],
            }
            pooling_type = (
                self.conf.get("cellmap_pooling")
                or self.conf.get("cellmap_attn")
                or self.conf.get("set_pooling_type")
                or "basic"
            ).lower()
            pooling_kwargs = (
                self.conf.get("cellmap_pooling_kwargs")
                or self.conf.get("set_pooling_kwargs")
                or {}
            )
            pooling_cls = (
                SetAttentionPooling
                if pooling_type in {"set", "set_attn", "attention_set"}
                else BasicAttentionPooling
            )
            self.set_attention_pooling = pooling_cls(
                {**set_attention_pooling_conf, **pooling_kwargs}
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, S, G] -> [B, S, output_dim] for tabular_attention.
        if self.add_pos_emd:
            pos_ids = torch.arange(x.shape[1], device=x.device)
            pos_emd = self.pos_emd(pos_ids)
            x = x + pos_emd
        if self.gene_encoder_type == "tabular_attention":
            b, s, _ = x.shape
            x = self.gene_reduction(x).reshape(b, s, self.n_hidden, self.token_dim)
            for layer in self.tabular_layers:
                x = layer(x, self.gene_pos_embedding)
            return x.reshape(b, s, self.output_dim)

        x = self.gene_project(x)
        x = self.set_attention_pooling(x)
        return x

class CellMapDecoder(nn.Module):
    def __init__(self, conf: dict):
        super().__init__()
        self.conf = get_module_conf(self.__class__.__name__, conf)
        self.input_dim = int(self.conf["input_dim"])
        self.dims = self.conf["dims"]
        self.output_dim = int(self.conf["output_dim"])
        self.output_seq_len = int(self.conf["output_seq_len"])
        self.decoder_distribution = self.conf.get("decoder_distribution", "deterministic")
        self.nb_min_dispersion = float(self.conf.get("nb_min_dispersion", 1e-4))

        if self.decoder_distribution == "nb":
            dims = list(self.conf.get("dims", [self.input_dim, self.input_dim * 2, self.output_dim * 2]))
            if dims[0] != self.input_dim:
                dims = [self.input_dim] + dims
            if dims[-1] != self.output_dim * 2:
                dims = dims[:-1] + [self.output_dim * 2]
            layers = []
            in_dim = dims[0]
            dropout_rate = float(self.conf.get("dropout_rate", 0.0))
            act_fn = self.conf.get("act_fn", "SiLU")
            for out_dim in dims[1:-1]:
                layers.append(nn.Linear(in_dim, out_dim))
                layers.append(_activation_from_name(act_fn))
                layers.append(nn.Dropout(dropout_rate))
                in_dim = out_dim
            layers.append(nn.Linear(in_dim, dims[-1]))
            self.nb_head = nn.Sequential(*layers)
            return

        gene_decoder_type = self.conf.get("gene_decoder_type", "mlp")

        if gene_decoder_type == "transformer":
            gene_transformer_conf = {
                "input_dim": self.input_dim,
                "d_model": self.dims[-1],
            }
            # Optional passthrough of config
            for k in ["num_heads", "num_layers", "dim_feedforward", "dropout_rate", "act_fn", "norm_first"]:
                if k in self.conf:
                    gene_transformer_conf[k] = self.conf[k]
            self.gene_project = TransformerEncoderBlock(gene_transformer_conf)
        elif gene_decoder_type == "mlp":
            mlp_conf = {
                "input_dim": self.input_dim,
                "dims": self.dims,
                "act_last_layer": True,
            }
            for k in ["dropout_rate", "act_fn"]:
                if k in self.conf:
                    mlp_conf[k] = self.conf[k]
            self.gene_project = MLPBlock(mlp_conf)
        else:
            raise ValueError(f"Unknown gene_decoder_type: {gene_decoder_type}")

        set_attention_pooling_conf = {
            "d_model": self.input_dim,
            "K": self.output_seq_len,
            "hidden": self.input_dim,
        }
        pooling_type = (
            self.conf.get("cellmap_pooling")
            or self.conf.get("cellmap_attn")
            or self.conf.get("set_pooling_type")
            or "basic"
        ).lower()
        pooling_kwargs = (
            self.conf.get("cellmap_pooling_kwargs")
            or self.conf.get("set_pooling_kwargs")
            or {}
        )
        pooling_cls = (
            SetAttentionPooling
            if pooling_type in {"set", "set_attn", "attention_set"}
            else BasicAttentionPooling
        )
        self.set_attention_pooling = pooling_cls(
            {**set_attention_pooling_conf, **pooling_kwargs}
        )
        self.output_relu = nn.ReLU()
    
    def forward(self, x: torch.Tensor, lib_size: torch.Tensor | None = None):
        if self.decoder_distribution == "nb":
            raw = self.nb_head(x).reshape(*x.shape[:-1], self.output_dim, 2)
            logits = raw[..., 0]
            theta = F.softplus(raw[..., 1]) + self.nb_min_dispersion
            px_scale = F.softmax(logits, dim=-1)
            if lib_size is None:
                raise ValueError("NB decoder requires lib_size with shape [B,S,1].")
            if lib_size.dim() == 2:
                lib_size = lib_size.unsqueeze(-1)
            lib_size = lib_size.to(dtype=px_scale.dtype, device=px_scale.device).clamp_min(1.0)
            mean = px_scale * lib_size
            nb_params = torch.stack([mean, theta], dim=-1)
            return {
                "nb_params": nb_params,
                "nb_mean": mean,
                "nb_dispersion": theta,
                "px_scale": px_scale,
                "px_scale_logits": logits,
            }

        # [B, N, D] -> [B, output_seq_len, dims[-1]]
        x = self.set_attention_pooling(x)
        x = self.gene_project(x)
        x = self.output_relu(x)
        return x

def get_module_conf(module_name, conf: dict | None):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 默认配置（DictConfig）
    default_module_conf = OmegaConf.load(
        os.path.join(current_dir, "default_params.yaml")
    )[module_name]
    
    # 用户传入为空
    if conf is None:
        return default_module_conf

    # 将 dict 转换成 DictConfig
    conf_cfg = OmegaConf.create(conf)

    # 使用 OmegaConf.merge 深度合并
    merged = OmegaConf.merge(default_module_conf, conf_cfg)

    return OmegaConf.to_container(merged, resolve=False)


def sinusoidal_time_encoder(t, time_freqs, time_max_period = 10000):
    """
    t : torch.Tensor
        A tensor of timesteps. May be fractional.
    """
    orig_shape = t.shape
    t = t.reshape(-1)

    if time_max_period is None:
        freq = 2 * torch.arange(time_freqs, device=t.device, dtype=t.dtype) * math.pi
        t = freq * t
        embedding = torch.cat([torch.cos(t), torch.sin(t)], dim=-1)
        return embedding.reshape(*orig_shape, -1)

    t = t * time_max_period
    half = time_freqs // 2
    freqs = torch.exp(-math.log(time_max_period) * torch.arange(start=0, end=half, device=t.device, dtype=torch.float32) / half)
    args = t[:,None] * freqs[None,:]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    return embedding.reshape(*orig_shape, -1)







class MLPBlock(nn.Module):
    """
    MLP block using pure dict config.
    """

    def __init__(self, conf: dict | None):
        super().__init__()

        # 这里返回的是普通 dict
        self.conf = get_module_conf(self.__class__.__name__, conf)

        assert len(self.conf["dims"]) > 1

        self.input_dim = self.conf["input_dim"]
        self.dims = self.conf["dims"]
        self.dropout_rate = self.conf["dropout_rate"]
        self.act_last_layer = self.conf["act_last_layer"]

        # e.g., "SiLU" → nn.SiLU
        act_fn_name = self.conf["act_fn"]
        if not hasattr(nn, act_fn_name):
            raise ValueError(f"Activation {act_fn_name} not found in torch.nn")
        self.act_fn = getattr(nn, act_fn_name)

        # ---- build layers ----
        layers = []
        in_dim = self.input_dim

        for out_dim in self.dims[:-1]:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(self.act_fn())
            layers.append(nn.Dropout(self.dropout_rate))
            in_dim = out_dim

        self.former_layers = nn.Sequential(*layers)

        self.last_linear = nn.Linear(in_dim, self.dims[-1])
        self.last_act = self.act_fn()
        self.last_dropout = nn.Dropout(self.dropout_rate)

    def forward(self, x):
        x = self.former_layers(x)
        x = self.last_linear(x)
        if self.act_last_layer:
            x = self.last_act(x)
        x = self.last_dropout(x)
        return x


class SelfAttention(nn.Module):
    def __init__(self, conf: dict):
        super().__init__()
        self.conf = get_module_conf(self.__class__.__name__, conf)

        # ★ 唯一需要修改的部分：dict 访问方式
        self.input_dim = self.conf["input_dim"]
        self.num_heads = self.conf["num_heads"]
        self.qkv_dim = self.conf["qkv_dim"]
        self.dropout_rate = self.conf["dropout_rate"]
        self.transformer_block = self.conf["transformer_block"]
        self.layer_norm = self.conf["layer_norm"]

        # 以下保持原逻辑完全不动
        self.attention = nn.MultiheadAttention(
            self.qkv_dim,
            self.num_heads,
            dropout=self.dropout_rate,
            batch_first=True
        )
        self.dropout = nn.Dropout(self.dropout_rate)

        if self.layer_norm:
            self.layer_norm_layer = nn.LayerNorm(self.qkv_dim)

        if self.transformer_block:
            self.fc = nn.Linear(self.qkv_dim, self.qkv_dim)

        # ★ dict 访问方式
        self.act_fn = getattr(nn, self.conf["act_fn"])()

    def forward(self, x, mask=None) -> torch.Tensor:
        squeeze = x.ndim == 2
        if squeeze:
            x = x.unsqueeze(1)

        attn_mask = None
        if mask is not None:
            if mask.dim() == 4:
                attn_mask = mask.squeeze(1)
            else:
                attn_mask = mask
            attn_mask = torch.where(attn_mask, 0.0, float('-inf'))

        z, _ = self.attention(x, x, x, attn_mask=attn_mask, need_weights=False)

        if self.transformer_block:
            z = self.dropout(z)
            z = z + x
            if self.layer_norm:
                z = self.layer_norm_layer(z)
            z_ = self.act_fn(self.fc(z))
            z_ = self.dropout(z_)
            z = z + z_

        return z.squeeze(1) if squeeze else z



class SelfAttentionBlock(nn.Module):
    def __init__(self, conf: dict):
        super().__init__()

        self.conf = get_module_conf(self.__class__.__name__, conf)

        # ★ dict 写法替换
        assert isinstance(self.conf["num_heads"], Sequence), "num_heads should be a sequnce."
        assert isinstance(self.conf["qkv_dim"], Sequence), "qkv_dim should be a sequnce."
        assert len(self.conf["num_heads"]) == len(self.conf["qkv_dim"]), \
            "The length of num_heads and qkv_dims should be the same."

        # ★ dict 写法替换
        self.num_heads = self.conf["num_heads"]
        self.qkv_dim = self.conf["qkv_dim"]
        self.dropout_rate = self.conf["dropout_rate"]
        self.transformer_block = self.conf["transformer_block"]
        self.layer_norm = self.conf["layer_norm"]
        self.act_fn = self.conf["act_fn"]

        # 保持逻辑完全一致，只把字段访问改成 dict
        module_confs = [{
            "num_heads": self.num_heads[i],
            "qkv_dim": self.qkv_dim[i],
            "dropout_rate": self.dropout_rate,
            "transformer_block": self.transformer_block,
            "layer_norm": self.layer_norm,
            "act_fn": self.act_fn
        } for i in range(len(self.num_heads))]

        # ★ 不再使用 OmegaConf.create，直接传 dict（和之前所有模块保持一致）
        self.attention_layers = nn.ModuleList([
            SelfAttention(m_conf)
            for m_conf in module_confs
        ])

    def forward(self, x, mask=None):
        z = x
        for attention_layer in self.attention_layers:
            z = attention_layer(z, mask)
        return z



class SeedAttentionPooling(nn.Module):
    def __init__(self, conf: dict):
        super().__init__()
        self.conf = get_module_conf(self.__class__.__name__, conf)

        # ★ dict 访问方式（唯一需要修改的部分）
        self.num_heads = self.conf["num_heads"]
        self.v_dim = self.conf["v_dim"]
        self.seed_dim = self.conf["seed_dim"]
        self.dropout_rate = self.conf["dropout_rate"]
        self.transformer_block = self.conf["transformer_block"]
        self.layer_norm = self.conf["layer_norm"]

        # Trainable seed
        self.seed = nn.Parameter(torch.randn(1, 1, self.seed_dim))
        self.q_proj = nn.Linear(self.seed_dim, self.v_dim)
        self.k_proj = nn.Linear(self.seed_dim, self.v_dim)
        self.v_proj = nn.Linear(self.seed_dim, self.v_dim)

        self.dropout = nn.Dropout(self.dropout_rate)

        if self.layer_norm:
            self.layer_norm_layer = nn.LayerNorm(self.v_dim)

        if self.transformer_block:
            self.fc = nn.Linear(self.v_dim, self.v_dim)

        # ★ dict 访问方式
        self.act_fn = getattr(nn, self.conf["act_fn"])()

    def forward(self, x, mask=None):
        squeeze = x.ndim == 2
        if squeeze:
            x = x.unsqueeze(1)

        batch_size = x.shape[0]

        S = self.seed.expand(batch_size, 1, -1)

        Q = self.q_proj(S)
        K = self.k_proj(x)
        V = self.v_proj(x)

        Q_ = Q.view(batch_size, 1, self.num_heads, self.v_dim // self.num_heads).transpose(1, 2)
        K_ = K.view(batch_size, -1, self.num_heads, self.v_dim // self.num_heads).transpose(1, 2)
        V_ = V.view(batch_size, -1, self.num_heads, self.v_dim // self.num_heads).transpose(1, 2)

        Q_ = Q_.contiguous().view(batch_size * self.num_heads, 1, self.v_dim // self.num_heads)
        K_ = K_.contiguous().view(batch_size * self.num_heads, -1, self.v_dim // self.num_heads)
        V_ = V_.contiguous().view(batch_size * self.num_heads, -1, self.v_dim // self.num_heads)

        A = torch.matmul(Q_, K_.transpose(-2, -1)) / math.sqrt(self.v_dim // self.num_heads)

        if mask is not None:
            mask_expanded = mask[:, 0, 0:1, :].repeat(self.num_heads, 1, 1)
            A = torch.where(mask_expanded, A, torch.tensor(float('-inf'), device=A.device))

        A = F.softmax(A, dim=-1)
        A = torch.matmul(A, V_)

        if self.transformer_block:
            O = (Q_ + A).view(batch_size, self.num_heads, 1, self.v_dim // self.num_heads)\
                        .transpose(1, 2).contiguous().view(batch_size, 1, self.v_dim)
            O = self.dropout(O)
            if self.layer_norm:
                O = self.layer_norm_layer(O)

            O_ = self.act_fn(self.fc(O))
            O_ = self.dropout(O_)
            O = O + O_

            if self.layer_norm:
                O = self.layer_norm_layer(O)
        else:
            O = A.view(batch_size, self.num_heads, 1, self.v_dim // self.num_heads)\
                 .transpose(1, 2).contiguous().view(batch_size, 1, self.v_dim)

        return O.squeeze(1) if squeeze else O



class TokenAttentionPooling(nn.Module):
    def __init__(self, conf: dict):
        super().__init__()
        self.conf = get_module_conf(self.__class__.__name__, conf)

        # ★ dict 访问替换
        self.num_heads = self.conf["num_heads"]
        self.qkv_dim = self.conf["qkv_dim"]
        self.dropout_rate = self.conf["dropout_rate"]

        # ★ 与之前所有模块保持一致：getattr(nn, act_fn)()
        self.act_fn = getattr(nn, self.conf["act_fn"])()

        self.token_embedding = nn.Embedding(1, self.qkv_dim)
        self.attention = nn.MultiheadAttention(
            self.qkv_dim,
            self.num_heads,
            dropout=self.dropout_rate,
            batch_first=True
        )

    def forward(self, x, mask=None):
        squeeze = x.ndim == 2
        if squeeze:
            x = x.unsqueeze(1)

        batch_size = x.shape[0]

        # add token
        class_token = self.token_embedding(
            torch.zeros(batch_size, 1, dtype=torch.long, device=x.device)
        )
        z = torch.cat((class_token, x), dim=1)

        emb, _ = self.attention(z, z, z, need_weights=False)

        z = emb[:, 0, :]

        return z.squeeze(1) if squeeze else z
        

class MeanPooling(nn.Module):
    def __init__(self, conf=None):
        super().__init__()
        self.conf = get_module_conf(self.__class__.__name__, conf)

    def forward(self, x):
        return torch.mean(x, dim=1)


class BasicMultiHeadAttentionPooling(nn.Module):
    """
    A basic multi-head attention pooling module.
    
    Process:
    1. Self-Attention: Perform standard multi-head self-attention on the input sequence
    2. Mean Pooling: Take the average of the attention output along the sequence dimension
    
    Input:  x [B, N, D]
    Output: y [B, D]
    """
    def __init__(self, conf: dict):
        super().__init__()
        self.conf = get_module_conf(self.__class__.__name__, conf)
        
        self.embed_dim = self.conf["embed_dim"]
        self.num_heads = self.conf.get("num_heads", 8)
        self.dropout_rate = self.conf.get("dropout_rate", 0.0)
        self.pool_method = self.conf.get("pool_method", "mean")  # "mean" or "first"
        
        # 标准 Multi-Head Attention
        self.attention = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            dropout=self.dropout_rate,
            batch_first=True
        )
        self.layer_norm = nn.LayerNorm(self.embed_dim)
    
    def forward(self, x, mask=None):
        """
        x: [B, N, D] - Input sequence
        mask: [B, N] - Optional padding mask (True to ignore)
        """
        squeeze = x.ndim == 2
        if squeeze:
            x = x.unsqueeze(1)
        
        # Self-Attention: Q=K=V=x
        # attn_output: [B, N, D]
        attn_output, _ = self.attention(
            query=x, 
            key=x, 
            value=x,
            key_padding_mask=mask,
            need_weights=False
        )
        
        # 残差连接 + LayerNorm
        x = self.layer_norm(x + attn_output)  # [B, N, D]
        
        # Pooling
        if self.pool_method == "mean":
            out = x.mean(dim=1)  # [B, D]
        elif self.pool_method == "first":
            out = x[:, 0, :]  # [B, D]
        else:
            out = x.mean(dim=1)
        
        return out.squeeze(-1) if squeeze and out.ndim > 1 else out
        

class TransformerEncoderBlock(nn.Module):
    """
    Standard Transformer Encoder Block using PyTorch native implementation.

    Input:  x [B, N, input_dim]
    Output: y [B, N, requested_d_model]

    If requested_d_model is not divisible by num_heads, we expand (pad) the
    last dimension to the nearest multiple for attention computation, and
    project back to requested_d_model.
    """
    def __init__(self, conf: dict | None):
        super().__init__()
        self.conf = get_module_conf(self.__class__.__name__, conf)

        self.input_dim = self.conf["input_dim"]
        self.requested_d_model = self.conf["d_model"]
        self.num_heads = self.conf["num_heads"]
        self.num_layers = self.conf["num_layers"]
        self.dim_feedforward = self.conf.get("dim_feedforward", None)
        self.dropout = self.conf.get("dropout_rate", 0.1)
        # torch.nn.TransformerEncoderLayer only accepts activation="relu"/"gelu" as strings.
        # For SiLU, pass a callable (torch.nn.functional.silu).
        act = self.conf.get("act_fn", "gelu")
        if isinstance(act, str):
            act_lower = act.lower()
            self.act_fn = F.silu if act_lower == "silu" else act_lower
        else:
            self.act_fn = act
        self.norm_first = self.conf.get("norm_first", True)

        # Expand last dim if needed so that d_model % num_heads == 0.
        if self.requested_d_model % self.num_heads != 0:
            pad = self.num_heads - (self.requested_d_model % self.num_heads)
            self.d_model = self.requested_d_model + pad
        else:
            self.d_model = self.requested_d_model

        self.input_proj = (
            nn.Linear(self.input_dim, self.d_model)
            if self.input_dim != self.d_model else nn.Identity()
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.num_heads,
            dim_feedforward=self.dim_feedforward or 4 * self.d_model,
            dropout=self.dropout,
            activation=self.act_fn,
            batch_first=True,
            norm_first=self.norm_first,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

        self.norm = nn.LayerNorm(self.d_model) if self.norm_first else nn.Identity()
        self.output_proj = (
            nn.Linear(self.d_model, self.requested_d_model)
            if self.d_model != self.requested_d_model else nn.Identity()
        )

    def forward(self, x: torch.Tensor, mask=None, src_key_padding_mask=None):
        # Allow x: [B, D] -> [B, 1, D] for convenience
        squeeze = x.ndim == 2
        if squeeze:
            x = x.unsqueeze(1)

        x = self.input_proj(x)
        x = self.transformer(x, mask=mask, src_key_padding_mask=src_key_padding_mask)
        x = self.norm(x)
        x = self.output_proj(x)
        return x.squeeze(1) if squeeze else x
        

class EmbeddingMLPBlock(nn.Module):
    """Embedding lookup followed by an MLPBlock."""
    def __init__(self, conf: dict):
        super().__init__()
        self.dims = conf["dims"]
        # conf expected: input_dim, output_dim, dims, dropout_rate, act_fn ...
        self.embedding = nn.Embedding(conf["num_embeddings"], conf["input_dim"])
        self.mlp = MLPBlock(conf)

    def forward(self, x):
        # Handle one-hot input for embedding layer
        if x.dim() > 1 and x.shape[-1] == self.embedding.num_embeddings:
            x = x.argmax(dim=-1)
        
        x = self.embedding(x.long())
        return self.mlp(x)


def get_customized_layer(layer_info, output_dim=None, dropout_rate=None):
    layer_info_conf = copy.deepcopy(layer_info)
    # OmegaConf.set_struct(layer_info_conf, False)  

    modules= []
    last_dim = None
    layer_type = layer_info_conf.pop("layer_type","mlp")

    if layer_type == "embedding":
        lay = EmbeddingMLPBlock(layer_info_conf)
        last_dim = lay.dims[-1]
    elif layer_type == "mlp":
        lay = MLPBlock(layer_info_conf)
        last_dim = lay.dims[-1]
    elif layer_type == "self_attention":
        lay = SelfAttentionBlock(layer_info_conf)
        last_dim = lay.qkv_dim[-1]
    else:
        raise ValueError(f"Unknown layer type: {layer_type}")
    modules.append(lay)

    if output_dim is not None:
        modules.append(nn.Linear(last_dim, output_dim))
        if dropout_rate is not None:
            modules.append(nn.Dropout(dropout_rate))
    return modules



def apply_modules(modules, conditions):
    """Apply modules to conditions."""
    for module in modules:
        conditions = module(conditions)
            
    return conditions
