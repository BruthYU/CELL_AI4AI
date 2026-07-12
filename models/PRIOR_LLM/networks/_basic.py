import abc
import math
from collections.abc import Callable, Sequence
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
import copy
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
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

class CellMapEncoder(nn.Module):
    def __init__(self, conf: dict):
        super().__init__()
        self.conf = get_module_conf(self.__class__.__name__, conf)
        self.input_dim = self.conf["input_dim"]
        self.input_seq_len = self.conf["input_seq_len"]
        self.dims = self.conf["dims"]
        self.output_dim = self.conf["output_dim"]
        self.output_seq_len = self.conf["output_seq_len"]
        self.add_pos_emd = self.conf["add_pos_emd"]
        if self.add_pos_emd:
            self.pos_emd = nn.Embedding(self.input_seq_len, self.input_dim)
        
       

        gene_mlp_conf = {
            "input_dim": self.input_dim,
            "dims": self.dims,
        }
        self.gene_mlp = MLPBlock(gene_mlp_conf)

        set_attention_pooling_conf = {
            "d_model": self.dims[-1],
            "K": self.output_seq_len,
            "hidden": self.dims[-1],
        }
        self.set_attention_pooling = SetAttentionPooling(set_attention_pooling_conf)
        
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # [B, N, D] -> [B, output_seq_len, dims[-1]]
        if self.add_pos_emd:
            pos_ids = torch.arange(self.input_seq_len, device=x.device)      # [N]
            pos_emd = self.pos_emd(pos_ids)
            x = x + pos_emd
        x = self.gene_mlp(x)
        x = self.set_attention_pooling(x)
        return x


        


class CellMapDecoder(nn.Module):
    def __init__(self, conf: dict):
        super().__init__()
        self.conf = get_module_conf(self.__class__.__name__, conf)
        self.input_dim = self.conf["input_dim"]
        self.dims = self.conf["dims"]
        self.output_dim = self.conf["output_dim"]
        
        self.output_seq_len = self.conf["output_seq_len"]

        gene_mlp_conf = {
            "input_dim": self.input_dim,
            "dims": self.dims,
        }
        self.gene_mlp = MLPBlock(gene_mlp_conf)

        set_attention_pooling_conf = {
            "d_model": self.input_dim,
            "K": self.output_seq_len,
            "hidden": self.input_dim,
        }
        self.set_attention_pooling = SetAttentionPooling(set_attention_pooling_conf)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, N, D] -> [B, output_seq_len, dims[-1]]
        x = self.set_attention_pooling(x)
        x = self.gene_mlp(x)
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



class FilmBlock(nn.Module):
    def __init__(self, conf: dict):
        super().__init__()
        self.conf = get_module_conf(self.__class__.__name__, conf)

        # 原来是 self.conf.input_dim → 改为 dict 访问方式
        self.input_dim = self.conf["input_dim"]

        # 原来是 getattr(nn, self.conf.act_fn)() → dict 访问方式
        self.act_fn = getattr(nn, self.conf["act_fn"])()

        # 原逻辑保持不变
        self.film_generator = nn.Linear(self.input_dim, self.input_dim * 2)

    def forward(self, x, cond):
        gamma_beta = self.film_generator(cond)
        gamma, beta = torch.split(gamma_beta, self.input_dim, dim=-1)
        return self.act_fn(gamma * x + beta)


class ResNetBlock(nn.Module):
    def __init__(self, conf: dict):
        super().__init__()
        self.conf = get_module_conf(self.__class__.__name__, conf)

        # 原 self.conf.xx → dict 访问方式
        self.input_dim = self.conf["input_dim"]
        self.hidden_dims = self.conf["hidden_dims"]
        self.projection_dims = self.conf["projection_dims"]
        self.act_fn = self.conf["act_fn"]
        self.dropout_rate = self.conf["dropout_rate"]

        # 构造子模块的配置（全是 dict）
        mlp_1_conf = {
            "input_dim": self.input_dim,
            "dims": self.hidden_dims,
            "act_fn": self.act_fn,
            "dropout_rate": self.dropout_rate,
            "act_last_layer": True,
        }
        mlp_2_conf = {
            "input_dim": self.hidden_dims[-1],
            "dims": self.hidden_dims + [self.input_dim],
            "act_fn": self.act_fn,
            "dropout_rate": self.dropout_rate,
            "act_last_layer": True,
        }
        proj_conf = {
            "input_dim": self.projection_dims[0],
            "dims": self.projection_dims,
            "act_fn": self.act_fn,
            "dropout_rate": self.dropout_rate,
            "act_last_layer": True,
        }

        # 不再用 OmegaConf.create —— 直接传 dict（符合你现在的 MLPBlock 写法）
        self.mlp_block_1 = MLPBlock(mlp_1_conf)
        self.mlp_block_2 = MLPBlock(mlp_2_conf)
        self.cond_proj = MLPBlock(proj_conf)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.mlp_block_1(x)
        h = h + self.cond_proj(cond)
        h = self.mlp_block_2(h)
        return h + x


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

