import torch
import torch.nn as nn
import torch.nn.functional as F
from models.PRIOR_LLM.networks._llm_utils import get_transformer_backbone
from megatron.core.models.common.language_module.language_module import LanguageModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.enums import ModelType
from torch import Tensor
def build_mlp(
    in_dim: int,
    out_dim: int,
    hidden_dim: int,
    n_layers: int,
    dropout: float = 0.0,
    activation: nn.Module = nn.ReLU,  # default to nn.ReLU class
) -> nn.Sequential:
    """
    Build an MLP of `n_layers` from `in_dim` to `out_dim`.
    ...
    """
    layers = []
    if n_layers < 1:
        raise ValueError("n_layers must be >= 1")

    if n_layers == 1:
        layers.append(nn.Linear(in_dim, out_dim))
    else:
        # First layer
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(activation())  # instantiate the class
        layers.append(nn.Dropout(dropout))

        # Intermediate layers
        for _ in range(n_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(activation())  # instantiate again
            layers.append(nn.Dropout(dropout))

        # Final layer
        layers.append(nn.Linear(hidden_dim, out_dim))

    return nn.Sequential(*layers)


class CellMapEncoder(nn.Module):
    def __init__(self, conf: dict):
        super().__init__()
        self.conf = conf
        self.input_dim = self.conf["cell_dim"]
        self.input_seq_len = self.conf["cell_sentence_len"]
        self.transformer_backbone_kwargs = self.conf["transformer_backbone_kwargs"]
        self.backbone_name = self.conf["transformer_backbone_kwargs"]["backbone_name"]


        self.gene_mlp = nn.Linear(self.input_dim, self.transformer_backbone_kwargs["hidden_size"])
        self.transformer_backbone, _ = get_transformer_backbone(self.backbone_name, self.conf["transformer_backbone_kwargs"])

        
       
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.gene_mlp(x)
        x = self.transformer_backbone(inputs_embeds=x)['last_hidden_state']
        return x


        
    


class CellMapDecoder(nn.Module):
    def __init__(self, conf: dict):
        super().__init__()
        
        self.conf = conf
        self.transformer_backbone_kwargs = self.conf["transformer_backbone_kwargs"]
        self.hidden_dim = self.transformer_backbone_kwargs["hidden_size"]
        self.n_decoder_layers = self.conf["n_decoder_layers"]

        self.output_dim = self.conf["cell_dim"]
        self.project_out = build_mlp(
            in_dim=self.hidden_dim,
            out_dim=self.output_dim,
            hidden_dim=self.hidden_dim,
            n_layers=self.n_decoder_layers,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, N, D] -> [B, output_seq_len, dims[-1]]
        x = self.project_out(x)
        return x


class CellMapClassifier(nn.Module):
    def __init__(self, conf: dict):
        super().__init__()
        self.conf = conf
        self.transformer_backbone_kwargs = self.conf["transformer_backbone_kwargs"]
        self.hidden_dim = self.transformer_backbone_kwargs["hidden_size"]
        self.hidden_channels = self.conf["hidden_channels"]
        self.batch_class = self.conf["batch_class"]
        self.cell_class = self.conf["cell_class"]

        self.conv = nn.Sequential(
            nn.Conv2d(1, self.hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden_channels, self.hidden_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),  # [B, C, 1, 1]
        )

        out_dim = self.hidden_channels * 2
        self.batch_head = nn.Linear(out_dim, self.batch_class)
        self.cell_head = nn.Linear(out_dim, self.cell_class)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)  # [B, 1, N, D]
        feat = self.conv(x).flatten(1)  # [B, out_dim]
        return self.batch_head(feat), self.cell_head(feat)



class SetAttentionPooling(nn.Module):
    """
    Permutation-invariant set pooling:
    x: [B, N, D]  ->  pooled: [B, K, D]
    """
    # def __init__(self, d_model: int, K: int, hidden: int = 256):
    def __init__(self, conf):
        super().__init__()
        self.deepsets_kwargs = conf["deepsets_kwargs"]
        self.K = self.deepsets_kwargs["K"]
        self.d_model = self.deepsets_kwargs["d_model"]
        self.hidden = self.deepsets_kwargs["hidden"]

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



class AE_Module(LanguageModule):
    def __init__(self, conf, base_config: TransformerConfig):
        super().__init__(base_config)
        self.config = base_config
        self.model_type = ModelType.encoder_and_decoder
        self.pre_process = True
        self.post_process = True
        self.share_embeddings_and_output_weights = True

        self.use_deepsets = conf["use_deepsets"]

        self.cellmap_encoder = CellMapEncoder(conf)

        self.cellmap_classifier = CellMapClassifier(conf)
        if self.use_deepsets:   
            self.cellmap_deepsets = SetAttentionPooling(conf)
        self.cellmap_decoder = CellMapDecoder(conf)

    def set_input_tensor(self, input_tensor: Tensor):
        self.input_tensor = input_tensor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cellmap_encoder(x) 
        if self.use_deepsets: 
            x = self.cellmap_deepsets(x)
        batch_logits, cell_logits = self.cellmap_classifier(x)
        x = self.cellmap_decoder(x)
        return x, batch_logits, cell_logits
