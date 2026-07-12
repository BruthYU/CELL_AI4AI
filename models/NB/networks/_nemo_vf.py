from megatron.core.transformer.enums import ModelType
import torch.nn as nn
from torch import Tensor

from models.NB.networks._condition_encoders import ConditionEncoder

from models.NB.networks._basic import(
    CellMapEncoder,
    CellMapDecoder,
    DiTBlock,
)

from megatron.core.models.common.language_module.language_module import LanguageModule
from megatron.core.transformer.transformer_config import TransformerConfig


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

    # ----------------------------------------------------------------------

    def _setup(self):
        
        # ---------- CellMap Encoder ----------
        self.cellmap_encoder_conf = {
            "add_pos_emd": True,
            "input_dim": self.cell_dim,
            "output_dim": self.cellmap_width,
            "input_seq_len": self.cell_sentence_len,
            "output_seq_len": self.cell_sentence_len,
            "gene_encoder_type": "tabular_attention",
            "n_hidden": 64,
            "token_dim": max(1, self.cellmap_width // 64),
            "n_tabular_layers": 2,
            "tabular_n_heads": 8,
            "tabular_mlp_ratio": 2.0,
            "dropout_rate": 0.0,
            "act_fn": self.conf.get("act_fn", "SiLU"),
        }
        
        # Merge extra config from yaml if present
        if "cellmap_encoder_kwargs" in self.conf:
            self.cellmap_encoder_conf.update(self.conf["cellmap_encoder_kwargs"])
        self.tgt_cellmap_encoder = CellMapEncoder(self.cellmap_encoder_conf)
        # ---------- CellMap Decoder ----------
        self.cellmap_decoder_conf = {
            "input_dim": self.cellmap_width,
            "output_dim": self.cell_dim,
            "output_seq_len": self.cell_sentence_len,
            "decoder_distribution": "nb",
            "dims": [self.cellmap_width, self.cellmap_width * 2, self.cell_dim * 2],
            "dropout_rate": self.conf.get("decoder_dropout", 0.0),
            "act_fn": self.conf.get("act_fn", "SiLU"),
            "nb_min_dispersion": self.conf.get("nb_min_dispersion", 1e-4),
        }
        
        # Merge extra config from yaml if present
        if "cellmap_decoder_kwargs" in self.conf:
            self.cellmap_decoder_conf.update(self.conf["cellmap_decoder_kwargs"])
        self.tgt_cellmap_decoder = CellMapDecoder(self.cellmap_decoder_conf)

        # ---------- Condition Encoder ----------
        self.condition_encoder = ConditionEncoder(self.conf)

        # ---------- DiT Velocity Field ----------
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
        # x_t: [B, S, 512]
        aggregated_condition = self.condition_encoder(cond_batch) # [B, 512]
        pred_vf = x_t
        for block in self.dit_blocks:
            pred_vf = block(pred_vf, aggregated_condition)
        return pred_vf

    def tower_parameters(self):
        yield from self.tgt_cellmap_encoder.parameters()
        yield from self.tgt_cellmap_decoder.parameters()










