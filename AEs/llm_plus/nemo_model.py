import torch
from functools import partial


from AEs.llm_plus.encoder_decoder import AE_Module

from omegaconf import OmegaConf
import torch.nn as nn
from bionemo.llm.model.biobert.lightning import get_batch_on_this_context_parallel_rank
import pytorch_lightning as pl
from bionemo.llm.api import MegatronLossType, MegatronModelType
from models.nemo_model_base import ModelInterfaceBase
from typing import Iterator, Optional, Dict, Tuple
from torch import Tensor
from bionemo.llm.model.loss import _Nemo2CompatibleLossReduceMixin
from megatron.core import parallel_state, tensor_parallel
from nemo.utils import logging
from nemo.collections.nlp.modules.common.megatron.utils import average_losses_across_data_parallel_group
from nemo.lightning.megatron_parallel import (
    MegatronLossReduction,
)
from nemo.lightning.pytorch.optim import MegatronOptimizerModule
from megatron.core.optimizer import OptimizerConfig
from megatron.core.transformer.transformer_config import TransformerConfig
from bionemo.llm.model.lr_scheduler import WarmupAnnealDecayHoldScheduler
default_transformer_config = TransformerConfig(
    num_layers=2,
    hidden_size=12,
    num_attention_heads=4,
    use_cpu_initialization=True,
    pipeline_dtype=torch.float32,
)
from _metrics import (
    Classification_Loss,
    MMDLoss, 
    MSELoss, 
    MAELoss, 
    Delta_Pearson, 
    Pearson,
    R2_Score)

class metrics:
    def __init__(self):
        self.mmd = MMDLoss(kernel="energy")
        self.mse = MSELoss(batch=True, reduction="mean", scale=10.0)
        self.mae = MAELoss(batch=True, reduction="mean", scale=10.0)
        self.delta_pearson = Delta_Pearson(scale=1.0)
        self.r2_score = R2_Score(scale=1.0)     
        self.pearson = Pearson(scale=1.0)
        self.flow_mse = MSELoss(batch=False, reduction="mean", scale=100.0)     
        self.classification_loss = Classification_Loss(scale=0.1)

class AE_LossWithReduction(_Nemo2CompatibleLossReduceMixin, MegatronLossReduction):
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
        unreduced_loss = forward_out['loss']
        # Normal case: average loss across data parallel group
        reduced = average_losses_across_data_parallel_group([unreduced_loss])
        return unreduced_loss, {'avg': reduced}


class llm_plus_Nemo_Model(ModelInterfaceBase[MegatronModelType, MegatronLossType]):
    def __init__(self, conf, model_transform=None, configure_init_model_parallel=False):
        super().__init__(
            model_transform=model_transform,
            configure_init_model_parallel=configure_init_model_parallel,
        )
        
        self.conf = conf
        self.config = default_transformer_config
        self.model_conf = conf["prepare_model"]
        self.metric_utils = metrics()

        self.configure_model()



        
       


    def data_step(self, dataloader_iter: Iterator) -> Dict:
        batch = next(dataloader_iter)
        if isinstance(batch, tuple) and len(batch) == 3:
            _batch = batch[0]
        else:
            _batch = batch
        def to_cuda(x):
            if isinstance(x, torch.Tensor):
                return x.cuda(non_blocking=True)
            else:
                return x
        _batch = {k: to_cuda(v) for k, v in _batch.items()}
        return get_batch_on_this_context_parallel_rank(_batch)  
    
    def configure_model(self):
        # Velocity Field Model
        self.module = AE_Module(self.model_conf, self.config)


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
        查找可用端口, 从start_port开始尝试
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

    # Megatron Required
    def loss_reduction(self, *args, **kwargs):
        return AE_LossWithReduction(**kwargs)

    def forward_step(self, batch, mode="train"):
        if mode == "train":
            return self.loss_fn(batch)
        else:
            return self.eval_fn(batch)
    

    def training_step(self, batch, batch_idx=None):
        forward_out = self.forward_step(batch, mode="train")
        log_info = self.add_mode_prefix(forward_out['log_info'], "train")
        self.log_dict(log_info, on_step=True, on_epoch=True, prog_bar=True)
        return forward_out
    
    def validation_step(self, batch, batch_idx=None):
        
        forward_out = self.forward_step(batch, mode="val")
        log_info = self.add_mode_prefix(forward_out['log_info'], "val")
        print(log_info)
        self.log_dict(log_info, on_step=True, on_epoch=True, prog_bar=False)
        return forward_out

    def predict_step(self, batch, batch_idx=None):
        return self.eval_fn(batch, test_mode=True)



    def save_to_torch_ckpt(self, ckpt_dir: str, out_path: str) -> None:
        """Save the model to a torch checkpoint."""
        '''
        self.save_to_torch_ckpt('/mnt/shared-storage-user/gaozhangyang/workspace/FoldCompression/results/struct_compress/eval_nn10_in_str_out_str_lr1e5_10Xsteps_droptoken/checkpoints/epoch=0-step=4012999-consumed_samples=64208000.0/weights', '/mnt/shared-storage-user/gaozhangyang/workspace/FoldCompression/results/struct_compress/eval_nn10_in_str_out_str_lr1e5_10Xsteps_droptoken/checkpoints/10Xsteps_droptoken_step_4012999_weights.pt')
        '''
        
        # 1. 生成 sharded_state_dict
        sharded_sd = self.module.sharded_state_dict()  # <— 一定要在 parallel 初始化后调用
        ckpt = self.trainer.strategy.checkpoint_io.load_checkpoint(
                str(ckpt_dir),
                sharded_state_dict=sharded_sd,            # <<< 关键：这里不能省略
            )  # 底层会调用 dist_checkpointing.load(sharded_state_dict=…, …)
        torch.save(ckpt, out_path)
        print(f"✅ 转换成功：{out_path}")

    
    def reshape_t(self, t, n):
        if t.dim()==0:
            t = t[None].expand(n) 
        return t
    
    def add_mode_prefix(self, data_dict: Dict, mode: str) -> Dict:
        assert mode in ["train", "val", "test"], f"mode must be 'train', 'val', or 'test', got {mode}"
        return {f"{mode}_{key}": value for key, value in data_dict.items()}
    
    def forward(self, t, x_t, cond):
        return self.module(t, x_t, cond)


    
    def loss_fn(self, batch):
        seq_len = self.conf["dataset"]["cell_sentence_len"]
        batch_size = self.conf["dataset"]["batch_size"]
        batch_labels = batch["batch_idx"]
        cell_labels = batch["cell_idx"]
        # Gather and Reshape Batch
        device = batch["ctrl_cell_emb"].device
        src_batch = batch["ctrl_cell_emb"].reshape(batch_size, seq_len, -1) # [B, N, D]
        tgt_batch = batch["pert_cell_emb"].reshape(batch_size, seq_len, -1) # [B, N, D]

        

       
        


        # TGT VAE Loss
        tgt_recon, tgt_batch_logits, tgt_cell_logits = self.module(tgt_batch)
        # tgt_vae_mse = self.metric_utils.mse(tgt_recon, tgt_batch)
        tgt_vae_mmd = self.metric_utils.mmd(tgt_recon, tgt_batch)
        tgt_cls_loss = self.metric_utils.classification_loss(tgt_batch_logits, batch_labels) + \
            self.metric_utils.classification_loss(tgt_cell_logits, cell_labels)
        
        # SRC VAE Loss
        # src_recon, src_batch_logits, src_cell_logits = self.module(src_batch)
        # src_vae_mse = self.metric_utils.mse(src_recon, src_batch)
        # src_vae_mmd = self.metric_utils.mmd(src_recon, src_batch)
        # src_cls_loss = self.metric_utils.classification_loss(src_batch_logits, batch_labels) + \
        #     self.metric_utils.classification_loss(src_cell_logits, cell_labels)
        


        

        # final_loss = tgt_vae_mse + tgt_vae_mmd + tgt_cls_loss
        final_loss = tgt_vae_mmd + tgt_cls_loss
        log_info = {"tgt_vae_mmd": tgt_vae_mmd.item(), "tgt_cls_loss": tgt_cls_loss.item(), "final_loss": final_loss.item()}
        # final_loss = src_vae_mse + src_vae_mmd + tgt_vae_mse + tgt_vae_mmd + tgt_cls_loss + src_cls_loss
        # log_info = {"src_vae_mse": src_vae_mse.item(), "src_vae_mmd": src_vae_mmd.item(), "tgt_vae_mse": tgt_vae_mse.item(), "tgt_vae_mmd": tgt_vae_mmd.item(), "final_loss": final_loss.item()}
        
        return {'loss': final_loss, 'log_info': log_info}
        



    def eval_fn(self, batch, test_mode=False):
        
        seq_len = self.conf["dataset"]["cell_sentence_len"]
        batch_size = self.conf["dataset"]["batch_size"]
        
        # Gather and Reshape Batch
        device = batch["ctrl_cell_emb"].device
        src_batch = batch["ctrl_cell_emb"].reshape(batch_size, seq_len, -1) # [B, N, D]
        tgt_batch = batch["pert_cell_emb"].reshape(batch_size, seq_len, -1) # [B, N, D]


        # TGT VAE Loss
        tgt_recon, _, _ = self.module(tgt_batch)
        # SRC VAE Loss
        # src_recon = self.module(src_batch)
        

        delta_pearson, mae, mse, gt_delta_pearson = self.compute_metrics(src_batch, tgt_batch, tgt_recon)
        log_info = {"delta_pearson": delta_pearson, "mse": mse, "mae": mae,"gt_delta_pearson": gt_delta_pearson}
        return {"loss":torch.zeros(1, device=device), "log_info": log_info}

    def compute_metrics(self, src_batch, tgt_batch, tgt_pred):
        delta_pearson = self.metric_utils.delta_pearson(src_batch, tgt_pred, tgt_batch)
        mae = self.metric_utils.mae(tgt_batch, tgt_pred)
        mse = self.metric_utils.mse(tgt_batch, tgt_pred)
        gt_delta_pearson = self.metric_utils.delta_pearson(src_batch, tgt_batch, tgt_batch)
        return delta_pearson, mae, mse, gt_delta_pearson

    
    
    


        
        
        



        
        






        




