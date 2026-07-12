
import torch
import os
import json
from nemo.core.classes import ModelPT
from functools import partial
from tqdm import tqdm
# Solver Tools
from torchdiffeq import odeint
# OT & Probability Path

from models.JIT_FP.flow.interpolant import Interpolant
from models.JIT_FP.networks._nemo_vf import ConditionalVelocityField
import models.JIT_FP.networks._basic as nn_basic

from torch.distributed.checkpoint import load
from torch.distributed.checkpoint.state_dict import get_model_state_dict, set_model_state_dict
import torch.distributed.checkpoint as dcp

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
    MMDLoss, 
    MSELoss, 
    MAELoss, 
    Delta_Pearson, 
    Pearson,
    R2_Score,
    Classification_Loss)

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

class FM_LossWithReduction(_Nemo2CompatibleLossReduceMixin, MegatronLossReduction):
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


class JIT_FP_Nemo_Model(ModelInterfaceBase[MegatronModelType, MegatronLossType]):
    def __init__(self, conf, model_transform=None, configure_init_model_parallel=False):
        super().__init__(
            model_transform=model_transform,
            configure_init_model_parallel=configure_init_model_parallel,
        )
        
        self.conf = conf
        self.config = default_transformer_config
        
        self.model_conf = conf["prepare_model"]

        self.condition_mode = self.model_conf["condition_mode"]
        self.regularization = self.model_conf["regularization"]

        self.configure_model()

        # flow utils
        self.flow_utils = Interpolant(conf["Interpolant"])
        self.metric_utils = metrics()
        self._val_steps_in_epoch = 0

        
       


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
        self.module = ConditionalVelocityField(self.model_conf, self.config)

        # dict-access version
        if self.model_conf["megatron_ckpt"] is not None:
            self.load_from_megatron_ckpt(self.model_conf["megatron_ckpt"])

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
        查找可用端口，从start_port开始尝试
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
        return FM_LossWithReduction(**kwargs)

    def forward_step(self, batch, mode="train"):
        if mode == "train":
            return self.loss_fn(batch)
        else:
            return self.eval_fn(batch)
    

    def training_step(self, batch, batch_idx=None):
        forward_out = self.forward_step(batch, mode="train")
        log_info = self.add_mode_prefix(forward_out['log_info'], "train")
        self.log_dict(log_info, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return forward_out

    def on_train_epoch_start(self):
        self._val_steps_in_epoch = 0
    
    def validation_step(self, batch, batch_idx=None):
        
        forward_out = self.forward_step(batch, mode="val")
        log_info = self.add_mode_prefix(forward_out['log_info'], "val")
        print(log_info)
        self._persist_metric_log(log_info, tag="val", batch=batch)
        self._val_steps_in_epoch += 1
        self.log_dict(log_info, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        return forward_out

    def on_train_epoch_end(self):
        # Force one val metrics write if Lightning validation loop is skipped.
        if self._val_steps_in_epoch > 0:
            return

        trainer = getattr(self, "trainer", None)
        if trainer is None:
            return
        datamodule = getattr(trainer, "datamodule", None)
        if datamodule is None:
            return

        val_loader = datamodule.val_dataloader()
        if val_loader is None:
            return

        val_iter = iter(val_loader)
        val_batch = next(val_iter, None)
        if val_batch is None:
            print("[warn] val_forced skipped: val_dataloader returned no batch.")
            return

        forced_iter = iter([val_batch])
        forced_batch = self.data_step(forced_iter)

        was_training = self.training
        self.eval()
        with torch.no_grad():
            forward_out = self.forward_step(forced_batch, mode="val")
        forced_log_info = self.add_mode_prefix(forward_out["log_info"], "val")
        print(forced_log_info)
        self._persist_metric_log(forced_log_info, tag="val_forced", batch=forced_batch)
        self.log_dict(forced_log_info, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        if was_training:
            self.train()

    def predict_step(self, batch, batch_idx=None):
        return self.eval_fn(batch, test_mode=True)

    def _persist_metric_log(self, log_info: Dict[str, Tensor], tag: str, batch: Dict[str, Tensor] | None = None):
        rank = int(getattr(self, "global_rank", 0))

        group = self.conf.get("experiment", {}).get("group", "default")
        out_dir = os.path.join("outputs", group)
        os.makedirs(out_dir, exist_ok=True)
        if rank == 0:
            out_file = os.path.join(out_dir, "val_metrics_forced.jsonl")
        else:
            out_file = os.path.join(out_dir, f"val_metrics_forced_rank{rank}.jsonl")

        payload = {
            "tag": tag,
            "epoch": int(getattr(self, "current_epoch", -1)),
            "global_step": int(getattr(self, "global_step", -1)),
            "global_rank": rank,
        }
        for k, v in log_info.items():
            if isinstance(v, torch.Tensor):
                t = v.detach().cpu()
                if t.numel() == 1:
                    payload[k] = float(t.item())
                else:
                    payload[k] = float(t.float().mean().item())
            elif isinstance(v, (int, float, str, bool)):
                payload[k] = v
            else:
                payload[k] = str(v)

        if batch is not None:
            for meta_key in ("batch_idx", "cell_idx", "pert_idx"):
                if meta_key not in batch:
                    continue
                meta_val = batch[meta_key]
                if isinstance(meta_val, torch.Tensor):
                    t = meta_val.detach().cpu().reshape(-1)
                    if t.numel() == 1:
                        payload[meta_key] = int(t.item())
                    else:
                        payload[meta_key] = [int(x) for x in t.tolist()]
                elif isinstance(meta_val, (int, float, str, bool)):
                    payload[meta_key] = meta_val
                else:
                    payload[meta_key] = str(meta_val)

        with open(out_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")



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
        # NOTE: do NOT trust config batch_size here; the actual batch can be smaller
        # (e.g., last batch) and `ctrl_cell_emb` may come either as [B, S, D] or [B*S, D].
        seq_len = self.conf["dataset"]["cell_sentence_len"]
        
        # Gather and Reshape Batch
        device = batch["ctrl_cell_emb"].device
        ctrl = batch["ctrl_cell_emb"]
        pert = batch["pert_cell_emb"]
        batch_labels = batch["batch_idx"]
        cell_labels = batch["cell_idx"]

        # Infer batch_size (and prefer runtime seq_len if available)
        if ctrl.dim() == 3:
            batch_size = ctrl.shape[0]
            seq_len = ctrl.shape[1]
            src_batch = ctrl  # [B, S, D]
            tgt_batch = pert if pert.dim() == 3 else pert.reshape(batch_size, seq_len, -1)
        else:
            total = ctrl.shape[0]
            if total % seq_len != 0:
                raise ValueError(
                    f"ctrl_cell_emb first dim ({total}) must be divisible by seq_len ({seq_len})."
                )
            batch_size = total // seq_len
            src_batch = ctrl.reshape(batch_size, seq_len, -1)  # [B, S, D]
            tgt_batch = pert.reshape(batch_size, seq_len, -1)  # [B, S, D]

        


        # TGT VAE Loss
        tgt_encoded_map = self.module.cellmap_encoder(tgt_batch)
        tgt_batch_logits, tgt_cell_logits = self.module.cellmap_classifier(tgt_encoded_map)
        tgt_decoded_map = self.module.cellmap_decoder(tgt_encoded_map)
        tgt_vae_mse = self.metric_utils.mse(tgt_decoded_map, tgt_batch)
        tgt_vae_mmd = self.metric_utils.mmd(tgt_decoded_map, tgt_batch)
        tgt_cls_loss = self.metric_utils.classification_loss(tgt_batch_logits, batch_labels) + \
            self.metric_utils.classification_loss(tgt_cell_logits, cell_labels)
        
        # SRC VAE Loss
        src_encoded_map = self.module.cellmap_encoder(src_batch)
        src_batch_logits, src_cell_logits = self.module.cellmap_classifier(src_encoded_map)
        src_decoded_map = self.module.cellmap_decoder(src_encoded_map)
        src_vae_mse = self.metric_utils.mse(src_decoded_map, src_batch)
        src_vae_mmd = self.metric_utils.mmd(src_decoded_map, src_batch)
        src_cls_loss = self.metric_utils.classification_loss(src_batch_logits, batch_labels) + \
            self.metric_utils.classification_loss(src_cell_logits, cell_labels)

        # Flow Loss
        # x_t, v_t, t = self.flow_utils.corrupt(tgt_encoded_map, device)
        x_t, v_t, t = self.flow_utils.interpolate(src_encoded_map, tgt_encoded_map, device)

        # reshape conditions
        cond_batch = {}
        # cond_batch["src_batch"] = src_batch
        cond_batch["t"] = t
        cond_keys = self.conf["layers_before_pool"].keys()
        for k in cond_keys:
            cov = batch[k]
            if cov.dim() == 3:
                cond_batch[k] = cov  # [B, S, C]
            elif cov.shape[0] == batch_size:
                cond_batch[k] = cov.unsqueeze(1).expand(-1, seq_len, -1)  # [B, S, C]
            elif cov.shape[0] == batch_size * seq_len:
                cond_batch[k] = cov.reshape(batch_size, seq_len, -1)  # [B, S, C]
            else:
                raise ValueError(
                    f"Unexpected shape for condition '{k}': {tuple(cov.shape)} "
                    f"(batch_size={batch_size}, seq_len={seq_len})."
                )
        
        # ------------------------------------------------------------
        # JIT ablation: (pred_type x v) × (loss_type x v)
        #
        # - pred_type == "x": model predicts x1 (tgt_encoded_map)
        # - pred_type == "v": model predicts v = x1 - x0
        #
        # - loss_type == "x": supervise x1_pred vs tgt_encoded_map
        # - loss_type == "v": supervise v_pred  vs v_t (from interpolate)
        # ------------------------------------------------------------
        pred_type = self.model_conf.get("pred_type", "x")   # "x" or "v"
        loss_type = self.model_conf.get("loss_type", "v")   # "x" or "v"

        pred = self.module(x_t, cond_batch)  # [B, W, W]

        if pred_type == "x":
            x1_pred = pred
            v_pred = x1_pred - src_encoded_map
        elif pred_type == "v":
            v_pred = pred
            x1_pred = src_encoded_map + v_pred
        else:
            raise ValueError(f"Unknown pred_type: {pred_type}. Expected 'x' or 'v'.")

        x_loss = self.metric_utils.flow_mse(x1_pred, tgt_encoded_map)
        v_loss = self.metric_utils.flow_mse(v_pred, v_t)

        if loss_type == "x":
            flow_loss = x_loss
        elif loss_type == "v":
            flow_loss = v_loss
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}. Expected 'x' or 'v'.")

        tgt_loss = tgt_vae_mse + tgt_vae_mmd + tgt_cls_loss
        src_loss = src_vae_mse + src_vae_mmd + src_cls_loss

        final_loss =  tgt_loss + self.model_conf["src_loss_weight"] * src_loss + flow_loss
        # log_info = {"tgt_vae_mse": tgt_vae_mse.item(), "tgt_vae_mmd": tgt_vae_mmd.item(), "src_vae_mse": src_vae_mse.item(), "src_vae_mmd": src_vae_mmd.item(), "flow_loss": flow_loss.item(), "final_loss": final_loss.item()}
        log_info = {
            "src_vae_mse": src_vae_mse.item(),
            "src_vae_mmd": src_vae_mmd.item(),
            "tgt_vae_mse": tgt_vae_mse.item(), 
            "tgt_vae_mmd": tgt_vae_mmd.item(),
            "tgt_cls_loss": tgt_cls_loss.item(),
            "src_cls_loss": src_cls_loss.item(),
            "x_loss": x_loss.item(),
            "v_loss": v_loss.item(),
            "flow_loss": flow_loss.item(),
            "final_loss": final_loss.item(),
        }
        return {'loss': final_loss, 'log_info': log_info}
        



    def eval_fn(self, batch, test_mode=False):
        
        seq_len = self.conf["dataset"]["cell_sentence_len"]
        cell_dim = self.model_conf["cell_dim"]
        ctrl = batch["ctrl_cell_emb"]
        if ctrl.dim() == 3:
            batch_size = ctrl.shape[0]
            seq_len = ctrl.shape[1]
        else:
            total = ctrl.shape[0]
            if total % seq_len != 0:
                raise ValueError(
                    f"ctrl_cell_emb first dim ({total}) must be divisible by seq_len ({seq_len})."
                )
            batch_size = total // seq_len
        

        # Gather and Reshape Batch
        device = batch["ctrl_cell_emb"].device
        src_batch = batch["ctrl_cell_emb"].reshape(batch_size, seq_len, cell_dim) # [B, N, D] src for ctrl cell map
        tgt_batch = batch["pert_cell_emb"].reshape(batch_size, seq_len, cell_dim) # [B, N, D] tgt for pert cell map

        
        # no flow prediction
        src_encoded_map = self.module.cellmap_encoder(src_batch)
        tgt_decoded_map = self.module.cellmap_decoder(src_encoded_map)
        mix_vae_mse = self.metric_utils.mse(tgt_decoded_map, tgt_batch)
        mix_vae_mmd = self.metric_utils.mmd(tgt_decoded_map, tgt_batch)
        mix_vae_delta_pearson = self.metric_utils.delta_pearson(src_batch, tgt_decoded_map, tgt_batch)
        
        # reshape conditions
        # conditions = {...}
        # cond = self.module.condition_encoder(conditions)
        
        # shape = [batch_size, cellmap_width, cellmap_width]
        # prior  = self.flow_utils.prior.sample(shape).to(device)
        prior = src_encoded_map
        

        # reshape conditions
        cond_batch = {}
        # cond_batch["src_batch"] = src_batch
        cond_keys = self.conf["layers_before_pool"].keys()
        for k in cond_keys:
            cov = batch[k]
            if cov.dim() == 3:
                cond_batch[k] = cov  # [B, S, C]
            elif cov.shape[0] == batch_size:
                cond_batch[k] = cov.unsqueeze(1).expand(-1, seq_len, -1)  # [B, S, C]
            elif cov.shape[0] == batch_size * seq_len:
                cond_batch[k] = cov.reshape(batch_size, seq_len, -1)  # [B, S, C]
            else:
                raise ValueError(
                    f"Unexpected shape for condition '{k}': {tuple(cov.shape)} "
                    f"(batch_size={batch_size}, seq_len={seq_len})."
                )

        
        ts = torch.linspace(0., 1., self.conf['experiment']['time_step'], device=device)
        x_t = prior
        for t0, t1 in tqdm(zip(ts[:-1], ts[1:]), desc=f"[Rank {self.global_rank}] Entering Evaluation"):
            cond_batch["t"] = t0.expand(batch_size)
            dt = (t1 - t0).expand(batch_size)
            # Align eval with training pred_type:
            # - pred_type=="x": model predicts x1, convert to v = x1 - x0(prior)
            # - pred_type=="v": model predicts v directly
            pred_type = self.model_conf.get("pred_type", "x")  # "x" or "v"
            pred = self.module(x_t, cond_batch)  # [B, cellmap_width, cellmap_width]
            if pred_type == "x":
                exp_vf = pred - prior  # v_pred = x1_pred - x0
            elif pred_type == "v":
                exp_vf = pred
            else:
                raise ValueError(f"Unknown pred_type: {pred_type}. Expected 'x' or 'v'.")
            x_t = self.flow_utils.denoise(x_t, exp_vf, dt)

        tgt_pred = self.module.cellmap_decoder(x_t)


        if test_mode:
            return tgt_pred

        # metrics
        delta_pearson, mae, mse, gt_delta_pearson = self.compute_metrics(src_batch, tgt_batch, tgt_pred)
        log_info = {"delta_pearson": delta_pearson, "mse": mse, "mae": mae}

        # log_info = {"delta_pearson": delta_pearson, "mse": mse, "mix_vae_mse": mix_vae_mse.item(), "mix_vae_mmd": mix_vae_mmd.item(), "mix_vae_delta_pearson": mix_vae_delta_pearson.item(), "gt_delta_pearson": gt_delta_pearson.item()}

        

        return {"loss":torch.zeros(1, device=device), "log_info": log_info}

    def compute_metrics(self, src_batch, tgt_batch, tgt_pred):
        delta_pearson = self.metric_utils.delta_pearson(src_batch, tgt_pred, tgt_batch)
        mae = self.metric_utils.mae(tgt_batch, tgt_pred)
        mse = self.metric_utils.mse(tgt_batch, tgt_pred)
        gt_delta_pearson = self.metric_utils.delta_pearson(src_batch, tgt_batch, tgt_batch)
        return delta_pearson, mae, mse, gt_delta_pearson
    
    