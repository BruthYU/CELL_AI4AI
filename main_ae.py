import os 
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["NCCL_DEBUG"] = "WARN"
# os.environ["DEBUGPY_UNLOAD_EXTENSIONS"] = "1"
# os.environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"
# os.environ["PYDEVD_DISABLE_GC"] = "1"


import time
start = time.time()
from omegaconf import OmegaConf, DictConfig

# 常规launch.json文件会导致加载速度变慢
import nemo.lightning.pytorch.callbacks as nlc
# from lightning.pytorch.trainer import Trainer
from megatron.core.distributed import DistributedDataParallelConfig
from nemo.collections import llm
import nemo.lightning as nl
from dsets import DInterface
from AEs.ae_interface import AInterface


from bionemo.llm.utils.logger_utils import WandbConfig,setup_nemo_lightning_logger 
from bionemo.core.utils.dtypes import PrecisionTypes, get_autocast_dtype
from nemo.lightning import resume
from nemo.lightning.pytorch.optim import MegatronOptimizerModule
from megatron.core.optimizer import OptimizerConfig
# from bionemo.llm.model.lr_scheduler import WarmupAnnealDecayHoldScheduler
from nemo.lightning.pytorch.optim.lr_scheduler import WarmupPolicyScheduler 
from dataclasses import dataclass, field, make_dataclass, is_dataclass
from omegaconf import DictConfig, ListConfig, OmegaConf
from typing import Any, List, Dict, Optional
end = time.time()
print("Import耗时: ", end - start, "秒")

output_dir = './outputs'

os.environ["WANDB_BASE_URL"] = "http://100.104.36.95:8080" 
os.environ["WANDB_API_KEY"] = "local-5eaac1f248b57f61d819e14e5700386b8e90e7d6"


def load_callbacks(conf):
    callback_list = []
    callback_list.append(
        nlc.ModelCheckpoint(
            monitor=conf["experiment"]["monitor"],
            filename='finetune_best-{step:02d}',
            save_top_k=3,
            mode='min',
            save_last=True,
            every_n_epochs=conf["experiment"]["ckpt_freq"],
            dirpath=os.path.join(output_dir, 'checkpoints',conf["experiment"]["group"]),
            always_save_context=True,
        )
        
    )
    return callback_list



def run(conf):
    # CellFlow
    conf = OmegaConf.to_container(conf, resolve=True)   # DictConf → dict, to support Megatron serialisation
    data_interface = DInterface(conf)
    ae_interface = AInterface(conf)

    # Trainer Context
    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        sequence_parallel=False,
        ddp=DistributedDataParallelConfig(
            check_for_nan_in_grad=True,
            overlap_grad_reduce=False,   # 单卡调试禁用 overlap，避免 grad bucket 逻辑触发
            overlap_param_gather=False,
            average_in_collective=True,
            grad_reduce_in_fp32=False,
            use_distributed_optimizer=False,
        ),
        find_unused_parameters=False,   # 避免未使用参数导致 grad bucket 同步断言
        gradient_as_bucket_view=False,  # 关闭 bucket 视图，减少同步假设
        ckpt_include_optimizer=False,
        ckpt_async_save=False,
        ckpt_parallel_load=True,
        num_layers_in_first_pipeline_stage=None
    )
   
    precision_setting=nl.MegatronMixedPrecision(
            precision=conf["experiment"]["precision"],
            params_dtype=get_autocast_dtype(conf["experiment"]["precision"]),
            pipeline_dtype=get_autocast_dtype(conf["experiment"]["precision"]),
            grad_reduce_in_fp32=False,
            autocast_enabled=False,
        )
    
    trainer_config = {
        "devices": conf["experiment"]["num_gpus"],
        "max_epochs": conf["experiment"]["num_epoch"],
        "num_nodes": conf["experiment"]["num_nodes"],
        "strategy": strategy,
        "accelerator": "cuda",
        "callbacks": load_callbacks(conf),
        "check_val_every_n_epoch": conf["experiment"]["check_val_every_n_epoch"],
        "num_sanity_val_steps": conf["experiment"]["num_sanity_val_steps"],
        "enable_checkpointing": True,
        "plugins": precision_setting,
        
        
        # "val_check_interval": conf["experiment"]["val_check_interval"],
        # "limit_val_batches": 1,
        # Don't set logger in training config!
    }
    trainer = nl.Trainer(**trainer_config)


    # Prepare Training
    pl_logger = None
    if conf["experiment"]["use_wandb"]:
        wandb_config = WandbConfig(
            offline=False,
            project='ae_exp',
            entity='yulang',
            group=conf["experiment"]["group"],
            tags=None,
            job_type=None,
            id=None,
            anonymous=False,
            log_model=False
        )
        pl_logger = setup_nemo_lightning_logger(
            root_dir=os.path.join(output_dir, 'wandb',conf["experiment"]["group"]),
            wandb_config=wandb_config,
            name=conf["experiment"]["group"],
            initialize_tensorboard_logger=True
        )

    optimizer = MegatronOptimizerModule(
        config=OptimizerConfig(
            lr=conf["prepare_model"]["learning_rate"],
            optimizer="adam",
            weight_decay=0.01,
            adam_beta1=0.9,
            adam_beta2=0.98,
            clip_grad=1.0,
            adam_eps=1e-8
            # use_distributed_optimizer=False,
        ),
        lr_scheduler=WarmupPolicyScheduler(
            warmup_steps=500,
            max_steps=-1, 
            min_lr=1e-4,
        )
    )

    
    auto_resume = resume.AutoResume(
        resume_from_directory=os.path.join(output_dir, 'checkpoints'),
        resume_if_exists=True,
        resume_ignore_no_checkpoint=True,
        resume_past_end=False,
    )

    # Start Training
    llm.train(
        model=ae_interface.model,
        data=data_interface.datamodule,
        trainer=trainer,
        log=pl_logger,
        optim=optimizer,
        # resume=auto_resume,
    )
    


if __name__ == '__main__':
    conf = OmegaConf.load('./config/ae_plus_exp.yaml')
    # conf = OmegaConf.load('./config/prior.yaml')
    run(conf)