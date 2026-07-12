import argparse
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
from models import MInterface


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

# os.environ["WANDB_BASE_URL"] = "http://100.104.36.95:8080" 
# os.environ["WANDB_API_KEY"] = "local-5eaac1f248b57f61d819e14e5700386b8e90e7d6"


# def apply_pbmc_env_overrides(conf):
#     overrides = [
#         ("PBMC_EXPERIMENT_GROUP", "experiment.group", str),
#         ("PBMC_TRAIN_NUM_GPUS", "experiment.num_gpus", int),
#         ("PBMC_NUM_EPOCH", "experiment.num_epoch", int),
#         ("PBMC_CKPT_FREQ", "experiment.ckpt_freq", int),
#         ("PBMC_CHECK_VAL_EVERY_N_EPOCH", "experiment.check_val_every_n_epoch", int),
#         ("PBMC_NUM_SANITY_VAL_STEPS", "experiment.num_sanity_val_steps", int),
#         ("PBMC_BATCH_SIZE", "dataset.batch_size", int),
#         ("PBMC_NUM_WORKERS", "dataset.num_workers", int),
#         ("PBMC_MEGATRON_CKPT", "prepare_model.megatron_ckpt", str),
#     ]
#     applied = []
#     for env_name, key_path, caster in overrides:
#         raw_value = os.environ.get(env_name)
#         if raw_value is None or raw_value == "":
#             continue
#         value = caster(raw_value)
#         OmegaConf.update(conf, key_path, value, merge=False)
#         applied.append(f"{key_path}={value}")
#     if applied:
#         print("PBMC config overrides:", ", ".join(applied))


def get_global_rank():
    for env_name in ("RANK", "SLURM_PROCID", "OMPI_COMM_WORLD_RANK"):
        rank = os.environ.get(env_name)
        if rank is not None:
            return int(rank)
    return 0


def sanitize_path_name(name):
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in str(name))


def resolve_run_timestamp(group):
    env_timestamp = os.environ.get("CELLFLOW_RUN_TIMESTAMP")
    if env_timestamp:
        return env_timestamp

    timestamp_dir = os.path.join(output_dir, "checkpoints", ".run_timestamps")
    os.makedirs(timestamp_dir, exist_ok=True)
    timestamp_file = os.path.join(timestamp_dir, f"{sanitize_path_name(group)}.timestamp")
    rank = get_global_rank()

    if rank == 0:
        run_timestamp = time.strftime("%Y%m%d_%H%M%S")
        with open(timestamp_file, "w", encoding="utf-8") as f:
            f.write(run_timestamp)
        return run_timestamp

    for _ in range(120):
        if os.path.exists(timestamp_file) and os.path.getmtime(timestamp_file) >= start:
            with open(timestamp_file, "r", encoding="utf-8") as f:
                run_timestamp = f.read().strip()
            if run_timestamp:
                return run_timestamp
        time.sleep(1)

    raise RuntimeError(f"Timed out waiting for run timestamp file: {timestamp_file}")


def build_run_dir_name(group, run_timestamp):
    return f"{sanitize_path_name(group)}_{run_timestamp}"


def load_callbacks(conf, run_timestamp):
    callback_list = []
    checkpoint_dir = os.path.join(
        output_dir,
        'checkpoints',
        build_run_dir_name(conf["experiment"]["group"], run_timestamp),
    )
    callback_list.append(
        nlc.ModelCheckpoint(
            monitor=conf["experiment"]["monitor"],
            filename='finetune_best-{step:02d}',
            save_top_k=3,
            mode='min',
            save_last=True,
            every_n_epochs=conf["experiment"]["ckpt_freq"],
            dirpath=checkpoint_dir,
            always_save_context=True,
        )
        
    )
    return callback_list



def run(conf):
    # CellFlow
    conf = OmegaConf.to_container(conf, resolve=True)   # DictConf → dict, to support Megatron serialisation
    run_timestamp = resolve_run_timestamp(conf["experiment"]["group"])
    os.environ["CELLFLOW_RUN_TIMESTAMP"] = run_timestamp
    checkpoint_dir = os.path.join(
        output_dir,
        'checkpoints',
        build_run_dir_name(conf["experiment"]["group"], run_timestamp),
    )
    print(f"Checkpoint dir: {checkpoint_dir}")
    data_interface = DInterface(conf)
    model_interface = MInterface(conf)

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
        "callbacks": load_callbacks(conf, run_timestamp),
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
            offline=True,
            project='nemo_cellflow',
            entity='lyu_ecnu-east-china-normal-university',
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
            warmup_steps=200,
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
        model=model_interface.model,
        data=data_interface.datamodule,
        trainer=trainer,
        log=pl_logger,
        optim=optimizer,
        # resume=auto_resume,
    )
    


def parse_args():
    parser = argparse.ArgumentParser(description="Run NeMo CellFlow training.")
    parser.add_argument("--config", required=True, help="Path to the training YAML config.")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print(f"Using config: {args.config}")
    conf = OmegaConf.load(args.config)
    run(conf)
