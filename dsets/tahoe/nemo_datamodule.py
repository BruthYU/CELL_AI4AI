
import random
import torch
from torch.utils.data import DataLoader, Dataset
import logging
from dsets.nemo_data_base import DataInterfaceBase
from nemo.lightning.pytorch.plugins import MegatronDataSampler
from bionemo.llm.utils.datamodule_utils import infer_global_batch_size
import scanpy as sc
import time
import logging
import glob
import re
from functools import partial
from pathlib import Path
from typing import Literal, Set, Dict
import h5py
import torch.distributed as dist
import numpy as np
from tqdm import tqdm
import pickle

from dsets.tahoe.tahoe_dataset import Tahoe_Dataset
import os

logger = logging.getLogger(__name__)
from nemo.utils import AppState
import lmdb

def worker_init_fn(worker_id):
    # 每个 worker 拥有不同 seed（可复现 & 无重复模式）
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset if worker_info is not None else None
    base_seed = getattr(dataset, "random_seed", 42)
    seed = int(base_seed) + worker_id
    random.seed(seed)
    np.random.seed(seed)
    if hasattr(dataset, "set_sampling_seed"):
        dataset.set_sampling_seed(seed)





class tahoe_Nemo_Datamodule(DataInterfaceBase):
    def __init__(self,conf):
        super().__init__()

        # Load and validate configuration
        self.conf = conf
        self.dataset_conf = conf['dataset']
        # Experiment level params
        self.batch_size = self.dataset_conf['batch_size']
        self.num_workers = self.dataset_conf['num_workers']
        self.random_seed = self.dataset_conf['random_seed']
        
        
        self.cell_sentence_len = self.conf["cell_sentence_len"]

        with open(self.dataset_conf["global_keys_path"], "rb") as f:
            self.global_keys = pickle.load(f)

        # lmdb paths
        self.train_lmdb_path = self.dataset_conf["train_lmdb_path"]
        self.val_lmdb_path = self.dataset_conf["val_lmdb_path"]
        self.test_lmdb_path = self.dataset_conf["test_lmdb_path"]
        self.control_lmdb_path = self.dataset_conf["control_lmdb_path"]






    def setup(self, stage: str | None = None):
        """
        Set up training and test datasets.
        """
        # initialize nemo data sampler (must be here)
        self.train_env = lmdb.open(self.train_lmdb_path, readonly=True, lock=False, max_readers=1024)
        self.val_env = lmdb.open(self.val_lmdb_path, readonly=True, lock=False, max_readers=1024)
        self.test_env = lmdb.open(self.test_lmdb_path, readonly=True, lock=False,max_readers=1024)
        self.control_env = lmdb.open(self.control_lmdb_path, readonly=True, lock=False,max_readers=1024)

        
        micro_batch_size=self.dataset_conf['batch_size']
        global_batch_size = infer_global_batch_size(
            micro_batch_size=micro_batch_size,
            num_nodes=self.conf["experiment"]["num_nodes"],
            devices=self.conf["experiment"]["num_gpus"],
            accumulate_grad_batches=self.conf["experiment"]["accumulate_grad_batches"],
            tensor_model_parallel_size=self.conf["experiment"]["tensor_model_parallel_size"],
            pipeline_model_parallel_size=self.conf["experiment"]["pipeline_model_parallel_size"],
        )

        self.data_sampler = MegatronDataSampler(seq_len=self.cell_sentence_len,micro_batch_size=micro_batch_size, 
                                                global_batch_size=global_batch_size, dataloader_type="single",
        )
    


    def train_dataloader(self):
        ds = Tahoe_Dataset(self.dataset_conf, self.train_env, self.control_env, self.global_keys, mode="train")
        prefetch_factor = 2 if int(self.num_workers) > 0 else None
        return DataLoader(
            ds,
            num_workers=self.num_workers,
            prefetch_factor=prefetch_factor,
            persistent_workers=int(self.num_workers) > 0,
            worker_init_fn=worker_init_fn,
        )

    def val_dataloader(self):
        ds = Tahoe_Dataset(self.dataset_conf, self.val_env, self.control_env, self.global_keys, mode="val")
        prefetch_factor = 2 if int(self.num_workers) > 0 else None
        return DataLoader(
            ds,
            num_workers=self.num_workers,
            prefetch_factor=prefetch_factor,
            persistent_workers=int(self.num_workers) > 0,
            worker_init_fn=worker_init_fn,
            # mode="val",
        )

    def test_dataloader(self):
        ds = Tahoe_Dataset(self.dataset_conf, self.test_env, self.control_env, self.global_keys, mode="test")
        prefetch_factor = 2 if int(self.num_workers) > 0 else None
        return DataLoader(
            ds,
            num_workers=self.num_workers,
            prefetch_factor=prefetch_factor,
            persistent_workers=int(self.num_workers) > 0,
            worker_init_fn=worker_init_fn,
        )


    # Helper functions to set up global maps and datasets






