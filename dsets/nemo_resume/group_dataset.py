import logging
from megatron.core import config
import numpy as np
import torch
from torch.utils.data import Dataset, Subset
import os
import pickle
import lmdb
from pathlib import Path
from tqdm import tqdm
import scipy.sparse as sp
from typing import Literal

class CellGroup_Dataset(Dataset):
    def __init__(
                self, 
                conf, 
                pert_env, 
                control_env, 
                global_keys,
                mode: Literal["train", "val", "test"] = "train"
                ):
        self.conf = conf
        self.mode = mode
        self.pert_env = pert_env
        self.control_env = control_env
        self.global_keys = global_keys


        self.control_pert = self.conf["control_pert"]
        self.cell_sentence_len = self.conf["cell_sentence_len"]

    
        
        # self.current_info = self.conf[self.mode]
        # self.pert_lmdb_path = self.current_info["pert_lmdb_path"]
        # self.control_lmdb_path = self.current_info["control_lmdb_path"]
        with self.pert_env.begin() as txn:
            self.dataset_len = int(txn.get(b"__len__"))
        

        self.rng = np.random.default_rng(self.conf['random_seed'])

    def __len__(self):
        # return self.dataset_len
        if self.mode == "train":
            return int(1e10)
        return self.dataset_len

    def __getitem__(self, idx):
        # if self.pert_env is None or self.control_env is None:
        #     self._open_env()

        idx = idx % self.dataset_len

        with self.pert_env.begin() as pert_txn:
            pert_buf = pert_txn.get(str(idx).encode("utf-8"))
            pert_group = pickle.loads(pert_buf)
            pert_cartesian_key = pert_group["cartesian_key"]
            control_cartesian_key = (pert_cartesian_key[0], pert_cartesian_key[1], self.control_pert)

        with self.control_env.begin() as control_txn:
            control_buf = control_txn.get(str(control_cartesian_key).encode())
            control_group = pickle.loads(control_buf)

        if self.mode == "test":
            # pert_data = pert_group['cell_matrix'].toarray()
            pert_data = self._randomly_select_cells(pert_group['cell_matrix'], self.cell_sentence_len).toarray()
            control_data = self._randomly_select_cells(control_group, len(pert_data)).toarray()
        else:
            pert_data = self._randomly_select_cells(pert_group['cell_matrix'], self.cell_sentence_len).toarray()
            control_data = self._randomly_select_cells(control_group, self.cell_sentence_len).toarray()

        # use index of key in global_keys for one_hot encoding
        plate, cell_line, drugname_drugconc = pert_cartesian_key



        
        sample = {
            "batch_onehot": self.onehot_from_str_np(plate, self.global_keys["plate"]),
            "cell_type_onehot": self.onehot_from_str_np(cell_line, self.global_keys["cell_line"]),
            "pert_onehot": self.onehot_from_str_np(drugname_drugconc, self.global_keys["drugname_drugconc"]),
            "batch_idx": self.idx_from_str_np(plate, self.global_keys["plate"]),
            "cell_type_idx": self.idx_from_str_np(cell_line, self.global_keys["cell_line"]),
            "pert_idx": self.idx_from_str_np(drugname_drugconc, self.global_keys["drugname_drugconc"]),
            "pert_cell_emb": pert_data,
            "ctrl_cell_emb": control_data,
            "pert_cartesian_key": list(pert_cartesian_key)
        }

           
        return sample


    def _randomly_select_cells(self, group, n_cells):
        n_rows = group.shape[0]
        ids = self.rng.choice(n_rows, size=n_cells, replace=True)
        return group[ids]

    def onehot_from_str_np(self, s: str, vocab: list[str]) -> np.ndarray:
        i = vocab.index(s)
        x = np.zeros(len(vocab), dtype=np.int32)
        x[i] = 1
        return x

    def idx_from_str_np(self, s: str, vocab: list[str]) -> int:
        return vocab.index(s)


    # def _open_env(self):
    #     self.pert_env = lmdb.open(self.pert_lmdb_path, readonly=True, lock=False, readahead=False, subdir=True, max_readers=512)
    #     self.control_env = lmdb.open(self.control_lmdb_path, readonly=True, lock=False, readahead=False, subdir=True, max_readers=512)


    # def __getstate__(self):
    #     """
    #     当 Dataset 被 pickle (比如 DataLoader 把它发到 worker 进程）时，
    #     不要把 LMDB env 一起序列化过去。
    #     """
    #     state = self.__dict__.copy()
    #     state["pert_env"] = None
    #     state["control_env"] = None
    #     return state

    # def __setstate__(self, state):
    #     """
    #     worker 进程反序列化时会调用这里。
    #     """
    #     self.__dict__.update(state)