import inspect
import importlib

import random
import torch
from torch.utils.data import DataLoader
import logging
LOG = logging.getLogger(__name__)


class DInterface():
    def __init__(self, conf):
        # self.nemo_model
        self.conf = conf
        
        self.nemo_datamodule = self.init_nemo_datamodule(self.conf['dataset_name'])
        self.datamodule = self.instancialize_nemo_model(self.nemo_datamodule)

    def init_nemo_datamodule(self, dataset_name):
        return getattr(importlib.import_module(f'dsets.{dataset_name}.nemo_datamodule'), f'{dataset_name}_Nemo_Datamodule')

    def instancialize_nemo_model(self, datamodule):
        return datamodule(self.conf)