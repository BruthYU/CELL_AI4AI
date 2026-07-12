import importlib
import logging


class AInterface():
    def __init__(self, conf):
        # self.lightning_model
        self.conf = conf
        self.lightning_model = self.init_lightning_model(self.conf['model_name'])
        self.model = self.instancialize_lightning_model(self.lightning_model)

    def init_lightning_model(self, ae_name):
        return getattr(importlib.import_module(f'AEs.{ae_name}.nemo_model'), f'{ae_name}_Nemo_Model')

    def instancialize_lightning_model(self, model):
        return model(self.conf)