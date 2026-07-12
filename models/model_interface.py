import importlib
import logging


class MInterface():
    def __init__(self, conf):
        # self.lightning_model
        self.conf = conf
        self.lightning_model = self.init_lightning_model(self.conf['model_name'])
        self.model = self.instancialize_lightning_model(self.lightning_model)

    def init_lightning_model(self, model_name):
        return getattr(importlib.import_module(f'models.{model_name}.nemo_model'), f'{model_name}_Nemo_Model')

    def instancialize_lightning_model(self, model):
        return model(self.conf)