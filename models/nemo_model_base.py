from abc import ABC, abstractmethod
from typing import Any, Callable, Generic, Iterable, Iterator, List, Optional, Sequence, Tuple, TypeVar, Dict, Union

import lightning.pytorch as pl
import torch
from megatron.core import parallel_state
from nemo.lightning import io as nlio
from nemo.lightning.megatron_parallel import DataT, MegatronLossReduction, ReductionT
from nemo.lightning.pytorch.optim import MegatronOptimizerModule
from bionemo.core.model.config import BionemoTrainableModelConfig
from bionemo.llm.api import MegatronLossType, MegatronModelType
# from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer

# --- Shared utility functions ---
# --- Abstract base class ---

class ModelInterfaceBase(
    Generic[MegatronModelType, MegatronLossType],
    pl.LightningModule,
    nlio.IOMixin,
    nlio.ConnectorMixin,
    ABC,
):
    """Base LightningModule for BioNemo Megatron models, split into common and user-customizable parts."""

    def __init__(
        self,
        model_transform: Optional[Callable[[MegatronModelType], MegatronModelType]] = None,
        configure_init_model_parallel: bool = False,
        **model_construct_args,
    ) -> None:
        super().__init__()
        self.model_transform = model_transform
        self.configure_init_model_parallel = configure_init_model_parallel
        self.module: Optional[MegatronModelType] = None

    @abstractmethod
    def configure_model(self) -> None:
        """Instantiate `self.module` using parameters."""
        ...

    def is_on_logging_device(self) -> bool:
        return (
            parallel_state.is_pipeline_last_stage()
            and parallel_state.get_tensor_model_parallel_rank() == 0
        )

    @abstractmethod
    def data_step(self, dataloader_iter: Iterator[DataT]) -> DataT:
        """Collate a micro-batch from the dataloader."""
        ...

    @abstractmethod
    def forward_step(self, batch: Any) -> Any:
        """Perform the core forward pass, returning model outputs or losses."""
        ...

    @abstractmethod
    def training_step(self, batch: Any, batch_idx: Optional[int] = None) -> Any:
        """Perform the training step, returning  outputs."""
        ... 

    @abstractmethod
    def validation_step(self, batch: Any, batch_idx: Optional[int] = None) -> Any:
        """Perform the validation step, returning outputs."""
        ...


    def predict_step(self, batch: Any, batch_idx: Optional[int] = None) -> Any:
        if len(batch) == 0:
            return None
        return self.forward_step(batch)


