"""
Defines the abstract base class for data interfaces, separating out all common Nemo/Megatron integration logic.
Users should subclass this and implement the four abstract methods.
"""
from abc import ABC, abstractmethod
from bionemo.llm.data.datamodule import MegatronDataModule

class DataInterfaceBase(MegatronDataModule, ABC):
    """
    Abstract base for Nemo data modules.
    Subclasses must implement setup, train_dataloader, val_dataloader, test_dataloader.
    """
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__()

    @abstractmethod
    def setup(self, stage: str = None) -> None:
        """
        Prepare datasets and clusters. Called by Lightning.
        """
        pass

    @abstractmethod
    def train_dataloader(self):
        """Return training DataLoader"""
        pass

    @abstractmethod
    def val_dataloader(self):
        """Return validation DataLoader"""
        pass

    # @abstractmethod
    # def test_dataloader(self):
    #     """Return test DataLoader"""
    #     pass

    