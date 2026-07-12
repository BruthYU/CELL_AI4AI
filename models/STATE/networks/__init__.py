from .base import PerturbationModel
from .decoders import FinetuneVCICountsDecoder
from .decoders_nb import NBDecoder, nb_nll
from .utils import apply_lora, build_mlp, get_activation_class, get_transformer_backbone, get_loss_fn

# Models
from .context_mean import ContextMeanPerturbationModel
from .decoder_only import DecoderOnlyPerturbationModel
from .embed_sum import EmbedSumPerturbationModel
from .perturb_mean import PerturbMeanPerturbationModel
from .old_neural_ot import OldNeuralOTPerturbationModel
from .state_transition import StateTransitionPerturbationModel
from .pseudobulk import PseudobulkPerturbationModel

# Submodules (optional, but good to have accessible)
import logging
logger = logging.getLogger(__name__)

try:
    from . import scgpt
except ImportError as e:
    logger.warning(f"Could not import scgpt submodule: {e}. scGPT functionality will be unavailable.")

try:
    from . import scvi
except ImportError as e:
    logger.warning(f"Could not import scvi submodule: {e}. scVI functionality will be unavailable.")

try:
    from . import cpa
except ImportError as e:
    logger.warning(f"Could not import cpa submodule: {e}. CPA functionality will be unavailable.")

__all__ = [
    "PerturbationModel",
    "PerturbMeanPerturbationModel",
    "ContextMeanPerturbationModel",
    "EmbedSumPerturbationModel",
    "StateTransitionPerturbationModel",
    "OldNeuralOTPerturbationModel",
    "DecoderOnlyPerturbationModel",
    "PseudobulkPerturbationModel",
    "FinetuneVCICountsDecoder",
    "NBDecoder",
    "nb_nll",
    "apply_lora",
    "build_mlp",
    "get_activation_class",
    "get_transformer_backbone",
    "get_loss_fn",
    # Submodules are not exported in __all__ to avoid confusing IDEs if they failed to import
]
