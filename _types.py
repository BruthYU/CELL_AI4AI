import os
from collections.abc import Callable, Sequence
from typing import Any, Union

import numpy as np
import torch

# TODO(michalk8): polish

try:
    from numpy.typing import NDArray

    ArrayLike = Union[NDArray[np.float64], torch.Tensor]
except (ImportError, TypeError):
    ArrayLike = Union[np.ndarray, torch.Tensor]  # type: ignore[misc]
    DTypeLike = np.dtype

ComputationCallback_t = Callable[[dict[str, ArrayLike], dict[str, ArrayLike]], dict[str, Any]]
LoggingCallback_t = Callable[[dict[str, ArrayLike]], dict[str, Any]]

Layers_t = Sequence[dict[str, Any]]
Layers_separate_input_t = dict[str, Layers_t]
PathLike = os.PathLike | str  # type: ignore[type-arg]
