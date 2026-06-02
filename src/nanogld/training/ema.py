"""EMA wrapper using torch's AveragedModel + get_ema_multi_avg_fn.

V1 EMA decay = 0.999. Deploy EMA, not raw weights, per V1-SPEC.
"""

from __future__ import annotations

from torch import nn
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn


def make_ema(model: nn.Module, decay: float = 0.999) -> AveragedModel:
    """Wrap a model in an EMA-averaging copy.

    Args:
        model: the live training model.
        decay: EMA decay rate (0.999 V1 default).

    Returns:
        AveragedModel — call `ema.update_parameters(model)` after each
        train step. Use `ema.module` for inference.
    """
    return AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(decay))
