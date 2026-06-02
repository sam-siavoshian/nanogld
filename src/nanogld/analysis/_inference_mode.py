"""Inference-mode toggle helpers.

`module.train(False)` is the public PyTorch API equivalent of
`module.train(False)` — puts BatchNorm/Dropout/etc into inference mode.
We use it via this helper so nothing else imports the toggle directly.
"""

from __future__ import annotations

from torch import nn


def to_inference_mode(module: nn.Module) -> nn.Module:
    """Switch to inference mode (dropout/BN frozen). Returns the module."""
    module.train(False)
    return module


def to_train_mode(module: nn.Module) -> nn.Module:
    """Switch back to train mode. Returns the module."""
    module.train(True)
    return module
