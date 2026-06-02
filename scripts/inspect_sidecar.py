"""Inspect what fields live in the fold 0 sidecar + unified.pt.

Used to figure out what arrays we need to thread into production.py to
make the gao_2014 / xgboost / ma_cross baselines actually run on real
data instead of zeros.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
unified = torch.load(REPO_ROOT / "data" / "processed" / "training_v1_unified.pt", weights_only=False)
sidecar = torch.load(REPO_ROOT / "data" / "processed" / "training_v1_sidecar_fold_0.pt", weights_only=False)

print("=== UNIFIED.PT keys ===")
for k, v in unified.items():
    if isinstance(v, (torch.Tensor, np.ndarray)):
        print(f"  {k}: shape={tuple(v.shape)} dtype={v.dtype}")
    elif isinstance(v, (list, tuple)):
        print(f"  {k}: len={len(v)} (list/tuple)")
    elif isinstance(v, dict):
        sub = list(v.keys())[:5]
        print(f"  {k}: dict, keys={sub}...")
    else:
        s = str(v)[:80]
        print(f"  {k}: {type(v).__name__} = {s}")

print()
print("=== SIDECAR.PT keys ===")
for k, v in sidecar.items():
    if isinstance(v, (torch.Tensor, np.ndarray)):
        print(f"  {k}: shape={tuple(v.shape)} dtype={v.dtype}")
    elif isinstance(v, (list, tuple)):
        print(f"  {k}: len={len(v)} (list/tuple)")
    elif isinstance(v, dict):
        sub = list(v.keys())[:5]
        print(f"  {k}: dict, keys={sub}...")
    else:
        s = str(v)[:80]
        print(f"  {k}: {type(v).__name__} = {s}")
