"""Inspect V4 fold 0 head weights to confirm tanh saturation diagnosis.

We expect:
- position head (Linear D->1) has a bias OR raw output magnitude > 5 always,
  which saturates tanh negative.
- class head (Linear D->3) has FLAT row dominating other rows.

Reads checkpoint, prints head weight stats.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
ckpt_path = REPO_ROOT / "checkpoints" / "v4" / "fold_0" / "llrd" / "llrd_final.pt"

ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
state = ckpt["model_state"]

print(f"checkpoint keys: {len(state)}")
print(f"final_loss recorded: {ckpt.get('final_loss')}")
print(f"n_steps recorded: {ckpt.get('n_steps')}")
print()

head_keys = [k for k in state.keys() if "head" in k]
print("== HEAD KEYS ==")
for k in head_keys:
    t = state[k]
    print(f"  {k}: shape={tuple(t.shape)} mean={t.mean().item():+.4f} std={t.std().item():.4f} min={t.min().item():+.4f} max={t.max().item():+.4f}")

print()
print("== CLASS HEAD vs POSITION HEAD ==")

cls_w = None
pos_w = None
for k, t in state.items():
    if "cls_head" in k and "weight" in k:
        cls_w = t
        print(f"  cls_head.weight ({tuple(t.shape)}):")
        for c in range(t.shape[0]):
            row = t[c]
            print(f"    class {c}: mean={row.mean().item():+.6f} std={row.std().item():.6f} max_abs={row.abs().max().item():.6f}")
    if "pos_head" in k and "weight" in k:
        pos_w = t
        print(f"  pos_head.weight ({tuple(t.shape)}): mean={t.mean().item():+.6f} std={t.std().item():.6f} max_abs={t.abs().max().item():.6f}")
    if "pos_head" in k and "bias" in k:
        print(f"  pos_head.bias ({tuple(t.shape)}): {[float(x) for x in t.flatten()]}")
    if "cls_head" in k and "bias" in k:
        print(f"  cls_head.bias ({tuple(t.shape)}): {[float(x) for x in t.flatten()]}")

if cls_w is not None and pos_w is not None:
    print()
    print("Class head row norms (L2):")
    for c in range(cls_w.shape[0]):
        print(f"  class {c}: ||w||={cls_w[c].norm().item():.4f}")
    print(f"Position head L2 norm: ||w||={pos_w.norm().item():.4f}")
