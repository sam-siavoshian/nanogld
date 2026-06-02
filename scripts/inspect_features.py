"""Find which feature index = close price, and what regime_vec dimensions encode."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
unified = torch.load(REPO_ROOT / "data" / "processed" / "training_v1_unified.pt", weights_only=False)
sidecar = torch.load(REPO_ROOT / "data" / "processed" / "training_v1_sidecar_fold_0.pt", weights_only=False)

names = unified["feature_names"]
print("=== feature_names matching close/price/c ===")
for i, n in enumerate(names):
    nl = n.lower()
    if "close" in nl or "_c" == nl[-2:] or "price" in nl:
        print(f"  [{i}] {n}")

print()
print("=== first 30 feature_names ===")
for i in range(min(30, len(names))):
    print(f"  [{i}] {names[i]}")

# regime_vec is 12-dim per V1-SPEC §4. Find which dim is high-vol.
print()
print("=== regime_vec stats (each dim) ===")
rv = sidecar["regime_vec"].numpy() if isinstance(sidecar["regime_vec"], torch.Tensor) else sidecar["regime_vec"]
for d in range(rv.shape[1]):
    col = rv[:, d]
    finite = col[np.isfinite(col)]
    if len(finite) == 0:
        print(f"  dim {d:2d}: all NaN")
        continue
    nuniq = len(np.unique(finite))
    print(f"  dim {d:2d}: mean={finite.mean():+.4f} std={finite.std():.4f} min={finite.min():+.4f} max={finite.max():+.4f} unique={nuniq}")

print()
print("=== bar_close_utc_ns sample ===")
bcn = unified["bar_close_utc_ns"].numpy() if isinstance(unified["bar_close_utc_ns"], torch.Tensor) else unified["bar_close_utc_ns"]
print(f"  first 5: {bcn[:5]}")
print(f"  last 5:  {bcn[-5:]}")
# Convert to readable dates
from datetime import datetime, timezone
print(f"  first date UTC: {datetime.fromtimestamp(bcn[0]/1e9, tz=timezone.utc).isoformat()}")
print(f"  last  date UTC: {datetime.fromtimestamp(bcn[-1]/1e9, tz=timezone.utc).isoformat()}")
# Bar spacing
diffs = np.diff(bcn)
print(f"  bar diff mean (s): {diffs.mean()/1e9:.1f}")
print(f"  bar diff unique seconds first 10: {np.unique(diffs/1e9)[:10]}")
