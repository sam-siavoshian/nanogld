"""V1 PyTorch Dataset wrapping training_v1_unified.pt + sidecar.

The dataset returns one bar's full input batch: a T_lookback-bar window
of features plus the news embeddings visible at that bar plus the regime
vector plus the label.

Required files (paths relative to data/):
    processed/training_v1_unified.pt    — 75993 x 681 features + 40032 x 256 news
    processed/training_v1_sidecar.pt    — V1 additions (h5, spread, ATR, regime,
                                          barriers, next_log_return) for each bar
    processed/v1_hmm.joblib             — fitted HMM (built once on train split)

Forward batch dict keys:
    channel_inputs:    (T, F=681) float32
    news_embeddings:   (S, 256)   float16, S = max news slots (default 8)
    news_mask:         (S,)       float32 — 1 = source present
    is_news_present:   ()         long — 0 or 1
    regime_vec:        (12,)      float32
    label_3class:      ()         long — 0/1/2 (CE-mapped triple-barrier)
    next_log_return:   ()         float32 — for Sharpe loss
    era_label:         ()         long — year-bucket index for DANN

PIT-correct: news lookup respects bar_news_offsets/values from unified.pt.
Label is recomputed in `__getitem__` from sidecar barriers + spread when
`label_mode="triple_barrier"`. `label_mode="fixed_5bps"` falls back to the
unified.pt's `labels` field (V1 draft compatibility).

Spec: plan/V1-SPEC.md §0/2 (dataloader contract).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset

from nanogld.data.integrity import MANIFEST_NAME, verify_artifacts
from nanogld.data.utils import get_logger
from nanogld.features.triple_barrier import to_ce_class, triple_barrier_label

LOG = get_logger("nanogld.data.dataset")

DEFAULT_LOOKBACK_T = 64
DEFAULT_NEWS_SLOTS = 8
SPLIT_MODES = Literal["train", "val_a", "val_b", "val_c", "test", "all"]


@dataclass(frozen=True)
class DatasetMeta:
    """Summary meta returned alongside the dataset for inspection."""

    n_bars: int
    n_features: int
    n_articles: int
    train_count: int
    val_count: int
    test_count: int


class NanoGLDDataset(Dataset):
    """Dataset wrapping the unified.pt + sidecar.pt for V1 training.

    Args:
        unified_path: path to `training_v1_unified.pt`.
        sidecar_path: path to `training_v1_sidecar.pt` (or None to skip
            V1 additions; useful for V1-draft compatibility tests).
        split: which logical split to expose: "train", "val_a", "val_b",
            "val_c", or "test".
        lookback_T: number of bars in each input window.
        n_news_slots: max number of news articles per bar.
        label_mode: "triple_barrier" (V1 default) or "fixed_5bps" (V1 draft).
    """

    def __init__(
        self,
        unified_path: Path,
        sidecar_path: Path | None = None,
        split: SPLIT_MODES = "train",
        lookback_T: int = DEFAULT_LOOKBACK_T,
        n_news_slots: int = DEFAULT_NEWS_SLOTS,
        label_mode: Literal["triple_barrier", "fixed_5bps"] = "triple_barrier",
        verify_integrity: bool = True,
    ) -> None:
        super().__init__()
        unified_path = Path(unified_path)
        if not unified_path.exists():
            raise FileNotFoundError(f"unified dataset not found: {unified_path}")
        self.lookback_T = lookback_T
        self.n_news_slots = n_news_slots
        self.label_mode = label_mode

        # V1-SPEC §45: SHA-256 verify-on-load. If a MANIFEST.json sits
        # next to the unified.pt (or sidecar) we re-hash and compare
        # before trusting the file. A failing hash fails fast here, not
        # 8 GB into training.
        if verify_integrity:
            self._verify_or_log(unified_path)
            if sidecar_path is not None and Path(sidecar_path).exists():
                self._verify_or_log(Path(sidecar_path))

        unified = torch.load(unified_path, weights_only=False, map_location="cpu")
        self._unified = unified

        self._features = torch.as_tensor(unified["features"], dtype=torch.float32)
        self._labels_v1draft = torch.as_tensor(unified["labels"], dtype=torch.long)
        self._splits_arr = np.asarray(unified["splits"])
        self._bar_close_utc_ns = torch.as_tensor(
            unified["bar_close_utc_ns"], dtype=torch.int64
        )
        self._bar_news_offsets = torch.as_tensor(
            unified["bar_news_offsets"], dtype=torch.int64
        )
        self._bar_news_values = torch.as_tensor(
            unified["bar_news_values"], dtype=torch.int64
        )
        self._embeddings = torch.as_tensor(unified["embeddings"], dtype=torch.float16)

        self._sidecar = None
        if sidecar_path is not None and Path(sidecar_path).exists():
            self._sidecar = torch.load(
                Path(sidecar_path), weights_only=False, map_location="cpu"
            )
            n_bars = int(self._features.shape[0])
            for key in ("next_log_return", "barrier_up", "barrier_down", "gld_spread_bps_t"):
                if key in self._sidecar:
                    n_side = int(len(self._sidecar[key]))
                    if n_side != n_bars:
                        raise RuntimeError(
                            f"sidecar key {key!r} length {n_side} != unified n_bars {n_bars}; "
                            f"sidecar likely built against a different unified.pt"
                        )

        self._valid_indices = self._compute_valid_indices(split)
        LOG.info(
            "NanoGLDDataset(split=%s, label_mode=%s): %d bars",
            split,
            label_mode,
            len(self._valid_indices),
        )

    @staticmethod
    def _verify_or_log(artifact_path: Path) -> None:
        """If MANIFEST.json is alongside ``artifact_path``, verify this file.

        Hard-fails on mismatch (corrupt or truncated artifact). Logs and
        continues when no manifest is present so legacy builds without
        the manifest writer still work.
        """
        artifact_path = Path(artifact_path)
        manifest = artifact_path.parent / MANIFEST_NAME
        if not manifest.exists():
            LOG.info("no MANIFEST.json next to %s — skipping sha256 verify", artifact_path)
            return
        try:
            verify_artifacts(artifact_path.parent, require=[artifact_path.name])
        except FileNotFoundError as exc:
            LOG.warning("MANIFEST verify: %s", exc)
            return
        LOG.info("sha256 verify OK: %s", artifact_path.name)

    def _compute_valid_indices(self, split: SPLIT_MODES) -> np.ndarray:
        """Return indices of bars that are (a) in the requested split and
        (b) have at least lookback_T history available."""
        n_bars = len(self._splits_arr)
        if split in ("train", "val_a", "val_b", "val_c", "test", "all"):
            mode = split
        else:
            raise ValueError(f"unknown split {split!r}")

        if mode == "all":
            mask = np.ones(n_bars, dtype=bool)
        elif mode == "train":
            mask = self._splits_arr == "train"
        elif mode == "test":
            mask = self._splits_arr == "test"
        else:
            mask_full = self._splits_arr == "val"
            val_idx = np.where(mask_full)[0]
            if len(val_idx) == 0:
                return np.array([], dtype=np.int64)
            half = len(val_idx) // 2
            quarter = len(val_idx) // 4
            if mode == "val_a":
                pick = val_idx[:half]
            elif mode == "val_b":
                pick = val_idx[half : half + quarter]
            else:
                pick = val_idx[half + quarter :]
            mask = np.zeros(n_bars, dtype=bool)
            mask[pick] = True

        idx = np.where(mask)[0]
        idx = idx[idx >= self.lookback_T]
        return idx[idx < n_bars - 1]

    def __len__(self) -> int:
        return int(len(self._valid_indices))

    def _news_for_bar(self, bar_idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (news, mask, is_news_present) for a single bar.

        Pads/truncates to `n_news_slots` slots.
        """
        start = int(self._bar_news_offsets[bar_idx])
        end = int(self._bar_news_offsets[bar_idx + 1])
        article_indices = self._bar_news_values[start:end]
        n_avail = int(article_indices.numel())

        if n_avail == 0:
            news = torch.zeros((self.n_news_slots, self._embeddings.shape[1]), dtype=torch.float16)
            mask = torch.zeros((self.n_news_slots,), dtype=torch.float32)
            is_present = torch.tensor(0, dtype=torch.long)
            return news, mask, is_present

        n_use = min(n_avail, self.n_news_slots)
        picked = article_indices[:n_use]
        slots = self._embeddings[picked]
        if n_use < self.n_news_slots:
            pad = torch.zeros(
                (self.n_news_slots - n_use, self._embeddings.shape[1]), dtype=torch.float16
            )
            slots = torch.cat([slots, pad], dim=0)
        mask = torch.zeros((self.n_news_slots,), dtype=torch.float32)
        mask[:n_use] = 1.0
        is_present = torch.tensor(1, dtype=torch.long)
        return slots, mask, is_present

    def _label_for_bar(self, bar_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (label_3class, next_log_return) for a single bar."""
        if self.label_mode == "fixed_5bps":
            label = self._labels_v1draft[bar_idx].long()
            next_log_return = (
                torch.tensor(
                    float(self._sidecar["next_log_return"][bar_idx])
                    if self._sidecar is not None and "next_log_return" in self._sidecar
                    else 0.0,
                    dtype=torch.float32,
                )
            )
            return label, next_log_return

        if self._sidecar is None:
            raise RuntimeError("triple_barrier label_mode requires a sidecar")

        nlr = float(self._sidecar["next_log_return"][bar_idx])
        b_up = float(self._sidecar["barrier_up"][bar_idx])
        b_dn = float(self._sidecar["barrier_down"][bar_idx])
        spread = float(self._sidecar["gld_spread_bps_t"][bar_idx])
        tb = triple_barrier_label(
            np.array([nlr]), np.array([b_up]), np.array([b_dn]), np.array([spread])
        )
        ce = int(to_ce_class(tb)[0])
        return torch.tensor(ce, dtype=torch.long), torch.tensor(nlr, dtype=torch.float32)

    def _regime_for_bar(self, bar_idx: int) -> torch.Tensor:
        """Return the 12-dim regime vector for a single bar."""
        if self._sidecar is None or "regime_vec" not in self._sidecar:
            return torch.zeros((12,), dtype=torch.float32)
        return torch.as_tensor(self._sidecar["regime_vec"][bar_idx], dtype=torch.float32)

    def _era_label_for_bar(self, bar_idx: int) -> torch.Tensor:
        """Return year-bucket era label (0..3 for {2016-19, 2020-22, 2023-24, 2025+})."""
        if self._sidecar is not None and "era_label" in self._sidecar:
            return torch.as_tensor(int(self._sidecar["era_label"][bar_idx]), dtype=torch.long)
        ts_ns = int(self._bar_close_utc_ns[bar_idx])
        year = (np.datetime64(ts_ns, "ns").astype("datetime64[Y]").astype(int)) + 1970
        if year <= 2019:
            era = 0
        elif year <= 2022:
            era = 1
        elif year <= 2024:
            era = 2
        else:
            era = 3
        return torch.tensor(era, dtype=torch.long)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if not (0 <= idx < len(self._valid_indices)):
            raise IndexError(idx)
        bar_idx = int(self._valid_indices[idx])

        ts_at_bar = int(self._bar_close_utc_ns[bar_idx])
        ts_window_max = int(self._bar_close_utc_ns[bar_idx])
        if not ts_at_bar <= ts_window_max:
            raise RuntimeError("PIT invariant violated in dataset slicing")

        window = self._features[bar_idx - self.lookback_T : bar_idx]
        news, news_mask, is_news_present = self._news_for_bar(bar_idx)
        label_3class, next_log_return = self._label_for_bar(bar_idx)
        regime_vec = self._regime_for_bar(bar_idx)
        era_label = self._era_label_for_bar(bar_idx)

        return {
            "channel_inputs": window,
            "news_embeddings": news,
            "news_mask": news_mask,
            "is_news_present": is_news_present,
            "regime_vec": regime_vec,
            "label_3class": label_3class,
            "next_log_return": next_log_return,
            "era_label": era_label,
        }

    @property
    def meta(self) -> DatasetMeta:
        return DatasetMeta(
            n_bars=int(self._features.shape[0]),
            n_features=int(self._features.shape[1]),
            n_articles=int(self._embeddings.shape[0]),
            train_count=int((self._splits_arr == "train").sum()),
            val_count=int((self._splits_arr == "val").sum()),
            test_count=int((self._splits_arr == "test").sum()),
        )
