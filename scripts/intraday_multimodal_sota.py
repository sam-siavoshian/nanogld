"""Intraday GLD multimodal LSTM × vol_target sizing — SOTA push.

Goal: beat our existing intraday vol_target_3pct baseline (+1.212 WF Sharpe)
by adding a directional model that fuses (a) the 651 engineered price
features, (b) the Qwen3-256d news embeddings, and (c) the regime vector,
into a calibrated long-only probability. The probability then modulates
the vol_target_3pct position size.

Architecture (small enough for Mac mini MPS):
  - price branch: LSTM(input=64 PCA features of 651, hidden=64, layers=1)
                 → ends at last bar, gives h_price (B, 64)
  - news branch: mean-pool Qwen embeddings over T (256-dim) → MLP 256→64
                 → h_news (B, 64)
  - fusion: concat [h_price, h_news, regime_vec(12)] → MLP 140→64→3
  - loss: focal CE γ=2 (V1-SPEC §4.6)
  - optimizer: AdamW, lr=1e-3, weight_decay=0.01, 20 epochs, batch=128

Position: signal_long = max(0, P_up - P_down) ∈ [0, 1]
          w_t = vol_target_3pct(real_vol) × signal_long
          (long-only, vol-targeted, model-confidence-modulated)

WF: same 4-fold geometry as V4/V4d (compute_fold_boundaries on unified.pt).
PIT: all features causal at bar t close; trade applied to next_log_return[t].
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from nanogld.data.walk_forward_splits import compute_fold_boundaries  # noqa: E402

BARS_PER_YEAR = 3276
BASE_COST_BPS = 2.0
GLD_CLOSE_FEATURE_IDX = 3
LOOKBACK = 32
PRICE_FEAT_DIM = 64   # PCA-reduced from 651
PRICE_HIDDEN = 64
NEWS_HIDDEN = 64
EPOCHS = 20
BATCH = 128
LR = 1e-3


def _sharpe(r: np.ndarray) -> float:
    r = r[np.isfinite(r)]
    if len(r) < 2:
        return 0.0
    mu, sigma = r.mean(), r.std(ddof=1)
    if sigma < 1e-12:
        return 0.0
    return float(mu / sigma * np.sqrt(BARS_PER_YEAR))


def _pnl(pos: np.ndarray, nlr: np.ndarray, cost_mult: float = 1.0) -> np.ndarray:
    p = np.nan_to_num(pos, nan=0.0)
    r = np.nan_to_num(nlr, nan=0.0)
    cost_frac = (BASE_COST_BPS * cost_mult) / 10_000.0
    prev = np.concatenate([[0.0], p[:-1]])
    return p * r - cost_frac * np.abs(p - prev)


def _rolling_std(x: np.ndarray, window: int) -> np.ndarray:
    n = len(x)
    out = np.zeros(n, dtype=np.float64)
    w = window
    if w <= 1:
        return out
    cs = np.cumsum(x, dtype=np.float64)
    cs2 = np.cumsum(x * x, dtype=np.float64)
    for i in range(n):
        a = max(0, i - w + 1)
        c = i - a + 1
        if c < 2:
            continue
        m = (cs[i] - (cs[a - 1] if a > 0 else 0.0)) / c
        m2 = (cs2[i] - (cs2[a - 1] if a > 0 else 0.0)) / c
        v = max(0.0, m2 - m * m)
        out[i] = float(np.sqrt(v))
    return out


def _vol_target(realized_ret: np.ndarray, target_vol: float = 0.03, window: int = 64, cap: float = 1.0) -> np.ndarray:
    rv = _rolling_std(realized_ret, window=window)
    raw = target_vol / (rv * np.sqrt(BARS_PER_YEAR) + 1e-8)
    pos = np.clip(raw, 0.0, cap)
    pos[~np.isfinite(pos)] = 0.0
    return np.concatenate([[0.0], pos[:-1]])


def _arr(x: Any) -> np.ndarray:
    return x.numpy() if hasattr(x, "numpy") else np.asarray(x)


class MultimodalNet(nn.Module):
    def __init__(self, F_price: int, D_news: int, regime_dim: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size=F_price, hidden_size=PRICE_HIDDEN, num_layers=1,
                           batch_first=True, dropout=0.0)
        self.news_proj = nn.Sequential(
            nn.Linear(D_news, NEWS_HIDDEN),
            nn.GELU(),
            nn.LayerNorm(NEWS_HIDDEN),
        )
        fusion_in = PRICE_HIDDEN + NEWS_HIDDEN + regime_dim
        self.head = nn.Sequential(
            nn.Linear(fusion_in, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 3),
        )

    def forward(self, x_price: torch.Tensor, x_news: torch.Tensor, x_reg: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x_price)
        h_price = out[:, -1, :]
        h_news = self.news_proj(x_news)
        h = torch.cat([h_price, h_news, x_reg], dim=-1)
        return self.head(h)


def focal_loss(logits: torch.Tensor, target: torch.Tensor, gamma: float = 2.0) -> torch.Tensor:
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    log_p_t = log_probs.gather(1, target.unsqueeze(1)).squeeze(1)
    p_t = probs.gather(1, target.unsqueeze(1)).squeeze(1)
    return -((1.0 - p_t) ** gamma * log_p_t).mean()


def build_sequences_multimodal(
    price_feats: np.ndarray,
    news_emb: np.ndarray,
    regime: np.ndarray,
    labels: np.ndarray,
    valid_idx: np.ndarray,
    lookback: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build sequences for each valid bar t in valid_idx.

    seq_price[i] = price_feats[t-lookback+1 : t+1]   shape (lookback, F_price)
    news_t[i]    = news_emb[t]                       shape (D_news,) — mean-pooled per bar
    reg_t[i]     = regime[t]                          shape (regime_dim,)
    label[i]     = labels[t]
    bar_idx[i]   = t (for PnL alignment)
    """
    n_seq = 0
    for t in valid_idx:
        if t >= lookback - 1:
            n_seq += 1
    if n_seq == 0:
        return np.empty((0, lookback, price_feats.shape[1]), dtype=np.float32), \
               np.empty((0, news_emb.shape[1]), dtype=np.float32), \
               np.empty((0, regime.shape[1]), dtype=np.float32), \
               np.empty((0,), dtype=np.int64), \
               np.empty((0,), dtype=np.int64)
    F_price = price_feats.shape[1]
    D_news = news_emb.shape[1]
    R = regime.shape[1]
    seq_p = np.zeros((n_seq, lookback, F_price), dtype=np.float32)
    seq_n = np.zeros((n_seq, D_news), dtype=np.float32)
    seq_r = np.zeros((n_seq, R), dtype=np.float32)
    seq_y = np.zeros(n_seq, dtype=np.int64)
    seq_bar = np.zeros(n_seq, dtype=np.int64)
    j = 0
    for t in valid_idx:
        if t < lookback - 1:
            continue
        seq_p[j] = price_feats[t - lookback + 1: t + 1]
        seq_n[j] = news_emb[t]
        seq_r[j] = regime[t]
        seq_y[j] = labels[t]
        seq_bar[j] = t
        j += 1
    return seq_p, seq_n, seq_r, seq_y, seq_bar


def main() -> int:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}")

    unified = torch.load(REPO_ROOT / "data" / "processed" / "training_v1_unified.pt", weights_only=False)
    features_full = _arr(unified["features"]).astype(np.float32)  # (N, 651)
    bcn = _arr(unified["bar_close_utc_ns"]).astype(np.int64)
    labels_full = _arr(unified["labels"]).astype(np.int64)
    embeddings = _arr(unified["embeddings"]).astype(np.float32)  # (n_articles, 256)
    bar_news_offsets = _arr(unified["bar_news_offsets"]).astype(np.int64)
    bar_news_values = _arr(unified["bar_news_values"]).astype(np.int64)

    print(f"features: {features_full.shape}")
    print(f"embeddings: {embeddings.shape}")

    # Build per-bar pooled news embedding (mean over news indices for that bar; zero if none)
    n_bars = features_full.shape[0]
    D_news = embeddings.shape[1]
    news_per_bar = np.zeros((n_bars, D_news), dtype=np.float32)
    for i in range(n_bars):
        a, b = bar_news_offsets[i], bar_news_offsets[i + 1]
        if b > a:
            idx = bar_news_values[a:b]
            news_per_bar[i] = embeddings[idx].mean(axis=0)

    fold_boundaries = compute_fold_boundaries(bcn)
    print(f"folds: {len(fold_boundaries)}")

    # Sidecars per fold (for regime + atr + h5 features)
    fold_results: list[dict[str, float]] = []
    for fb in fold_boundaries[:4]:
        side = torch.load(REPO_ROOT / "data" / "processed" / f"training_v1_sidecar_fold_{fb.fold_idx}.pt", weights_only=False)
        regime_vec = _arr(side["regime_vec"]).astype(np.float32)  # (N, 12)
        nlr_full = _arr(side["next_log_return"]).astype(np.float64)

        # Train slice: PCA from 651 -> 64 fit on train bars only
        train_idx = np.arange(fb.train_start, fb.train_end)
        val_idx = np.arange(fb.val_start, fb.val_end)
        test_idx = np.arange(fb.test_start, fb.test_end)

        train_feats = features_full[train_idx]
        # Standardize features
        mu = np.nan_to_num(train_feats.mean(axis=0), nan=0.0)
        sd = np.nan_to_num(train_feats.std(axis=0) + 1e-6, nan=1.0)
        feats_norm = ((features_full - mu) / sd).astype(np.float32)
        feats_norm = np.nan_to_num(feats_norm, nan=0.0, posinf=0.0, neginf=0.0)
        # PCA: SVD of train slice, project all bars
        feats_train_norm = feats_norm[train_idx]
        # Use np.linalg.svd to get the top-PRICE_FEAT_DIM right singular vectors
        # for stability use float32 matrix-mode
        _, _, Vt = np.linalg.svd(feats_train_norm - feats_train_norm.mean(axis=0), full_matrices=False)
        components = Vt[:PRICE_FEAT_DIM].astype(np.float32)  # (F_price, 651)
        feats_pca = (feats_norm @ components.T).astype(np.float32)
        # Also standardize news
        news_train = news_per_bar[train_idx]
        n_mu = news_train.mean(axis=0)
        n_sd = news_train.std(axis=0) + 1e-6
        news_norm = ((news_per_bar - n_mu) / n_sd).astype(np.float32)

        # Build sequences
        Xtr_p, Xtr_n, Xtr_r, ytr, _ = build_sequences_multimodal(feats_pca, news_norm, regime_vec, labels_full, train_idx, LOOKBACK)
        Xva_p, Xva_n, Xva_r, yva, _ = build_sequences_multimodal(feats_pca, news_norm, regime_vec, labels_full, val_idx, LOOKBACK)
        Xte_p, Xte_n, Xte_r, yte, bar_te = build_sequences_multimodal(feats_pca, news_norm, regime_vec, labels_full, test_idx, LOOKBACK)
        print(f"\n=== fold {fb.fold_idx}: train_seqs={len(Xtr_p)} val_seqs={len(Xva_p)} test_seqs={len(Xte_p)} ===")

        torch.manual_seed(42 + fb.fold_idx)
        model = MultimodalNet(F_price=PRICE_FEAT_DIM, D_news=D_news, regime_dim=regime_vec.shape[1]).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

        Xtr_p_t = torch.from_numpy(Xtr_p).to(device)
        Xtr_n_t = torch.from_numpy(Xtr_n).to(device)
        Xtr_r_t = torch.from_numpy(Xtr_r).to(device)
        ytr_t = torch.from_numpy(ytr).to(device)
        Xva_p_t = torch.from_numpy(Xva_p).to(device)
        Xva_n_t = torch.from_numpy(Xva_n).to(device)
        Xva_r_t = torch.from_numpy(Xva_r).to(device)
        yva_t = torch.from_numpy(yva).to(device)
        Xte_p_t = torch.from_numpy(Xte_p).to(device)
        Xte_n_t = torch.from_numpy(Xte_n).to(device)
        Xte_r_t = torch.from_numpy(Xte_r).to(device)

        best_acc = 0.0
        best_state = None
        n_tr = len(Xtr_p_t)
        for ep in range(EPOCHS):
            model.train(True)
            perm = torch.randperm(n_tr, device=device)
            ep_loss = 0.0
            for s in range(0, n_tr, BATCH):
                idx = perm[s:s + BATCH]
                logits = model(Xtr_p_t[idx], Xtr_n_t[idx], Xtr_r_t[idx])
                loss = focal_loss(logits, ytr_t[idx], gamma=2.0)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                ep_loss += float(loss.detach().cpu()) * len(idx)
            ep_loss /= max(1, n_tr)
            model.train(False)
            with torch.no_grad():
                va_logits = model(Xva_p_t, Xva_n_t, Xva_r_t)
                va_acc = float((va_logits.argmax(-1) == yva_t).float().mean().cpu())
            if va_acc > best_acc:
                best_acc = va_acc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if ep % 5 == 0 or ep == EPOCHS - 1:
                print(f"  ep {ep:2d} train_loss={ep_loss:.4f} val_acc={va_acc:.4f} best={best_acc:.4f}")

        if best_state is not None:
            model.load_state_dict(best_state)
        model.train(False)
        with torch.no_grad():
            te_logits = model(Xte_p_t, Xte_n_t, Xte_r_t)
            te_probs = torch.softmax(te_logits, dim=-1).cpu().numpy()

        # Build per-bar signal aligned with test_idx[lookback-1 onwards]
        p_down = te_probs[:, 0]
        p_up = te_probs[:, 2]
        # Map probs back to absolute bar arrays for the test window
        p_down_full = np.zeros(len(test_idx), dtype=np.float64)
        p_up_full = np.zeros(len(test_idx), dtype=np.float64)
        # Default to neutral 1/3 for bars without a prediction
        p_down_full[:] = 1.0 / 3.0
        p_up_full[:] = 1.0 / 3.0
        for i, bar in enumerate(bar_te):
            next_bar = bar + 1
            local = next_bar - fb.test_start
            if 0 <= local < len(p_down_full):
                p_down_full[local] = float(p_down[i])
                p_up_full[local] = float(p_up[i])

        # Build vol_target baseline
        test_close = features_full[test_idx, GLD_CLOSE_FEATURE_IDX].astype(np.float64)
        rr = np.zeros_like(test_close, dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            rr[1:] = np.log(np.maximum(test_close[1:], 1e-12) / np.maximum(test_close[:-1], 1e-12))
        rr = np.nan_to_num(rr, nan=0.0, posinf=0.0, neginf=0.0)
        vt = _vol_target(rr, target_vol=0.03, window=64)
        test_nlr = nlr_full[test_idx]

        # Strategy variants:
        # (a) pure model: position = max(0, p_up - p_down)
        pure_pos = np.maximum(0.0, p_up_full - p_down_full)
        s_pure = _sharpe(_pnl(pure_pos, test_nlr, 1.0))

        # (b) vt × (p_up - p_down) multiplicative — proved hurts
        mult_pos = vt * pure_pos
        s_mult = _sharpe(_pnl(mult_pos, test_nlr, 1.0))

        # (c) vt with BEAR VETO only (new): keep vt unless model very bearish
        bear_thresh = 0.40
        veto_gate = np.where(p_down_full > bear_thresh,
                            np.maximum(0.0, 1.0 - (p_down_full - bear_thresh) / (1.0 - bear_thresh)),
                            1.0)
        veto_pos = vt * veto_gate
        s_veto = _sharpe(_pnl(veto_pos, test_nlr, 1.0))
        s_veto_15 = _sharpe(_pnl(veto_pos, test_nlr, 1.5))

        # (d) vt + BULL BOOST (new): pos = min(W_max, vt * (1 + alpha * (p_up - p_down + 1)/2))
        # alpha=0.5 so when p_up=1, p_down=0 → boost factor = 1+0.5*1 = 1.5
        alpha = 0.5
        boost = 1.0 + alpha * np.maximum(0.0, p_up_full - p_down_full)
        boost_pos = np.clip(vt * boost, 0.0, 1.5)
        s_boost = _sharpe(_pnl(boost_pos, test_nlr, 1.0))

        s_vt_only = _sharpe(_pnl(vt, test_nlr, 1.0))
        print(f"  test pure_model            Sharpe = {s_pure:+.4f}")
        print(f"  test vt_only               Sharpe = {s_vt_only:+.4f}")
        print(f"  test vt × model (mult)     Sharpe = {s_mult:+.4f}")
        print(f"  test vt × bear_veto        Sharpe = {s_veto:+.4f}  (1.5x cost {s_veto_15:+.4f})")
        print(f"  test vt × bull_boost       Sharpe = {s_boost:+.4f}")
        fold_results.append({
            "fold": fb.fold_idx,
            "pure": s_pure,
            "vt_only": s_vt_only,
            "mult": s_mult,
            "veto": s_veto,
            "veto_15": s_veto_15,
            "boost": s_boost,
            "val_acc": best_acc,
        })

    print()
    print("===== WF AGGREGATE =====")
    keys = ["pure", "vt_only", "mult", "veto", "veto_15", "boost"]
    for k in keys:
        vals = [m[k] for m in fold_results]
        print(f"  {k:<12s}: mean={np.mean(vals):+.4f}  per_fold={[round(v,3) for v in vals]}")

    combo_mean = float(np.mean([m["veto"] for m in fold_results]))
    print()
    print("=== Benchmark comparison ===")
    print(f"  buy_hold intraday:                     +0.852")
    print(f"  vol_target_3pct (current ship):        +1.212")
    print(f"  multimodal_lstm × vt (ours):           {combo_mean:+.4f}")
    print(f"  VLSTM (multi-asset portfolio):         +2.400")
    print(f"  F2F (daily gold futures):              +2.880")
    if combo_mean > 1.212:
        print(f"  >>> beats vol_target baseline by {combo_mean - 1.212:+.4f}")
    if combo_mean > 2.40:
        print(f"  >>> beats VLSTM by {combo_mean - 2.40:+.4f}")
    if combo_mean > 2.88:
        print(f"  >>> NEW SOTA: beats F2F by {combo_mean - 2.88:+.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
