"""Daily-gold SOTA chase v3 — tiny LSTM on Mac mini MPS.

Architecture (matches VLSTM-style spirit but trimmed for 6k bars):
  input: T=60 lookback × F=14 engineered features
  LSTM(input=14, hidden=64, layers=2, dropout=0.1)
  head: Linear(64, 3) → 3-class softmax (DOWN/FLAT/UP next-day)
  loss: focal CE γ=2

Training: per-fold (4-fold WF) on Mac mini MPS, ~10 min/fold.
Inference: continuous signal = (P_up - P_down) ∈ [-1, +1]
Sizing: vol-target × signal_long_only

Target: WF mean Sharpe > 2.40 (VLSTM), ideally > 2.88 (F2F).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

DAILY_BARS_PER_YEAR = 252
BASE_COST_BPS = 2.0
LOOKBACK = 60
HIDDEN = 64
LAYERS = 2
DROPOUT = 0.1
EPOCHS = 30
LR = 1e-3
BATCH = 256


def _sharpe(r: np.ndarray) -> float:
    r = r[np.isfinite(r)]
    if len(r) < 2:
        return 0.0
    mu, sigma = r.mean(), r.std(ddof=1)
    if sigma < 1e-12:
        return 0.0
    return float(mu / sigma * np.sqrt(DAILY_BARS_PER_YEAR))


def _pnl(positions: np.ndarray, daily_ret: np.ndarray, cost_mult: float = 1.0) -> np.ndarray:
    p = np.nan_to_num(positions, nan=0.0)
    r = np.nan_to_num(daily_ret, nan=0.0)
    cost_frac = (BASE_COST_BPS * cost_mult) / 10_000.0
    prev = np.concatenate([[0.0], p[:-1]])
    return p * r - cost_frac * np.abs(p - prev)


def _rolling_std(x: np.ndarray, window: int) -> np.ndarray:
    n = len(x)
    out = np.zeros(n, dtype=np.float64)
    if window <= 1:
        return out
    cs = np.cumsum(x, dtype=np.float64)
    cs2 = np.cumsum(x * x, dtype=np.float64)
    for i in range(n):
        a = max(0, i - window + 1)
        c = i - a + 1
        if c < 2:
            continue
        m = (cs[i] - (cs[a - 1] if a > 0 else 0.0)) / c
        m2 = (cs2[i] - (cs2[a - 1] if a > 0 else 0.0)) / c
        v = max(0.0, m2 - m * m)
        out[i] = float(np.sqrt(v))
    return out


def _rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    n = len(x)
    out = np.zeros(n, dtype=np.float64)
    cs = np.cumsum(x, dtype=np.float64)
    for i in range(n):
        a = max(0, i - window + 1)
        c = i - a + 1
        out[i] = float((cs[i] - (cs[a - 1] if a > 0 else 0.0)) / c)
    return out


def _vol_target(daily_ret: np.ndarray, target_vol: float, window: int = 60, cap: float = 1.0) -> np.ndarray:
    rv = _rolling_std(daily_ret, window=window)
    raw = target_vol / (rv * np.sqrt(DAILY_BARS_PER_YEAR) + 1e-8)
    pos = np.clip(raw, 0.0, cap)
    pos[~np.isfinite(pos)] = 0.0
    return np.concatenate([[0.0], pos[:-1]])


def engineer_features(close: np.ndarray, daily_ret: np.ndarray, ts: pd.Series) -> tuple[np.ndarray, list[str]]:
    feats: dict[str, np.ndarray] = {"ret_1": daily_ret}
    for lag in [5, 10, 20, 60, 120]:
        cs = np.cumsum(daily_ret, dtype=np.float64)
        out = np.zeros_like(daily_ret)
        for i in range(lag, len(daily_ret)):
            a = i - lag
            out[i] = cs[i] - (cs[a - 1] if a > 0 else 0.0)
        feats[f"ret_{lag}"] = out
    for w in [20, 60]:
        feats[f"rv_{w}"] = _rolling_std(daily_ret, w)
    for w in [20, 50, 200]:
        sma = _rolling_mean(close, w)
        feats[f"close_over_sma_{w}"] = close / np.where(sma > 0, sma, 1.0) - 1.0
    diff = np.diff(close, prepend=close[0])
    gain = np.where(diff > 0, diff, 0.0)
    loss = np.where(diff < 0, -diff, 0.0)
    rs = _rolling_mean(gain, 14) / (_rolling_mean(loss, 14) + 1e-12)
    feats["rsi_14"] = 100.0 - 100.0 / (1.0 + rs)
    dow = pd.to_datetime(ts).dt.dayofweek.values.astype(np.float64) / 4.0
    month = pd.to_datetime(ts).dt.month.values.astype(np.float64) / 12.0
    feats["dow"] = dow
    feats["month"] = month
    names = list(feats.keys())
    X = np.column_stack([feats[k] for k in names]).astype(np.float32)
    return X, names


def make_labels(daily_ret: np.ndarray, neutral_eps: float = 0.001) -> np.ndarray:
    n = len(daily_ret)
    y = np.zeros(n, dtype=np.int64)
    for i in range(n - 1):
        r = daily_ret[i + 1]
        if r > neutral_eps:
            y[i] = 2
        elif r < -neutral_eps:
            y[i] = 0
        else:
            y[i] = 1
    y[-1] = 1
    return y


def fetch_gold_daily() -> pd.DataFrame:
    import yfinance as yf  # noqa: PLC0415
    df = yf.download("GC=F", start="1990-01-01", end="2026-05-25", progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df = df.reset_index()
    df["close"] = df["Close"].astype(float)
    df = df[["Date", "close"]].rename(columns={"Date": "ts"})
    df["log_ret"] = np.log(df["close"] / df["close"].shift(1)).fillna(0.0)
    return df


def split_4fold_wf(n: int) -> list[tuple[int, int, int, int]]:
    folds = []
    step = int(n * 0.07)
    train_len = int(n * 0.55)
    val_len = int(n * 0.10)
    test_len = int(n * 0.14)
    for k in range(4):
        ts = k * step
        te = ts + train_len
        ve = te + val_len
        tte = ve + test_len
        if tte > n:
            break
        folds.append((ts, te, ve, tte))
    return folds


def build_sequences(X: np.ndarray, y: np.ndarray, lookback: int) -> tuple[np.ndarray, np.ndarray]:
    """X (N, F), y (N,) → seq X (N-lookback, lookback, F), y (N-lookback,).
    seq[i] uses X[i:i+lookback], label y[i+lookback-1] (the bar predicting the next day).
    """
    n_seqs = len(X) - lookback
    if n_seqs <= 0:
        return np.empty((0, lookback, X.shape[1]), dtype=np.float32), np.empty((0,), dtype=np.int64)
    seq_X = np.zeros((n_seqs, lookback, X.shape[1]), dtype=np.float32)
    seq_y = np.zeros(n_seqs, dtype=np.int64)
    for i in range(n_seqs):
        seq_X[i] = X[i:i + lookback]
        seq_y[i] = y[i + lookback - 1]
    return seq_X, seq_y


def main() -> int:
    import torch  # noqa: PLC0415
    import torch.nn as nn  # noqa: PLC0415

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}")

    df = fetch_gold_daily()
    daily_ret = df["log_ret"].values.astype(np.float64)
    close = df["close"].values.astype(np.float64)
    n = len(df)
    print(f"bars: {n}  range: {df['ts'].iloc[0].date()} → {df['ts'].iloc[-1].date()}")
    X_raw, feat_names = engineer_features(close, daily_ret, df["ts"])
    y_all = make_labels(daily_ret, neutral_eps=0.001)
    F_dim = X_raw.shape[1]
    print(f"features: {F_dim}  label dist: DOWN={(y_all==0).sum()} FLAT={(y_all==1).sum()} UP={(y_all==2).sum()}")

    folds = split_4fold_wf(n)
    print(f"folds: {len(folds)}")
    fold_pure: list[float] = []
    fold_comp: list[float] = []
    fold_comp_2x: list[float] = []

    for fi, (ts_, te, ve, tte) in enumerate(folds):
        print(f"\n=== fold {fi}: train=[{ts_},{te-1}) val=[{te},{ve}) test=[{ve},{tte}) ===")
        # Normalize features using training-only mean/std (PIT)
        train_X = X_raw[ts_:te - 1]
        mu = np.nan_to_num(train_X.mean(axis=0), nan=0.0)
        sd = np.nan_to_num(train_X.std(axis=0) + 1e-8, nan=1.0)
        X_norm = (X_raw - mu) / sd
        X_norm = np.nan_to_num(X_norm, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        y_train_full = y_all
        # Build train sequences from [ts_:te-1] only
        Xtr_seq, ytr = build_sequences(X_norm[ts_:te - 1], y_train_full[ts_:te - 1], LOOKBACK)
        # Build val + test sequences from continuous slice (includes lookback history)
        Xva_seq, yva = build_sequences(X_norm[max(0, te - LOOKBACK):ve], y_train_full[max(0, te - LOOKBACK):ve], LOOKBACK)
        # Test: predict each bar in [ve:tte]; we need lookback prior bars
        Xte_seq, _ = build_sequences(X_norm[max(0, ve - LOOKBACK + 1):tte], y_train_full[max(0, ve - LOOKBACK + 1):tte], LOOKBACK)
        print(f"  train_seqs={len(Xtr_seq)}  val_seqs={len(Xva_seq)}  test_seqs={len(Xte_seq)}")

        # Build model
        torch.manual_seed(42 + fi)
        model = nn.Sequential(
            # LSTM expects (B, T, F)
        )
        class LSTMNet(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lstm = nn.LSTM(input_size=F_dim, hidden_size=HIDDEN, num_layers=LAYERS,
                                   batch_first=True, dropout=DROPOUT)
                self.head = nn.Linear(HIDDEN, 3)
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                out, _ = self.lstm(x)
                return self.head(out[:, -1, :])
        model = LSTMNet().to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
        loss_fn = nn.CrossEntropyLoss()

        # Train
        Xtr_t = torch.from_numpy(Xtr_seq).to(device)
        ytr_t = torch.from_numpy(ytr).to(device)
        Xva_t = torch.from_numpy(Xva_seq).to(device)
        yva_t = torch.from_numpy(yva).to(device)
        Xte_t = torch.from_numpy(Xte_seq).to(device)

        best_val_acc = 0.0
        best_state = None
        n_tr = len(Xtr_t)
        for ep in range(EPOCHS):
            model.train(True)
            perm = torch.randperm(n_tr, device=device)
            ep_loss = 0.0
            for s in range(0, n_tr, BATCH):
                idx = perm[s:s + BATCH]
                logits = model(Xtr_t[idx])
                loss = loss_fn(logits, ytr_t[idx])
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                ep_loss += float(loss.detach().cpu()) * len(idx)
            ep_loss /= max(1, n_tr)
            model.train(False)
            with torch.no_grad():
                va_logits = model(Xva_t)
                va_pred = va_logits.argmax(dim=-1)
                va_acc = float((va_pred == yva_t).float().mean().cpu())
            if va_acc > best_val_acc:
                best_val_acc = va_acc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if ep % 5 == 0 or ep == EPOCHS - 1:
                print(f"  ep {ep:2d} train_loss={ep_loss:.4f} val_acc={va_acc:.4f} best={best_val_acc:.4f}")

        if best_state is not None:
            model.load_state_dict(best_state)
        model.train(False)

        # Predict on test
        with torch.no_grad():
            te_logits = model(Xte_t)
            te_probs = torch.softmax(te_logits, dim=-1).cpu().numpy()
        # Align: Xte_t covers bars [ve:tte] (we built sequences ending at each of these bars)
        signal = (te_probs[:, 2] - te_probs[:, 0]).astype(np.float64)
        signal = np.clip(signal, -1.0, 1.0)
        # PnL: position[i] applied to daily_ret[ve + i] (the model predicted next-day at end of bar ve+i-1; trade on bar ve+i)
        # But our sequence at position 0 represents bar at index (ve - LOOKBACK + 1) + LOOKBACK - 1 = ve, predicting bar ve+1.
        # So signal[i] should be applied to daily_ret[ve + i + 1].
        # Equivalent: shift signal +1.
        signal_shift = np.concatenate([[0.0], signal[:-1]])
        # Test slice for daily_ret
        test_ret = daily_ret[ve:tte]
        # Take only the first len(test_ret) of signal_shift
        signal_used = signal_shift[:len(test_ret)] if len(signal_shift) >= len(test_ret) else np.pad(signal_shift, (0, len(test_ret) - len(signal_shift)))

        # Vol target sizing
        vt = _vol_target(test_ret, target_vol=0.15, window=60, cap=1.0)
        long_signal = np.where(signal_used > 0, signal_used, 0.0)
        comp = vt * long_signal

        s_pure = _sharpe(_pnl(signal_used, test_ret, 1.0))
        s_comp = _sharpe(_pnl(comp, test_ret, 1.0))
        s_comp_2x = _sharpe(_pnl(comp, test_ret, 2.0))
        print(f"  test pure_lstm Sharpe = {s_pure:+.4f}")
        print(f"  test lstm × vt15 long Sharpe = {s_comp:+.4f} (1x), {s_comp_2x:+.4f} (2x)")
        fold_pure.append(s_pure)
        fold_comp.append(s_comp)
        fold_comp_2x.append(s_comp_2x)

    print()
    print("===== WF AGGREGATE =====")
    pm = float(np.mean(fold_pure))
    cm = float(np.mean(fold_comp))
    cm2 = float(np.mean(fold_comp_2x))
    print(f"  pure LSTM signal       : mean={pm:+.4f}  per_fold={['%.3f' % x for x in fold_pure]}")
    print(f"  LSTM × vol_target_15%  : mean={cm:+.4f}  per_fold={['%.3f' % x for x in fold_comp]}  (2x={cm2:+.4f})")
    print()
    print("Benchmarks:")
    print("  VLSTM (Saly-Kaufmann 2026) : +2.4000")
    print("  F2F (Wright 2026)          : +2.8800")
    best = max(pm, cm)
    if best > 2.88:
        print(f"  >>> NEW SOTA ({best:+.4f}, +{best - 2.88:.4f} vs F2F)")
    elif best > 2.40:
        print(f"  beats VLSTM by {best - 2.40:+.4f}; F2F gap {best - 2.88:+.4f}")
    else:
        print(f"  best={best:+.4f}; VLSTM gap {best - 2.40:+.4f}; F2F gap {best - 2.88:+.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
