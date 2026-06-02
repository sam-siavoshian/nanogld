"""LSTM daily-gold across 3 seeds — addresses reviewer concern about single-seed
DL negative result."""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
DAILY_BPY = 252
BASE_COST_BPS = 2.0
LOOKBACK = 60
HIDDEN = 64
LAYERS = 2
DROPOUT = 0.1
EPOCHS = 30
LR = 1e-3
BATCH = 256
SEEDS = [42, 137, 256]


def _sharpe(r):
    r = r[np.isfinite(r)]
    if len(r) < 2: return 0.0
    mu, sigma = r.mean(), r.std(ddof=1)
    if sigma < 1e-12: return 0.0
    return float(mu / sigma * np.sqrt(DAILY_BPY))


def _pnl(pos, ret, cost_mult=1.0):
    p = np.nan_to_num(pos); r = np.nan_to_num(ret)
    cf = (BASE_COST_BPS * cost_mult) / 10000.0
    prev = np.concatenate([[0.0], p[:-1]])
    return p * r - cf * np.abs(p - prev)


def _rolling_std(x, w):
    n = len(x); out = np.zeros(n)
    cs = np.cumsum(x); cs2 = np.cumsum(x * x)
    for i in range(n):
        a = max(0, i - w + 1); c = i - a + 1
        if c < 2: continue
        m = (cs[i] - (cs[a - 1] if a > 0 else 0)) / c
        m2 = (cs2[i] - (cs2[a - 1] if a > 0 else 0)) / c
        v = max(0, m2 - m * m); out[i] = np.sqrt(v)
    return out


def _rolling_mean(x, w):
    n = len(x); out = np.zeros(n)
    cs = np.cumsum(x)
    for i in range(n):
        a = max(0, i - w + 1); c = i - a + 1
        out[i] = (cs[i] - (cs[a - 1] if a > 0 else 0)) / c
    return out


def _vol_target(daily_ret, tv, w=60, cap=1.0):
    rv = _rolling_std(daily_ret, w)
    raw = tv / (rv * np.sqrt(DAILY_BPY) + 1e-8)
    p = np.clip(raw, 0, cap); p[~np.isfinite(p)] = 0.0
    return np.concatenate([[0.0], p[:-1]])


def engineer_features(close, daily_ret, ts):
    feats = {"ret_1": daily_ret}
    for lag in [5, 10, 20, 60, 120]:
        cs = np.cumsum(daily_ret, dtype=np.float64)
        out = np.zeros_like(daily_ret)
        for i in range(lag, len(daily_ret)):
            a = i - lag
            out[i] = cs[i] - (cs[a - 1] if a > 0 else 0)
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
    X = np.column_stack([feats[k] for k in feats]).astype(np.float32)
    return X


def make_labels(daily_ret, eps=0.001):
    n = len(daily_ret); y = np.zeros(n, dtype=np.int64)
    for i in range(n - 1):
        if daily_ret[i + 1] > eps: y[i] = 2
        elif daily_ret[i + 1] < -eps: y[i] = 0
        else: y[i] = 1
    y[-1] = 1
    return y


def fetch():
    import yfinance as yf
    df = yf.download("GC=F", start="1990-01-01", end="2026-05-25", progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df = df.reset_index()
    df["close"] = df["Close"].astype(float)
    df = df[["Date", "close"]].rename(columns={"Date": "ts"})
    df["log_ret"] = np.log(df["close"] / df["close"].shift(1)).fillna(0.0)
    return df


def split_4fold(n):
    train_len = int(n * 0.55); val_len = int(n * 0.10); test_len = int(n * 0.14)
    step = int(n * 0.07)
    folds = []; s = 0
    for k in range(4):
        ts = s + k * step; te = ts + train_len; ve = te + val_len; tte = ve + test_len
        if tte > n: break
        folds.append((ts, te, ve, tte))
    return folds


def build_seqs(X, y, lookback):
    n = len(X) - lookback
    if n <= 0:
        return np.empty((0, lookback, X.shape[1]), dtype=np.float32), np.empty((0,), dtype=np.int64)
    sx = np.zeros((n, lookback, X.shape[1]), dtype=np.float32)
    sy = np.zeros(n, dtype=np.int64)
    for i in range(n):
        sx[i] = X[i:i + lookback]
        sy[i] = y[i + lookback - 1]
    return sx, sy


class LSTMNet(nn.Module):
    def __init__(self, F_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_size=F_dim, hidden_size=HIDDEN, num_layers=LAYERS, batch_first=True, dropout=DROPOUT)
        self.head = nn.Linear(HIDDEN, 3)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


def run_one_seed(seed, df, X, y, daily_ret, close, folds, device):
    np.random.seed(seed)
    torch.manual_seed(seed)
    F_dim = X.shape[1]
    fold_results = []
    for fi, (a, b, c, d) in enumerate(folds):
        train_X = X[a:b - 1]
        mu = np.nan_to_num(train_X.mean(axis=0))
        sd = np.nan_to_num(train_X.std(axis=0) + 1e-6, nan=1.0)
        X_norm = ((X - mu) / sd).astype(np.float32)
        X_norm = np.nan_to_num(X_norm, nan=0.0, posinf=0.0, neginf=0.0)

        Xtr, ytr = build_seqs(X_norm[a:b - 1], y[a:b - 1], LOOKBACK)
        Xva, yva = build_seqs(X_norm[max(0, b - LOOKBACK):c], y[max(0, b - LOOKBACK):c], LOOKBACK)
        Xte, _ = build_seqs(X_norm[max(0, c - LOOKBACK + 1):d], y[max(0, c - LOOKBACK + 1):d], LOOKBACK)

        model = LSTMNet(F_dim).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
        loss_fn = nn.CrossEntropyLoss()

        Xtr_t = torch.from_numpy(Xtr).to(device); ytr_t = torch.from_numpy(ytr).to(device)
        Xva_t = torch.from_numpy(Xva).to(device); yva_t = torch.from_numpy(yva).to(device)
        Xte_t = torch.from_numpy(Xte).to(device)

        best_acc = 0.0; best_state = None
        n_tr = len(Xtr_t)
        for ep in range(EPOCHS):
            model.train(True)
            perm = torch.randperm(n_tr, device=device)
            for s in range(0, n_tr, BATCH):
                idx = perm[s:s + BATCH]
                logits = model(Xtr_t[idx])
                loss = loss_fn(logits, ytr_t[idx])
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            model.train(False)
            with torch.no_grad():
                va_acc = float((model(Xva_t).argmax(-1) == yva_t).float().mean().cpu())
            if va_acc > best_acc:
                best_acc = va_acc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if best_state is not None:
            model.load_state_dict(best_state)
        model.train(False)
        with torch.no_grad():
            probs = torch.softmax(model(Xte_t), dim=-1).cpu().numpy()
        signal = np.clip(probs[:, 2] - probs[:, 0], -1, 1)
        signal_shift = np.concatenate([[0.0], signal[:-1]])
        test_ret = daily_ret[c:d]
        sig_used = signal_shift[:len(test_ret)] if len(signal_shift) >= len(test_ret) else np.pad(signal_shift, (0, len(test_ret) - len(signal_shift)))
        vt = _vol_target(test_ret, 0.15, 60)
        long_sig = np.where(sig_used > 0, sig_used, 0.0)
        comp = vt * long_sig
        fold_results.append(_sharpe(_pnl(comp, test_ret, 1.0)))
    return fold_results


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}")
    df = fetch()
    daily_ret = df["log_ret"].values.astype(np.float64)
    close = df["close"].values.astype(np.float64)
    X = engineer_features(close, daily_ret, df["ts"])
    y = make_labels(daily_ret, eps=0.001)
    folds = split_4fold(len(df))
    print(f"bars: {len(df)}, folds: {len(folds)}")

    per_seed_per_fold = {}
    for seed in SEEDS:
        print(f"\n=== seed {seed} ===")
        results = run_one_seed(seed, df, X, y, daily_ret, close, folds, device)
        per_seed_per_fold[seed] = results
        print(f"  per-fold combo Sharpe: {[round(x, 3) for x in results]}")

    print()
    print("=== MULTI-SEED AGGREGATE ===")
    all_means = []
    for seed in SEEDS:
        mean_s = np.mean(per_seed_per_fold[seed])
        all_means.append(mean_s)
        print(f"  seed {seed}: per-fold mean = {mean_s:+.4f}")
    print()
    print(f"Across {len(SEEDS)} seeds: mean = {np.mean(all_means):+.4f}  std = {np.std(all_means, ddof=1):.4f}  min = {np.min(all_means):+.4f}  max = {np.max(all_means):+.4f}")


if __name__ == "__main__":
    main()
