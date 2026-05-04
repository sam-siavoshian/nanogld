# 09 — Live Trading Loop

## YOU ARE THE PRODUCTION ENGINEER AGENT

You own the live trading cycle. You build the launchd cron + Alpaca integration that actually runs the trained model on real money on the Macbook M4 Pro every 30 minutes during market hours.

**Read 00-OVERVIEW.md FIRST.**
**Read 05-MODEL-TRAINING-CALIBRATION.md** + **05-MODEL-TRAINING-CALIBRATION.md** + **07-SIZING-AND-EXITS.md** — you load checkpoints, call sizer.
**Also read 00-OVERVIEW.md "Execution Mode" section before coding.**

### Execution Mode (short — full rules in 00-OVERVIEW.md)

**This doc IS the plan. Do not rewrite it.** Plan to *execute* what is written. If a claim is wrong, AskUserQuestion. Silent scope drift = fired.

- **Research with Nia** before guessing on libs / APIs / papers. Spawn a subagent: `nia papers` / `nia github` / `nia search` / `nia oracle` / `nia packages` / `nia tracer`.
- **Execution skills only:** `/investigate`, `/review`, `/qa`, `/qa-only`, `/cso`, `/benchmark`, `/ship`, `/land-and-deploy`, `/canary`, `/document-release`, `/retro`, `/learn`, `/browse`, `/setup-browser-cookies`. **`/cso` mandatory** before any code touching live Alpaca + secrets.
- **NO planning skills:** `/office-hours`, `/plan-*`, `/autoplan`, `/design-*`, `/devex-review`. Planning is done.
- **Loop:** read doc → Nia for unknowns → code → `/review` → `/qa` → `/cso` → `/ship`.
- **Escalate after 3 failed attempts.** AskUserQuestion. Bad work is worse than no work.

### Files You Create

```
src/nanogld/live/
├── __init__.py
├── cycle.py                # Main 30min cycle: fetch → embed → predict → reconcile → order
├── market_hours.py         # NYSE calendar via pandas_market_calendars
├── lock.py                 # Process lock to prevent overlapping cycles (filelock pkg)
├── state.py                # SQLite local state (cycles, drawdown, drift)
├── drift.py                # Entropy z-score + KL divergence detection
├── alerts.py               # Pushover or Telegram or email
├── alpaca_client.py        # Wrappers around alpaca-py for orders + positions + news
├── reconcile.py            # Pre-cycle: cancel pending orders + compute delta from current
├── feature_window.py       # Build feature vector for current bar (uses doc 04 funcs)
├── runtime.py              # Caffeinate + sleep prevention helpers
└── cli.py                  # `python -m nanogld.live cycle`

# OS-level files you also produce:
~/Library/LaunchAgents/com.samsiavoshian.nanogld.plist  # StartCalendarInterval (NOT StartInterval)
~/.config/nanogld/.env.paper                             # paper keys (chmod 600)
~/.config/nanogld/.env.live                              # live keys (1Password CLI ideally)

tests/
├── test_market_hours.py    # NYSE calendar correctness (DST, holidays, half-days)
├── test_drift.py           # Synthetic drift triggers alert
├── test_reconcile.py       # Pending orders correctly cancelled before delta
└── test_lock.py            # Concurrent cycle gracefully skips
```

### Files You DO NOT Touch

- Anything in `src/nanogld/{data,features,embed,model,training,backtest,sizing}/`
- Other doc files
- The trained model or sizing logic — you USE them, don't change them

### Stable Interface You Consume

```python
# From doc 05/05:
from nanogld.model.tiny_trader import nanoGLDV1
model = nanoGLDV1(...)
model.load_state_dict(torch.load("checkpoints/fold_3_seed_42_ema.pt")['ema_state_dict'])

# From doc 07:
from nanogld.sizing.stage2 import stage2_sizing
from nanogld.sizing.conformal import ConformalSizer

# From doc 03:
from nanogld.embed.live_embed import embed_news_live

# From doc 04:
from nanogld.features.build import build_feature_window_for_bar
```

### Acceptance Criteria

1. ✅ launchd plist installed via `launchctl bootstrap gui/$UID <plist>`
2. ✅ `gitleaks detect --no-git -v` passes (no secrets in repo history)
3. ✅ `pmset -c sleep 0 disablesleep 1` configured (verify with `pmset -g | grep sleep`)
4. ✅ Paper trading runs 5+ days without crash, all cycles logged to wandb
5. ✅ Drift detection fires correctly on synthetic shift (test it!)
6. ✅ Pushover/Telegram alerts received on failure (test it!)
7. ✅ `caffeinate -dims` keeps Macbook awake during market hours
8. ✅ Idempotent recovery: kill cycle mid-run, next cycle reconciles correctly
9. ✅ Pre-cycle order check cancels stale `pending_new` / `partially_filled` orders
10. ✅ Pre-live checklist (in this doc) all green before funding $100

### Spawn Nia Agents When You Need To

- **alpaca-py current API** — `submit_order(order_data=...)` keyword vs positional
- **launchctl bootstrap vs load** — verify modern macOS commands
- **filelock package** vs raw `fcntl.flock` — community-confirmed best for cron lockfiles
- **`pmset` flags** for current macOS Sequoia 15.x / macOS 26.x
- **Pushover free tier limits** (10K/month vs 7.5K — check if changed in May 2026)
- **alpaca-py rate limits** in 2026 — current built-in retry logic

### V1 Critical Decisions (DO NOT REVERT)

1. **`StartCalendarInterval` (array)** NOT `StartInterval=1800` — fires only RTH M-F
2. **`launchctl bootstrap gui/$UID`** NOT `launchctl load` (deprecated since macOS 10.11)
3. **`pmset -c sleep 0 disablesleep 1`** — #1 production risk if skipped
4. **Pre-cycle: check open orders, cancel stale ones** — bug fix from earlier draft
5. **`get_all_positions()` + filter** vs `get_open_position()` (clean no-exception path)
6. **`submit_order(order_data=order)` keyword** required in newer alpaca-py
7. **Two-key separation:** `.env.paper` for dev, `.env.live` ONLY in launchd env (never in dev shell)
8. **1Password CLI for live keys** preferred over chmod 600 file
9. **wandb: ONE run per trading day** with `wandb.init(id=..., resume='allow')` — NOT one per cycle
10. **Reuters RSS dropped** — paywalled 2024. Use Alpaca News (Benzinga) + Yahoo Finance + MarketWatch + SEC EDGAR

### Pre-Live Checklist (Week 3 Day 14)

Before funding $100 and switching to live:

- [ ] 5+ days of clean paper trading, no errors
- [ ] All cycles logged to wandb + SQLite
- [ ] Drift detection fired at least once on synthetic test
- [ ] Pushover alerts working (test alert received on phone)
- [ ] launchd starts on reboot (test by rebooting Macbook)
- [ ] gitleaks pre-commit installed and tested (try fake key, must fail)
- [ ] `.env.live` chmod 600 OR 1Password CLI configured
- [ ] Bank linked to Alpaca, $100 funded
- [ ] Risk register reviewed (max DD, position limit, drawdown CB tested in backtest)
- [ ] Caffeinate verified (Macbook stays awake during market hours)
- [ ] X post drafted: "Today I'm switching to real money. $100. Whatever happens is public."

### Hand-off Protocol

1. Update STATUS.md with: launchd installed, paper-trade duration, live $ funded date
2. After live: daily wandb run summary auto-posted (or manual until comfortable)
3. Document any cycle failures + root cause IN this doc

Now read the implementation specifics.

---

# 09 — Live Trading Loop

**Status:** ✅ Complete, implementation-ready, Nia-verified
**Last verified:** 2026-04-30

## CRITICAL CORRECTIONS (Nia verification)

- ❌ `submit_order(order)` → ✅ `submit_order(order_data=order)` (keyword required in some alpaca-py versions)
- ❌ `get_open_position()` → ✅ wrap in try/except APIError (raises 404 when no position) OR use `get_all_positions()` + filter (single network call, no exception path)
- ❌ `StartInterval=1800` (fires 24/7) → ✅ **`StartCalendarInterval` array** (one entry per weekday × time, fires only during market hours, no need for `is_market_open` check at runtime)
- ❌ `launchctl load` is deprecated → ✅ **`launchctl bootstrap gui/$UID <plist>`** to load, `bootout` to unload, `kickstart -k` to restart
- ❌ Order delta computed from `current_qty` only → ✅ **also check open orders** (`get_orders(status='open')`) — if previous order is `partially_filled`, naive delta double-submits. Cancel-pending or skip-cycle pattern.
- ❌ MISSING macOS sleep prevention → ✅ **`sudo pmset -c sleep 0 disablesleep 1`** + `caffeinate -dims` during market hours. **#1 production risk** — without this, Macbook silently sleeps, missed cycles, missed trades. CRITICAL.
- ❌ Reuters/Bloomberg/FT RSS feeds → ✅ Reuters killed RSS in 2020 + paywalled 2024. **Drop. Use Alpaca News (Benzinga) + free alternatives** (Yahoo Finance, MarketWatch, SEC EDGAR).
- ❌ wandb run-per-cycle → ✅ **one run per trading day** with `wandb.init(id=..., resume='allow')` (273 cycles/month would create unusable workspace)
- ❌ Pushover free 7,500/mo → ✅ actually 10,000/mo (and changing to per-account May 1 2026)
- ❌ Single error path (always alert) → ✅ distinguish transient (network, 5xx) vs permanent (auth fail, account locked) — alert loudly only on permanent
- ⚠️ Consider Telegram bot or ntfy.sh as fallback alerts (more resilient than single Pushover dependency)
- ⚠️ `filelock` package (py-filelock) cleaner than raw `fcntl.flock` — same primitive, better error handling
**Owner:** samsiavoshian
**Implementation effort:** 1.5 days

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│ Macbook M4 Pro (always-on, plugged in, Wi-Fi)                │
│                                                               │
│ launchd cron: every 30min during market hours (M-F)           │
│   ▼                                                           │
│ src/live/cycle.py:                                            │
│   1. Read current market state from Alpaca                    │
│   2. Pull latest price bar (or wait for it to close)          │
│   3. Pull recent news (Alpaca News API + RSS scrape live)     │
│   4. Run Llama-3.1-8B-4bit on news → embeddings (~30 sec)     │
│   5. Run TinyTransformer on (last 64 bars + new bar) → logits│
│   6. Run Stage 2 sizer → desired position                     │
│   7. Compare to current Alpaca position                       │
│   8. Submit order (delta only, idempotent)                    │
│   9. Log everything to wandb + local SQLite                   │
│   10. Send alert if anomaly detected                          │
└──────────────────────────────────────────────────────────────┘
```

## What's Locked

- launchd cron on Macbook (not Mac mini — that's the trainer)
- Alpaca SDK (alpaca-py) for orders + data
- Paper trading for week 1-2, real $100 from week 3
- Two-key separation: paper keys in dev `.env`, live keys ONLY in production cron environment
- gitleaks pre-commit hook installed before first commit
- Idempotent ordering: cycle resumes correctly if interrupted mid-run
- Drift detection: alert if model logits distribution shifts >2σ from training distribution

## Now Designed (post-deep-dive)

All resolved:
- ✅ Full launchd plist with env handling, 30min interval, restart-on-failure with throttle
- ✅ Restart-recovery: cycle queries current Alpaca position before computing delta (self-healing)
- ✅ Rate limit handling: under threshold by 100x, exponential backoff if hit
- ✅ Network failure: try/except, alert, exit with non-zero (cron retries)
- ✅ Order: market order, time_in_force=DAY (day order, no overnight queue)
- ✅ Stop-loss: handled at strategy level via drawdown circuit-breaker (not order-level)
- ✅ Alpaca down: skip cycle, log, alert; cron retries next 30min
- ✅ Drift detection: entropy z-score (>3σ) + KL divergence on argmax distribution
- ✅ Alert: Pushover ($5 one-time, instant phone push)
- ✅ Daily summary: deferred to manual review of wandb workspace

## Secrets Discipline (CRITICAL — real money)

```
# .gitignore
.env
.env.live
secrets/
*.pem
alpaca-keys*

# Pre-commit (Husky or pre-commit framework)
- gitleaks detect --no-git --redact
- check no .env files staged
```

**Two-key strategy:**
- `.env` — paper account keys, used during dev
- `.env.live` — live account keys, ONLY readable by the launchd user, NEVER on dev machine
- Cron loads from `.env.live` exclusively
- Dev shell never has `.env.live` exported

## Skeleton Code

```python
# src/live/cycle.py
import os
from datetime import datetime, timezone
from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

# Load production keys (NEVER on dev machine)
ALPACA_KEY = os.environ['ALPACA_LIVE_KEY']
ALPACA_SECRET = os.environ['ALPACA_LIVE_SECRET']
PAPER = os.environ.get('TRADING_MODE', 'paper') == 'paper'

trading = TradingClient(ALPACA_KEY, ALPACA_SECRET, paper=PAPER)
data = StockHistoricalDataClient(ALPACA_KEY, ALPACA_SECRET)

def run_cycle():
    """One trade cycle. Idempotent: re-running mid-cycle should converge to same state."""
    cycle_id = datetime.now(timezone.utc).isoformat()
    
    try:
        # 1. Get current bar (wait for close if mid-bar)
        latest_bar = fetch_latest_30min_bar('GLD')
        
        # 2. Fetch live news
        news = fetch_recent_news(window_min=30)
        
        # 3. Embed news (load Llama on Macbook for inference; ~5GB unified memory)
        news_embs = embed_news(news)
        
        # 4. Build feature vector for current bar (using last 64 bars)
        history = fetch_last_n_bars('GLD', n=64)
        features = build_feature_window(history, news_embs)
        
        # 5. Predict
        model = load_latest_model_checkpoint()
        with torch.no_grad():
            logits = model(features['numeric'].unsqueeze(0), features['news_raw'].unsqueeze(0))
            probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
        
        # 6. Stage 2 size
        realized_vol = compute_realized_vol(history.close, lookback=480)
        target_size = stage2_sizing(probs, realized_vol)
        
        # 7. Determine current position
        position = trading.get_open_position('GLD') if has_position('GLD') else None
        current_qty = float(position.qty) if position else 0
        target_qty = target_size_to_shares(target_size, capital=100)  # round to nearest share for $100
        delta = target_qty - current_qty
        
        # 8. Submit order if delta meaningful
        if abs(delta) >= 1:
            side = OrderSide.BUY if delta > 0 else OrderSide.SELL
            order = MarketOrderRequest(symbol='GLD', qty=abs(int(delta)), side=side, time_in_force=TimeInForce.DAY)
            trading.submit_order(order)
        
        # 9. Log to wandb + SQLite
        log_cycle(cycle_id, probs, target_size, current_qty, target_qty, delta)
        
        # 10. Drift detection
        check_drift(probs)
        
    except Exception as e:
        log_error(cycle_id, e)
        send_alert(f"Cycle {cycle_id} failed: {e}")
        raise

if __name__ == "__main__":
    run_cycle()
```

## Drift Detection (concept)

```python
def check_drift(probs: np.ndarray, history_window: int = 100):
    """
    Compare current prediction distribution to recent history.
    Alert if entropy collapses (model getting overconfident in one class)
    or shifts dramatically.
    """
    recent = load_recent_probs(history_window)
    
    current_entropy = -np.sum(probs * np.log(probs + 1e-9))
    historical_entropy_mean = recent.entropy.mean()
    historical_entropy_std = recent.entropy.std()
    
    z = (current_entropy - historical_entropy_mean) / (historical_entropy_std + 1e-6)
    
    if abs(z) > 2:
        send_alert(f"Drift: current entropy z={z:.2f} vs history")
```

## launchd plist (skeleton)

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.samsiavoshian.nanogld.cycle</string>
    <key>ProgramArguments</key>
    <array>
        <string>/Users/samsiavoshian/.local/bin/nanogld-cycle</string>
    </array>
    <key>StartInterval</key>
    <integer>1800</integer>  <!-- 30min -->
    <key>EnvironmentVariables</key>
    <dict>
        <key>TRADING_MODE</key>
        <string>live</string>
    </dict>
    <key>StandardOutPath</key>
    <string>/Users/samsiavoshian/Library/Logs/nanogld.log</string>
    <key>StandardErrorPath</key>
    <string>/Users/samsiavoshian/Library/Logs/nanogld-error.log</string>
</dict>
</plist>
```

## Implementation Plan

| Day | Task |
|-----|------|
| 8 AM | Alpaca SDK + `.env` setup, paper account verified |
| 8 PM | Cycle script end-to-end on paper |
| 9 AM | launchd plist + cron schedule |
| 9 PM | 24-hour paper run, verify orders fire correctly |
| 10 (week 2) | Drift detection + alerts |
| 11 (week 2) | Run live for 5 days paper, review |
| 14 (week 3) | Switch to live keys, fund $100 |

## Open Questions for Deep-Dive

1. Alpaca paper has fewer feature than live (no fractional shares, slower data). Plan accordingly.
2. What happens during weekend/holiday — does cron still fire? Add market-open check.
3. Order slippage: market order vs limit at best-ask?
4. How to handle partial fills?
5. Data drift between training distribution and live distribution — when to retrain?
