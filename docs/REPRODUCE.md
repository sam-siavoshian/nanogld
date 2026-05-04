# REPRODUCE — fresh-clone walkthrough

How to take this repo from `git clone` to "training nanoGLD" on a clean Mac.

> Spec source: `plan/01-INFRA-AND-SECURITY.md` ("Repo Bootstrap Order").

---

## Prerequisites (one-time, per machine)

```bash
# Homebrew (if missing) — see brew.sh
xcode-select --install                                   # Apple developer tools (~5 min)
brew install uv gh gitleaks pre-commit                   # core dev tools
brew install --cask google-cloud-sdk                     # for BigQuery (doc 02)
```

Verify:

```bash
uv --version          # >= 0.10
gh --version          # >= 2.83
gitleaks version      # >= 8.24
pre-commit --version  # >= 4.0
```

Hardware:
- Mac mini M4 16GB OR Macbook M4 Pro 16GB (training + backtest target).
- macOS 14+ (15.x preferred per spec).

Accounts (signup time ~30min total):
- Alpaca (paper + eventually live) — see [SETUP.md](./SETUP.md).
- FRED — free, instant.
- HuggingFace — free, instant.
- Weights & Biases — free, instant.
- GCP (for BigQuery free tier on GDELT) — billing approval may take a day for non-US cards.

---

## Clone and sync

```bash
git clone https://github.com/sam-siavoshian/nanogld.git
cd nanogld
uv python install 3.11
uv sync --frozen        # ~40-60s on M4 cold cache
uv run pre-commit install
make test               # 3 smoke tests pass; CI mirrors this
```

Expected: `make test` exits 0 with `3 passed`. If anything else, stop and read the error — every dep should resolve cleanly because `uv.lock` is committed.

---

## Configure secrets

Follow [SETUP.md](./SETUP.md) to populate `~/.config/nanogld/.env.paper`. Required for any doc beyond 01.

```bash
mkdir -p ~/.config/nanogld
chmod 700 ~/.config/nanogld
# Paste the template from SETUP.md into ~/.config/nanogld/.env.paper, fill values.
chmod 600 ~/.config/nanogld/.env.paper
```

For BigQuery / GDELT (doc 02):

```bash
gcloud init
gcloud auth application-default login
```

---

## Per-doc execution order

Implementation is sequential. Each doc hands off to the next via a stable interface (parquet schema, checkpoint signature, etc.).

| Doc | Owner role | Effort | Output |
|---|---|---|---|
| `plan/01-INFRA-AND-SECURITY.md` | DevOps | 0.5 day (DONE) | working scaffold |
| `plan/02-DATA-PIPELINE.md` | Data engineer | 4-5 days | `data/snapshots/v1_<sha>.parquet` |
| `plan/03-NEWS-EMBEDDING.md` | ML engineer | 1.5d + ~120min precompute | `data/embeddings/v1_<sha>_articles.parquet` |
| `plan/04-FEATURE-ENGINEERING.md` | Feature engineer | 1.5 days | feature DataFrame |
| `plan/05-MODEL-TRAINING-CALIBRATION.md` | ML systems engineer | 3 days | `checkpoints/v1_<sha>.pt` |
| `plan/06-BACKTEST.md` | Quant engineer | 1 day | `reports/v1_<sha>_backtest.md` |
| `plan/07-SIZING-AND-EXITS.md` | Quant risk engineer | 2 days | position-mgmt module |
| `plan/08-LIVE-TRADING.md` | Production engineer | 1.5 days | launchd cron + Alpaca live |

Total wall-time: ~14-16 days end-to-end (sequential).

Each doc's "Hand-off Protocol" section explains how to mark it complete in `plan/STATUS.md` so the next agent unblocks.

---

## What "doc 01 complete" means

Acceptance criteria (verified at the end of step 1):

1. ✅ Public repo at `github.com/sam-siavoshian/nanogld`.
2. ✅ Pre-commit blocks fake keys (gitleaks tested with `ALPACA_API_KEY=PKTEST...` — exit 1).
3. ✅ `uv sync --frozen` produces a working env from a fresh clone.
4. ✅ `uv.lock` committed.
5. ✅ CI green on push: `tests` workflow runs ruff + pytest + gitleaks-action.
6. ✅ Monthly smoke-test cron scheduled (`0 0 1 * *`).
7. ✅ `~/.config/nanogld/.env.paper` chmod 600 with template values.
8. ✅ Fresh-clone-to-`make test`-pass under 5 minutes (target met at ~2min on M4).

If any of the above fails on your machine, the issue is environmental (missing brew tool, network, GCP billing) — fix before starting doc 02.

---

## Useful day-to-day commands

```bash
make help          # see all targets
make install       # uv sync --frozen
make lint          # ruff check
make format        # ruff format (modifies files)
make test          # pytest -q
make pre-commit    # run all hooks on all files
make clean         # remove caches (safe)
```

Troubleshooting:
- `pre-commit` is slow on first run (downloads hook envs). Subsequent runs use cache.
- `uv sync` warns about MPS-only deps on Linux — expected. `mlx-lm` skips on non-Darwin via env marker.
- If `pre-commit run` modifies files: `git add -u && git commit --amend --no-edit` is fine for whitespace-only fixes during dev. Avoid amending after pushing.
