# 10 — Infrastructure & Security

## YOU ARE THE DEVOPS / INFRASTRUCTURE AGENT

You own day-1 setup. You bootstrap the repo, install pre-commit hooks (gitleaks BEFORE first commit), set up uv-managed Python env, configure CI, manage secrets. You finish first because every other agent depends on a working repo.

**Read 00-OVERVIEW.md FIRST.**
**Also read 00-OVERVIEW.md "Execution Mode" section before coding.**

### Execution Mode (short — full rules in 00-OVERVIEW.md)

**This doc IS the plan. Do not rewrite it.** Plan to *execute* what is written. If a claim is wrong, AskUserQuestion. Silent scope drift = fired.

- **Research with Nia** before guessing on libs / APIs / papers. Spawn a subagent: `nia papers` / `nia github` / `nia search` / `nia oracle` / `nia packages` / `nia tracer`.
- **Execution skills only:** `/investigate`, `/review`, `/qa`, `/qa-only`, `/cso`, `/benchmark`, `/ship`, `/land-and-deploy`, `/canary`, `/document-release`, `/retro`, `/learn`, `/browse`, `/setup-browser-cookies`. **`/cso` mandatory** since you own secrets + CI + pre-commit hooks.
- **NO planning skills:** `/office-hours`, `/plan-*`, `/autoplan`, `/design-*`, `/devex-review`. Planning is done.
- **Loop:** read doc → Nia for unknowns → code → `/review` → `/qa` → `/cso` → `/ship`.
- **You finish first.** Every other agent depends on a working repo. Bootstrap, then unblock the team.
- **Escalate after 3 failed attempts.** AskUserQuestion. Bad work is worse than no work.

### Files You Create

```
ml-trading/                           # Repo root
├── pyproject.toml                    # uv-managed, full dep list per spec below
├── uv.lock                           # committed
├── .gitignore                        # secrets + ML cruft (per spec below)
├── .pre-commit-config.yaml           # gitleaks v8.24.2 + ruff v0.11+ + hooks v5.0
├── .github/
│   └── workflows/
│       ├── test.yml                  # pytest + ruff + gitleaks on PR
│       └── smoke-test.yml            # monthly cron: full reproduce-fast
├── pyproject.toml                    # canonical 2026 sections (see below)
├── README.md                         # quickstart + reproduce instructions
├── Makefile                          # `make help`, `make data`, `make train`, etc.
└── docs/
    ├── SETUP.md                      # secrets management, key rotation, etc.
    └── REPRODUCE.md                  # step-by-step replication guide

~/.config/nanogld/                 # OS-level (NOT in repo)
├── .env.paper                        # chmod 600
└── .env.live                         # chmod 600 OR 1Password CLI
```

### Files You DO NOT Touch

- Anything in `src/nanogld/` — that's other docs' agents
- Other doc files
- Your job is to provide the SCAFFOLD; other agents build inside it

### Day-1 Critical Path

You unblock everything. Run this sequence:

```bash
# 1. Create repo (public)
gh repo create nanogld --public --description "LLM-augmented gold trader on local hardware"
cd nanogld

# 2. .gitignore FIRST (BEFORE any code, BEFORE any commit)
cat > .gitignore <<'EOF'
# Secrets — never commit
.env
.env.*
*.key
*.pem
service-account*.json
secrets/
alpaca-*

# Python
__pycache__/
*.pyc
.venv/
.pytest_cache/
.mypy_cache/
.ruff_cache/
.coverage
htmlcov/
dist/
build/
*.egg-info/

# Data — never commit raw or processed
data/raw/
data/snapshots/
data/embeddings/

# ML cruft
*.parquet
*.h5
*.ckpt
*.pt
*.safetensors
checkpoints/
wandb/
mlruns/
lightning_logs/
.neptune/
.comet/
outputs/
.hydra/

# Notebooks
.ipynb_checkpoints/

# OS / IDE
.DS_Store
.idea/
.vscode/
EOF
git add .gitignore && git commit -m "chore: gitignore (secrets + data + checkpoints)"

# 3. Pre-commit hooks (gitleaks BEFORE any code goes in)
cat > .pre-commit-config.yaml <<'EOF'
repos:
  - repo: https://github.com/gitleaks/gitleaks
    rev: v8.24.2
    hooks:
      - id: gitleaks
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v5.0.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
        args: ['--maxkb=1024']
      - id: check-merge-conflict
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.11.0
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
EOF

# 4. uv project init + dependencies (PINNED versions)
uv init --python 3.11
uv add 'torch>=2.11.0,<2.12' transformers 'sentence-transformers>=5.0' \
       'huggingface_hub>=1.13' 'accelerate>=1.13,<2.0' \
       alpaca-py 'yfinance==1.3.0' 'fredapi==0.5.2' \
       pandas numpy scikit-learn xgboost google-cloud-bigquery 'wandb>=0.18' \
       feedparser beautifulsoup4 'pandas-ta-classic==0.5.44' \
       'pandas-market-calendars==5.3.2' python-dotenv 'arch>=7.2' \
       'mlx-lm>=0.31.3' 'pyarrow>=16' filelock pandas-stubs \
       schedulefree

uv add --dev pytest ruff mypy pre-commit

# 5. Pin uv.lock + commit
uv lock
git add pyproject.toml uv.lock .pre-commit-config.yaml
git commit -m "chore: scaffold project + pre-commit + pinned deps"

# 6. CRITICAL — verify gitleaks works (try committing a fake key)
echo "ALPACA_API_KEY=PKfake_test_key_pretend_secret_1234567890" > test_secret.txt
git add test_secret.txt
git commit -m "test: should be blocked"   # MUST FAIL
# If it doesn't fail, gitleaks isn't installed. Stop and fix.
git restore --staged test_secret.txt
rm test_secret.txt

# 7. Push initial scaffold
git push -u origin main

# 8. CI workflow (free for public repos, unlimited minutes)
mkdir -p .github/workflows
# (test.yml + smoke-test.yml per templates in this doc)
git add .github/workflows
git commit -m "chore: add CI workflows"
git push
```

### Acceptance Criteria

1. ✅ Public repo created, `.gitignore` is FIRST commit
2. ✅ Pre-commit hooks installed and verified (fake key commit blocks)
3. ✅ `uv sync` produces working environment (run on fresh clone to verify)
4. ✅ `uv.lock` committed
5. ✅ CI green on first PR (test workflow runs ruff + pytest + gitleaks)
6. ✅ Monthly smoke test cron scheduled
7. ✅ `~/.config/nanogld/{.env.paper,.env.live}` chmod 600 with paper keys populated
8. ✅ Other agents can `uv sync` and start coding within 5 min

### Spawn Nia Agents When You Need To

- **gitleaks v8.24.2 current** — verify PR rev tag still works
- **uv version policy** — pin to a specific minor (e.g. `uv>=0.5,<0.6`) or accept latest
- **GitHub Actions free tier** for public repos (unlimited as of 2026)
- **1Password CLI integration with launchd** — for live keys management
- **Modern `pyproject.toml` template** — KDnuggets / Astral docs evolve

### V1 Pinned Versions (DO NOT DOWNGRADE)

```toml
[project]
name = "nanogld"
version = "0.1.0"
requires-python = ">=3.11,<3.13"
dependencies = [
    "torch>=2.11.0,<2.12",        # MPS SDPA fix #174945
    "transformers>=5.7.0",
    "sentence-transformers>=5.0",
    "huggingface_hub>=1.13.0",
    "accelerate>=1.13,<2.0",      # device-agnostic boilerplate (V1)
    "alpaca-py>=0.43,<1.0",
    "yfinance==1.3.0",            # exact pin — fixed April 2026 dividends bug
    "fredapi==0.5.2",
    "pandas>=2.2",
    "numpy>=2.0",
    "scikit-learn>=1.5",
    "xgboost>=2.1",
    "google-cloud-bigquery>=3.40",
    "wandb>=0.18",
    "feedparser>=6.0",
    "beautifulsoup4>=4.12",
    "pandas-ta-classic==0.5.44",  # NOT `ta` (stale, broken API)
    "pandas-market-calendars==5.3.2",
    "python-dotenv>=1.0",
    "arch>=7.2",                  # for stationary block bootstrap
    "mlx-lm>=0.31.3",             # for Qwen3-Embedding-4B 4-bit
    "pyarrow>=16",
    "filelock>=3.13",
    "schedulefree>=1.0",          # Schedule-Free AdamW (Defazio ICLR 2025)
]

[dependency-groups]
dev = [
    "pytest>=8",
    "ruff>=0.11",
    "mypy>=1.10",
    "pre-commit>=4.0",
]

[tool.ruff]
target-version = "py311"
line-length = 100
[tool.ruff.lint]
select = ["E", "F", "W", "I", "B", "C4", "UP", "SIM"]
[tool.ruff.format]
quote-style = "double"

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q --strict-markers"

[tool.mypy]
strict = true
python_version = "3.11"
```

### Critical Security Rules (DO NOT VIOLATE)

1. **`.gitignore` is the FIRST commit.** Before any code, before any test, before any data.
2. **`gitleaks` pre-commit MUST be installed before first commit.** Verify by trying to commit a fake key.
3. **`.env.live` NEVER appears in dev shell.** Sourced ONLY by launchd (doc 08).
4. **GCP service account JSON keys are FORBIDDEN.** Use `gcloud auth application-default login` for dev (ADC), Workload Identity Federation for CI.
5. **`maximum_bytes_billed` cap on every BigQuery query** (doc 02).
6. **Custom BigQuery quota: 1024 GiB/day per user** (Google Console).
7. **Pin every dep version.** No `>=` without upper bound.
8. **`uv.lock` committed**, no manual `requirements.txt`.
9. **Public repo from commit 1.** No private→public migration (history may leak).

### Hand-off Protocol

1. Update STATUS.md with: repo URL, CI status, fresh-clone time, gitleaks verified
2. Notify all other agents that scaffold is ready and they can begin
3. Stand by for dependency requests from other agents (they ASK YOU to add deps; you don't preemptively add)

Now read the implementation specifics.

---

# 10 — Infrastructure & Security

**Status:** ✅ Complete, implementation-ready, Nia-verified
**Last verified:** 2026-04-30

## CRITICAL CORRECTIONS (Nia verification)

**Version bumps:**
- gitleaks `v8.18.0` → ✅ **`v8.24.2`** (Feb 2026)
- pre-commit-hooks `v4.5.0` → ✅ **`v5.0.0`**
- ruff `v0.1.6` (late 2023!) → ✅ **2026 release** (0.8-0.11.x range)
- astral-sh/setup-uv `@v3` → ✅ **`@v8.1.0`**
- `actions/checkout@v4` (current major) — keep
- DROP `black` from dev deps — `ruff format` is drop-in replacement (one less tool)

**Security upgrades:**
- ❌ chmod 600 on `.env.live` for live trading keys → ⚠️ **below industry standard**. Upgrade to **1Password CLI** (`op://Personal/alpaca-live/...`) with `OP_SERVICE_ACCOUNT_TOKEN` injected via launchd, OR macOS Keychain via `security` CLI. chmod 600 is acceptable fallback if you document the tradeoff explicitly.
- ❌ `GOOGLE_APPLICATION_CREDENTIALS=/path/to/sa.json` for GCP → ✅ **ADC** (`gcloud auth application-default login`) for dev, **WIF** (`google-github-actions/auth@v2`) for CI. Service account JSON keys are #1 leaked-credential cause on GitHub.
- ❌ gitleaks alone → ✅ gitleaks as pre-commit blocker + periodic TruffleHog sweeps (defense-in-depth)

**`.gitignore` additions:**
```
# ML-specific (missing from original)
mlruns/
lightning_logs/
.neptune/
.comet/
.ipynb_checkpoints/
.DS_Store
.idea/
.vscode/
*.ckpt
*.pt
*.safetensors
outputs/
.hydra/
.coverage
htmlcov/
dist/
build/
*.egg-info/
```

**Snapshot hashing fix:**
- ❌ `hashlib.sha256(df.to_csv().encode())` → slow, brittle (float repr, locale, NaN encoding) → ✅ **`hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values.tobytes() + str(tuple(df.columns)).encode() + str(df.dtypes.to_dict()).encode())`** — 10-100× faster, deterministic, covers column names + dtypes (closes hash_pandas_object's known column-name bug)

**Log rotation (missing):**
- ❌ No log rotation policy → ✅ **`logging.handlers.RotatingFileHandler(maxBytes=10_000_000, backupCount=14)`** in-process. Avoids macOS newsyslog root-permission trap.

**`pyproject.toml` template (missing concrete content):**
- Add explicit canonical 2026 sections: `[project]`, `[build-system]` (hatchling default), `[tool.ruff]`, `[tool.ruff.lint]`, `[tool.ruff.format]`, `[tool.pytest.ini_options]`, `[tool.mypy]`, `[dependency-groups]` (PEP 735)
**Owner:** samsiavoshian
**Implementation effort:** 0.5 day (day 1)

## Repo Setup (day 1, BEFORE any code)

```bash
# 1. Create the repo
gh repo create nanogld --public --description "LLM-augmented gold trader on local hardware"
cd nanogld

# 2. Init Python project with uv
uv init --python 3.11
uv add torch transformers accelerate alpaca-py yfinance fredapi pandas numpy scikit-learn xgboost \
       google-cloud-bigquery wandb feedparser beautifulsoup4 ta python-dotenv

# 3. Add dev dependencies
uv add --dev pytest ruff black mypy pre-commit gitleaks-hook

# 4. .gitignore (CRITICAL — must be first commit, before anything else)
cat > .gitignore <<'EOF'
# Secrets — never commit
.env
.env.*
*.key
*.pem
service-account*.json
secrets/
alpaca-*

# Python
__pycache__/
*.pyc
.venv/
.pytest_cache/
.mypy_cache/
.ruff_cache/

# Data — never commit raw data
data/raw/
data/snapshots/
data/embeddings/

# Generated
*.parquet
*.h5
checkpoints/
wandb/
```

# 5. Pre-commit hooks
cat > .pre-commit-config.yaml <<'EOF'
repos:
  - repo: https://github.com/gitleaks/gitleaks
    rev: v8.24.2          # bumped from v8.18.0 per Nia
    hooks:
      - id: gitleaks
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v5.0.0           # bumped from v4.5.0 per Nia
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
        args: ['--maxkb=1024']
      - id: check-merge-conflict
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.11.0          # bumped from ancient v0.1.6 (late 2023!)
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format    # ruff format replaces black
EOF

uv run pre-commit install
```

## Secrets Management

**The two-key principle:**

```
~/.config/nanogld/.env        ← paper trading keys (dev machine)
~/.config/nanogld/.env.live   ← live trading keys (Macbook only, in launchd env)
```

`.env` (dev, paper):
```
ALPACA_API_KEY=PK...
ALPACA_API_SECRET=...
TRADING_MODE=paper
FRED_API_KEY=...
GOOGLE_APPLICATION_CREDENTIALS=/path/to/sa.json
HF_TOKEN=hf_...
WANDB_API_KEY=...
```

`.env.live` (production cron only):
```
ALPACA_API_KEY=AK...   ← LIVE keys, different prefix
ALPACA_API_SECRET=...
TRADING_MODE=live
```

The cron loads from `.env.live` exclusively. Dev shell never sources `.env.live`.

**File permissions:**
```bash
chmod 600 ~/.config/nanogld/.env
chmod 600 ~/.config/nanogld/.env.live
```

## Version Pinning

```bash
# Pin everything in uv.lock
uv lock

# Commit uv.lock
git add uv.lock pyproject.toml
git commit -m "chore: pin all dependency versions via uv"
```

Future-you reproducing in 12 months runs `uv sync` and gets identical environments.

## Snapshot Hashing Discipline

Per doc 02 and 02, every dataset/embedding artifact gets a SHA256 hash in its filename. Schema:

```
data/snapshots/v1_<sha256_first_16>.parquet
data/snapshots/v1_<sha256_first_16>_meta.json
data/embeddings/v1_<sha256_first_16>_llama-3.1-8b.parquet
data/anchors/v1.npz
```

Meta JSON example:
```json
{
  "snapshot_version": "v1",
  "snapshot_hash": "abc123...",
  "created_utc": "2026-05-01T14:32:00Z",
  "row_count": 87532,
  "time_range_utc": ["2020-01-01T00:00:00Z", "2024-12-31T23:30:00Z"],
  "data_sources": [
    {"name": "alpaca_bars", "version": "...", "license": "..."},
    {"name": "alpaca_news", ...},
    {"name": "gdelt", ...},
    {"name": "fred_alfred", ...}
  ],
  "schema": [...],
  "git_commit": "abc..."
}
```

## CI (GitHub Actions, free tier)

```yaml
# .github/workflows/test.yml
name: tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
      - run: uv sync
      - run: uv run pytest tests/
      - run: uv run gitleaks detect --no-git --verbose
```

## Quarterly Smoke Test (CI cron)

```yaml
# .github/workflows/smoke-test.yml
name: smoke-test
on:
  schedule:
    - cron: '0 0 1 * *'   # first of every month
jobs:
  smoke:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: uv sync
      - run: uv run python -m nanogld.smoke_test
        env:
          # Read-only test creds, separate from production
          ALPACA_API_KEY: ${{ secrets.ALPACA_TEST_KEY }}
          ...
```

Catches yfinance / Alpaca API breakage before you find out via a failed live cycle.

## Now Resolved (post-deep-dive)

- ✅ CI workflow YAMLs designed (above)
- ✅ Stay 100% local — Modal/cloud rejected per pivot
- ✅ State store: SQLite at `~/.config/nanogld/state.sqlite` (designed in doc 08)
- ✅ Checkpoint backup: weekly cron uploads best checkpoint to HF Hub (private repo, ~30MB compressed)
- ✅ Monitoring: wandb workspace (public for X-thread), `~/Library/Logs/nanogld.log` for ops, weekly manual review

## Repo Bootstrap Order (day 1, exact sequence)

```bash
# 1. Create repo
gh repo create nanogld --public --description "LLM-augmented gold trader on local hardware"
cd nanogld

# 2. .gitignore FIRST (before any code)
cat > .gitignore <<'EOF'
.env
.env.*
*.key
*.pem
service-account*.json
secrets/
alpaca-*
__pycache__/
*.pyc
.venv/
.pytest_cache/
.mypy_cache/
.ruff_cache/
data/raw/
data/snapshots/
data/embeddings/
checkpoints/
wandb/
EOF
git add .gitignore && git commit -m "chore: gitignore (secrets + data + checkpoints)"

# 3. Pre-commit hooks (gitleaks before any code goes in)
cat > .pre-commit-config.yaml <<'EOF'
repos:
  - repo: https://github.com/gitleaks/gitleaks
    rev: v8.18.0
    hooks:
      - id: gitleaks
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
        args: ['--maxkb=1024']
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.6
    hooks:
      - id: ruff
EOF

# 4. uv project init + dependencies
uv init --python 3.11
uv add torch transformers accelerate alpaca-py yfinance fredapi pandas numpy \
       scikit-learn xgboost google-cloud-bigquery wandb feedparser beautifulsoup4 \
       ta python-dotenv pandas-market-calendars arch
uv add --dev pytest ruff pre-commit
uv run pre-commit install

# 5. First test commit (pre-commit must pass)
echo "# nanogld" > README.md
git add README.md pyproject.toml uv.lock .pre-commit-config.yaml
git commit -m "chore: scaffold project + pre-commit"

# 6. Push
git push -u origin main

# 7. Verify gitleaks works (try committing a fake key)
echo "ALPACA_API_KEY=PKfake_test_key_pretend_secret_1234567890" > test_secret.txt
git add test_secret.txt
git commit -m "test: should be blocked"   # ← must fail!
git restore --staged test_secret.txt
rm test_secret.txt
```

If step 7 doesn't fail, gitleaks isn't actually installed — fix before continuing.

## Open Questions

1. Should `data/snapshots/` go to HF Hub for distribution? (Low priority for X-thread era; worth it if asked)
2. Should training checkpoints go to HF Hub? (Yes, after week 4. LoRA adapters are tiny.)
3. Add a `make` target that runs everything (`make all`) or keep granular?
