# SETUP — secrets and per-machine configuration

This file explains how to populate the secrets nanoGLD needs and where they live. Anything in this doc is per-developer-machine — never committed.

> Spec source: `plan/01-INFRA-AND-SECURITY.md` ("Secrets Management" + "Critical Security Rules").

---

## Secrets layout

All secrets live OUTSIDE the repo, under `~/.config/nanogld/`. The directory is `chmod 700`, files are `chmod 600`.

```
~/.config/nanogld/
  .env.paper      # paper-trading dev environment (used 99% of the time)
  .env.live       # live-trading prod environment (Macbook only, sourced ONLY by launchd in doc 08)
```

**Two-key principle:** the dev shell sources `.env.paper` only. `.env.live` is never sourced manually. Doc 08 wires it into a launchd plist with `EnvironmentVariables` so live keys never touch a dev shell, never appear in shell history, never get printed by mistake.

Verify permissions any time with:

```bash
ls -la ~/.config/nanogld/
# expected: -rw------- on each .env.* file
```

If the perms drift (e.g. you copied the file from somewhere), reset:

```bash
chmod 700 ~/.config/nanogld
chmod 600 ~/.config/nanogld/.env.paper
chmod 600 ~/.config/nanogld/.env.live
```

---

## How to get each key

### Alpaca (paper)
1. Sign up at [alpaca.markets](https://alpaca.markets/) and verify the account.
2. Switch to **paper trading** in the dashboard.
3. Generate API key + secret. Paper keys start with `PK...`.
4. Paste into `~/.config/nanogld/.env.paper`:
   - `ALPACA_API_KEY=PK...`
   - `ALPACA_API_SECRET=...`
   - `ALPACA_BASE_URL=https://paper-api.alpaca.markets`
   - `TRADING_MODE=paper`

Rotate paper keys every 90 days. They are low-stakes but rotation is good muscle memory.

### Alpaca (live)
Only needed when doc 08 ships and you're ready to fund $100. Live keys start with `AK...`. Live base URL is `https://api.alpaca.markets`. NEVER paste live keys into `.env.paper` — wrong file = real money + paper logic = bad day.

### FRED (Federal Reserve Economic Data)
1. Free, instant. Sign up at [fred.stlouisfed.org/docs/api/api_key.html](https://fred.stlouisfed.org/docs/api/api_key.html).
2. Add to `.env.paper`: `FRED_API_KEY=<your-key>`.

### HuggingFace
1. Account at [huggingface.co](https://huggingface.co/).
2. Create a token under Settings → Access Tokens. **Read** scope is enough for downloading Qwen3-Embedding-4B (Apache 2.0).
3. Add to `.env.paper`: `HF_TOKEN=hf_...`.

### Weights & Biases (training tracking)
1. Free at [wandb.ai](https://wandb.ai/).
2. Get the API key from your Settings page.
3. Add to `.env.paper`: `WANDB_API_KEY=...`.

### Google Cloud Platform (BigQuery for GDELT)
**Don't put a service-account JSON in `.env.paper`.** GCP service-account JSON keys are the #1 leaked-credential cause on GitHub.

Use Application Default Credentials (ADC) instead:

```bash
brew install --cask google-cloud-sdk   # if you don't have gcloud
gcloud init                            # pick the project that owns BigQuery
gcloud auth application-default login  # opens a browser, writes ~/.config/gcloud/application_default_credentials.json
```

Doc 02 (data pipeline) reads ADC automatically. There is no env var to set in `.env.paper` for this — `gcloud auth application-default login` is enough.

For CI BigQuery access (only needed if a future workflow runs queries), use Workload Identity Federation (`google-github-actions/auth@v2`). Out of scope for doc 01.

---

## `.env.paper` template

After creating `~/.config/nanogld/`, the file should look exactly like this (with placeholders filled):

```bash
# nanoGLD paper-trading dev environment
# Spec: plan/01-INFRA-AND-SECURITY.md
ALPACA_API_KEY=<FILL_ME_PAPER>
ALPACA_API_SECRET=<FILL_ME_PAPER>
ALPACA_BASE_URL=https://paper-api.alpaca.markets
TRADING_MODE=paper

FRED_API_KEY=<FILL_ME>
HF_TOKEN=<FILL_ME>
WANDB_API_KEY=<FILL_ME>

# GCP: prefer ADC (gcloud auth application-default login).
# Do NOT paste a service-account JSON path here.
```

`.env.live` mirrors the shape but with `TRADING_MODE=live`, `ALPACA_BASE_URL=https://api.alpaca.markets`, and live `AK...` keys. Doc 08 wires it.

---

## How code loads the env

Doc 02+ uses `python-dotenv`:

```python
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path.home() / ".config" / "nanogld" / ".env.paper")
```

`alpaca-py` reads `ALPACA_API_KEY` + `ALPACA_API_SECRET` from `os.environ`. `fredapi.Fred(api_key=os.environ["FRED_API_KEY"])`. `huggingface_hub.login(token=os.environ["HF_TOKEN"])`. `wandb` reads `WANDB_API_KEY` automatically.

---

## What protects you if a secret leaks anyway

1. **gitleaks pre-commit hook** blocks any commit containing leaked credentials before it lands. Verified with a fake key during initial repo setup.
2. **`.gitignore`** blocks `.env`, `.env.*`, `*.key`, `*.pem`, `service-account*.json`, `secrets/`, `credentials.json`, `alpaca-*` from ever being staged accidentally.
3. **GitHub repo gitleaks-action** runs on every push + PR. If anything slips through pre-commit, CI catches it before merge.
4. **Public repo from commit 1.** No private→public migration where old commit history could leak.

If a secret does leak: rotate the key immediately at the issuer (Alpaca / FRED / HF / wandb), then `git filter-repo --invert-paths --path <leaked-file>` + `git push --force` to scrub history. Rotate first, scrub second — the leak is the rotation trigger, scrubbing is hygiene.

---

## Optional: 1Password CLI for `.env.live` (deferred to doc 08)

For live trading, `chmod 600` on `.env.live` is the spec fallback but below industry standard. The upgrade path is to inject live keys via `op://Personal/alpaca-live/...` references read by `op run` inside the launchd cron.

Doc 08 wires this. Doc 01 just reserves the slot.
