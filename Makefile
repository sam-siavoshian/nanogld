# nanoGLD Makefile — drives common dev tasks via uv.
# Run `make help` to see targets.
# Spec: plan/01-INFRA-AND-SECURITY.md "Files You Create".

.DEFAULT_GOAL := help
.PHONY: help install lock sync upgrade lint format test test-verbose pre-commit clean data train backtest live

help:  ## Show available targets.
	@awk 'BEGIN {FS = ":.*##"; printf "Usage: make \033[36m<target>\033[0m\n\nTargets:\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

install:  ## Install all deps from uv.lock (frozen).
	uv sync --frozen

lock:  ## Re-resolve deps and update uv.lock.
	uv lock

sync:  ## Install deps using current uv.lock (writes if missing).
	uv sync

upgrade:  ## Bump deps to latest allowed by pyproject.toml constraints.
	uv lock --upgrade
	uv sync

lint:  ## Run ruff lint check.
	uv run ruff check .

format:  ## Run ruff formatter (modifies files in place).
	uv run ruff format .

test:  ## Run pytest suite.
	uv run pytest -q

test-verbose:  ## Run pytest with full output.
	uv run pytest -v

pre-commit:  ## Run all pre-commit hooks against all files.
	uv run pre-commit run --all-files

clean:  ## Remove cached build artifacts (safe — does NOT touch data/, checkpoints/, .venv/).
	rm -rf .ruff_cache .pytest_cache .mypy_cache .coverage htmlcov dist build *.egg-info

data:  ## Pull and snapshot raw data (doc 02 implements).
	@echo "data pipeline lives in plan/02-DATA-PIPELINE.md — implement under src/nanogld/data/"

train:  ## Train nanoGLD model (doc 05 implements).
	@echo "training lives in plan/05-MODEL-TRAINING-CALIBRATION.md — implement under src/nanogld/training/"

backtest:  ## Run vectorized backtest (doc 06 implements).
	@echo "backtest lives in plan/06-BACKTEST.md — implement under src/nanogld/backtest/"

live:  ## Run live trading cycle (doc 08 implements).
	@echo "live cycle lives in plan/08-LIVE-TRADING.md — implement under src/nanogld/live/"
