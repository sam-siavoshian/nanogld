"""Data-pipeline CLI — `python -m nanogld.data build` and friends.

Sub-commands:
  build [--skip-keyed]      pull all sources, join, snapshot+meta
  pull <name>               pull a single source (one of: calendar, cot, wgc,
                            gpr, brent_wti, alpaca_bars, alpaca_etfs,
                            alpaca_news, fred, gdelt, gdelt_materialize, fnspid,
                            kitco, investing, bullionvault, central_bank,
                            reddit, kaggle)
  join                      re-run join from existing data/raw/ parquets
  snapshot                  alias for join + write snapshot

Loads ~/.config/nanogld/.env.paper via python-dotenv. Idempotent: each pull
checks if its output parquet already exists and skips unless --force is set.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

from nanogld.data import (
    alpaca_bars,
    alpaca_etfs,
    alpaca_news,
    calendar_events,
    cot,
    fred,
    gdelt,
    gpr,
    news_alpha_vantage,
    news_bullionvault,
    news_central_bank,
    news_fnspid,
    news_investing,
    news_kaggle,
    news_kitco,
    news_multisource,
    news_polygon,
    news_reddit,
    polygon_bars,
    polygon_etfs,
    wgc,
    yfinance_helpers,
)
from nanogld.data import (
    join as joiner,
)
from nanogld.data import (
    snapshot as snap,
)
from nanogld.data.utils import get_logger, raw_dir, snapshots_dir

LOG = get_logger("nanogld.data.cli")

ENV_PATH = Path.home() / ".config" / "nanogld" / ".env.paper"

# (source_name, write_fn, output_parquet_filename) — used by `pull` + idempotency.
SOURCES: dict[str, tuple[callable, str | None]] = {
    # No-key / key-less first
    "calendar": (calendar_events.write_calendar_parquet, "calendar_events_v1.parquet"),
    "cot": (cot.write_cot_parquet, "cftc_cot_gold_weekly.parquet"),
    "wgc": (wgc.write_wgc_parquet, "wgc_central_bank_monthly.parquet"),
    "gpr": (gpr.write_gpr_parquet, "gpr_combined.parquet"),
    "brent_wti": (yfinance_helpers.write_yfinance_parquet, None),  # 2 files
    # Key-required
    "alpaca_bars": (alpaca_bars.write_gld_parquet, "alpaca_bars_GLD_30min.parquet"),
    "alpaca_etfs": (alpaca_etfs.write_etf_parquets, None),  # 9 files
    "alpaca_news": (alpaca_news.write_news_parquet, "alpaca_news_GLD.parquet"),
    "fred": (fred.write_all_parquets, None),  # 35 files
    # GDELT — owner runs materialize separately
    "gdelt_materialize": (gdelt.materialize, None),
    "gdelt": (gdelt.write_gdelt_parquet, "gdelt_gkg_5y.parquet"),
    # News scrapers
    "fnspid": (news_fnspid.write_fnspid_parquet, "fnspid_gold_relevant.parquet"),
    "kitco": (news_kitco.write_kitco_parquet, "kitco_news.parquet"),
    "investing": (news_investing.write_investing_parquet, "investing_gold_news.parquet"),
    "bullionvault": (news_bullionvault.write_bullionvault_parquet, "bullionvault_news.parquet"),
    "central_bank": (news_central_bank.write_central_bank_parquet, "central_bank_news.parquet"),
    "reddit": (news_reddit.write_reddit_parquet, "reddit_gold_filtered.parquet"),
    "kaggle": (news_kaggle.write_kaggle_parquet, "kaggle_gold_labeled.parquet"),
    # Wire-news + macro extensions added in phase News-1
    "polygon_news": (news_polygon.write_polygon_news_parquet, "polygon_news_GLD.parquet"),
    "alpha_vantage": (news_alpha_vantage.write_alpha_vantage_parquet, "alpha_vantage_news.parquet"),
    "multisource": (news_multisource.write_multisource_parquet, "multisource_news.parquet"),
    # Polygon prices — replaces dropped Alpaca bars/etfs after KYC detour
    "polygon_bars": (polygon_bars.write_gld_parquet, "polygon_bars_GLD_30min.parquet"),
    "polygon_etfs": (polygon_etfs.write_etf_parquets, None),  # 9 files
}

# Sources that only need network (no API key). Used by `--skip-keyed`.
KEYLESS_SOURCES = {
    "calendar",
    "cot",
    "wgc",
    "gpr",
    "brent_wti",
    "kitco",
    "investing",
    "bullionvault",
    "central_bank",  # HF datasets are public; HF_TOKEN optional
    "reddit",  # arctic HF mirror via DuckDB, no key
    "kaggle",  # HF mirror, public
    "fnspid",  # HF, gated by NANOGLD_NONCOMMERCIAL not by an API key
    "multisource",  # same gate
    # polygon_* + alpha_vantage are key-required but ENV is now populated.
    # Keeping them OUT of KEYLESS_SOURCES so --skip-keyed semantics stay clean.
}


def _load_env() -> None:
    if ENV_PATH.exists():
        load_dotenv(ENV_PATH)
    else:
        LOG.warning("%s missing — sources requiring keys will fail. See docs/SETUP.md.", ENV_PATH)


def _is_idempotent_skip(name: str, *, force: bool) -> bool:
    if force:
        return False
    fn_info = SOURCES.get(name)
    if not fn_info or fn_info[1] is None:
        return False
    out = raw_dir() / fn_info[1]
    if out.exists():
        LOG.info("[%s] %s already exists — skipping (use --force to rebuild)", name, out)
        return True
    return False


def _run_one(name: str, *, force: bool = False) -> int:
    if name not in SOURCES:
        LOG.error("unknown source %r — choose from %s", name, sorted(SOURCES))
        return 2
    if _is_idempotent_skip(name, force=force):
        return 0
    fn = SOURCES[name][0]
    LOG.info("[%s] starting", name)
    try:
        fn()
    except Exception as e:  # noqa: BLE001
        LOG.error("[%s] failed: %s", name, e, exc_info=True)
        return 1
    LOG.info("[%s] done", name)
    return 0


def cmd_pull(args: argparse.Namespace) -> int:
    return _run_one(args.source, force=args.force)


def cmd_build(args: argparse.Namespace) -> int:
    targets = sorted(KEYLESS_SOURCES) if args.skip_keyed else list(SOURCES)
    # Always exclude gdelt_materialize from `build` (catastrophic 1 TB scan).
    targets = [t for t in targets if t != "gdelt_materialize"]
    LOG.info("build: %d sources -> %s", len(targets), targets)

    failed: list[str] = []
    for src in targets:
        if _run_one(src, force=args.force) != 0:
            failed.append(src)

    LOG.info("running joiner")
    sources = joiner.load_default_sources()
    df = joiner.join_snapshot(sources)
    LOG.info("join produced %d rows × %d cols", len(df), df.shape[1])

    parquet, meta_path, meta = snap.write_snapshot(
        df,
        sources=[{"name": s, "expected": SOURCES[s][1]} for s in SOURCES if SOURCES[s][1]],
        extra={"failed_sources": failed} if failed else None,
        overwrite=args.force,
    )
    LOG.info("snapshot %s (%d rows) -> %s", meta["snapshot_hash"], meta["row_count"], parquet)
    LOG.info("meta -> %s", meta_path)

    if failed:
        LOG.warning("FAILED sources: %s — re-run after fixing", failed)
        return 1
    return 0


def cmd_join(args: argparse.Namespace) -> int:
    sources = joiner.load_default_sources()
    df = joiner.join_snapshot(sources)
    parquet, meta_path, meta = snap.write_snapshot(df, overwrite=args.force)
    LOG.info("snapshot %s (%d rows) -> %s", meta["snapshot_hash"], meta["row_count"], parquet)
    LOG.info("meta -> %s", meta_path)
    return 0


def cmd_list(_: argparse.Namespace) -> int:
    print("Available sources:")
    for name in sorted(SOURCES):
        out = SOURCES[name][1] or "(multi-file output)"
        keyless = " (keyless)" if name in KEYLESS_SOURCES else ""
        print(f"  {name:22s} -> {out}{keyless}")
    print(f"\nSnapshots dir: {snapshots_dir()}")
    print(f"Raw dir:       {raw_dir()}")
    return 0


def main(argv: list[str] | None = None) -> int:
    _load_env()
    parser = argparse.ArgumentParser(prog="nanogld.data")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_build = sub.add_parser("build", help="pull every source + join + snapshot")
    p_build.add_argument("--skip-keyed", action="store_true", help="only run keyless sources")
    p_build.add_argument("--force", action="store_true", help="ignore idempotency checks")
    p_build.set_defaults(func=cmd_build)

    p_pull = sub.add_parser("pull", help="pull a single source")
    p_pull.add_argument("source", choices=sorted(SOURCES))
    p_pull.add_argument("--force", action="store_true")
    p_pull.set_defaults(func=cmd_pull)

    p_join = sub.add_parser("join", help="re-run joiner from existing parquets")
    p_join.add_argument("--force", action="store_true")
    p_join.set_defaults(func=cmd_join)

    sub.add_parser("list", help="list available sources").set_defaults(func=cmd_list)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
