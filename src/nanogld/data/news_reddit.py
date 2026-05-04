"""Reddit Arctic Shift dumps — Source 17.

Pushshift successor. Free torrents through 2026-04. Retail-sentiment proxy.
Often counter-indicator at extremes (r/wallstreetbets euphoria precedes pullbacks).

Spec: plan/02-DATA-PIPELINE.md "Source 17".
- Bias tier: retail_social.
- Subreddits: Gold, wallstreetbets, investing, Goldandsilverstackers, Commodities.
- Keyword filter: gold|GLD|silver|SLV|gdx|comex|xau|bullion.
- Each post + comment has UTC timestamp; t_visible = created_utc.
- V1: keep posts + top-comment only (comments outnumber posts 50:1).

Owner downloads Arctic Shift torrents manually (multi-GB, multi-day). This
module parses the resulting zstd-compressed JSONL files.
"""

from __future__ import annotations

import gzip
import json
import re
from pathlib import Path

import pandas as pd

from nanogld.data.schema import NEWS_MANIFEST, validate
from nanogld.data.utils import get_logger, raw_dir

LOG = get_logger("nanogld.data.news_reddit")

SUBREDDITS = ("Gold", "wallstreetbets", "investing", "Goldandsilverstackers", "Commodities")
GOLD_KEYWORDS = re.compile(r"\b(gold|GLD|silver|SLV|gdx|comex|xau|bullion|miner)\b", re.IGNORECASE)
BIAS_TIER = "retail_social"


def _read_jsonl(path: Path):
    """Yield JSON records from .jsonl, .jsonl.gz, or .jsonl.zst."""
    suffix = path.suffix.lower()
    if suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                yield json.loads(line)
    elif suffix == ".zst":
        try:
            import zstandard as zstd  # noqa: PLC0415
        except ImportError as e:
            raise RuntimeError(
                "zstandard package required for .zst dumps — pip install zstandard"
            ) from e
        with path.open("rb") as f:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(f) as r, gzip.io.TextIOWrapper(r, encoding="utf-8") as txt:
                for line in txt:
                    yield json.loads(line)
    else:
        with path.open(encoding="utf-8") as f:
            for line in f:
                yield json.loads(line)


def filter_dump(path: Path, *, kind: str = "submission") -> pd.DataFrame:
    """Filter one Arctic Shift dump file. kind = 'submission' | 'comment'."""
    rows: list[dict[str, object]] = []
    for obj in _read_jsonl(path):
        sub = obj.get("subreddit", "")
        if sub not in SUBREDDITS:
            continue
        text = (
            (obj.get("title") or "")
            + " "
            + (obj.get("selftext") or "")
            + " "
            + (obj.get("body") or "")
        )
        if not GOLD_KEYWORDS.search(text):
            continue
        created = obj.get("created_utc")
        if created is None:
            continue
        rows.append(
            {
                "article_id": str(obj.get("id") or obj.get("name", f"reddit_{len(rows)}")),
                "source": f"reddit_{sub.lower()}_{kind}",
                "created_at": pd.to_datetime(int(created), unit="s", utc=True),
                "title": str(obj.get("title", "") or ""),
                "body": str(obj.get("selftext") or obj.get("body") or ""),
                "url": str(obj.get("url") or obj.get("permalink") or ""),
                "symbols": pd.NA,
            }
        )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    for c in ("article_id", "source", "title", "body", "url"):
        df[c] = df[c].astype("string")
    df["symbols"] = df["symbols"].astype("string")
    df["bias_tier"] = pd.Series([BIAS_TIER] * len(df), dtype="string")
    df["release_ts"] = df["created_at"]
    df["t_visible"] = df["created_at"]
    return df[[c.name for c in NEWS_MANIFEST.columns]].reset_index(drop=True)


def write_reddit_parquet(dump_dir: Path | None = None) -> tuple[pd.DataFrame, str]:
    """Walk dump_dir/* and concatenate filtered Reddit rows.

    Owner drops Arctic Shift `.jsonl.zst` files into `data/raw/reddit/`.
    """
    dump_dir = dump_dir or (raw_dir() / "reddit")
    if not dump_dir.exists():
        LOG.warning("Reddit dump dir %s missing — owner downloads Arctic Shift torrents", dump_dir)
        return pd.DataFrame(), ""

    frames: list[pd.DataFrame] = []
    for path in sorted(dump_dir.glob("*.jsonl*")):
        kind = "comment" if "_RC_" in path.name or "comment" in path.name.lower() else "submission"
        try:
            df = filter_dump(path, kind=kind)
            if not df.empty:
                frames.append(df)
        except Exception as e:  # noqa: BLE001
            LOG.warning("filter failed for %s: %s", path.name, e)

    if not frames:
        return pd.DataFrame(), ""
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["article_id", "source"])
    validate(df, NEWS_MANIFEST)
    out_path = raw_dir() / "reddit_gold_filtered.parquet"
    df.to_parquet(out_path, compression="zstd", index=False)
    LOG.info("wrote %d Reddit rows -> %s", len(df), out_path)
    return df, str(out_path)


if __name__ == "__main__":
    df, p = write_reddit_parquet()
    print(f"Reddit Arctic Shift filtered: {len(df)} rows -> {p}")
