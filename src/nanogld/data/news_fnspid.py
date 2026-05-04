"""FNSPID Historical News Corpus — Source 12.

15.7M articles, 1999-2023, multi-source (Reuters, NASDAQ, Benzinga, Lenta, etc).
arXiv:2402.06698, CC BY 4.0. Single biggest free win for pre-2021 history.

Spec: plan/02-DATA-PIPELINE.md "Source 12".
- Filter to gold-relevant tickers + commodity miners + macro proxies.
- Date-precise only (YYYY-MM-DD); t_visible = first RTH bar of date+1.
- License: CC BY 4.0 — cite arXiv:2402.06698 in README.
"""

from __future__ import annotations

import os

import pandas as pd

from nanogld.data.schema import NEWS_MANIFEST, validate
from nanogld.data.utils import ET, UTC, get_logger, raw_dir

LOG = get_logger("nanogld.data.news_fnspid")

DATASET = "Zihan1004/FNSPID"
GOLD_RELEVANT_TICKERS = frozenset(
    [
        "GLD",
        "GDX",
        "SLV",
        "IAU",
        "NEM",
        "GOLD",
        "FNV",
        "AEM",  # major gold miners
        "TLT",
        "IEF",
        "UUP",  # macro proxies (rates, dollar)
    ]
)
BIAS_BY_SOURCE = {
    "reuters": "mainstream_neutral",
    "nasdaq": "mainstream_neutral",
    "benzinga": "mainstream_neutral",
    "lenta": "mainstream_neutral",
    "cnnmoney": "mainstream_neutral",
    "marketwatch": "mainstream_neutral",
    "yahoo": "aggregator_neutral",
}
DEFAULT_BIAS = "mainstream_neutral"


def _t_visible_for_date(d: pd.Timestamp) -> pd.Timestamp:
    """09:30 ET on date+1 (first RTH bar after the article date)."""
    next_day = d.tz_convert(UTC).normalize() + pd.Timedelta(days=1)
    et = pd.Timestamp(next_day.date()).replace(hour=9, minute=30).tz_localize(ET)
    return et.tz_convert(UTC)


def _ensure_token() -> str | None:
    tok = os.environ.get("HF_TOKEN")
    if not tok or "FILL_ME" in str(tok):
        LOG.warning("HF_TOKEN missing — public datasets will still load, gated may fail")
        return None
    return tok


def fetch_filtered(
    tickers: frozenset[str] = GOLD_RELEVANT_TICKERS,
    *,
    streaming: bool = True,
) -> pd.DataFrame:
    """Stream FNSPID, filter to relevant tickers. Streaming avoids 50 GB local download.

    Args:
        tickers: which symbols to keep.
        streaming: True streams from HF; False downloads then filters (slow).
    """
    from datasets import load_dataset  # noqa: PLC0415

    LOG.info("loading FNSPID streaming=%s", streaming)
    token = _ensure_token()
    ds = load_dataset(DATASET, split="train", streaming=streaming, token=token)
    rows: list[dict[str, object]] = []
    for ex in ds:
        sym = str(ex.get("symbol", "")).upper().strip()
        if sym not in tickers:
            continue
        date_str = ex.get("date")
        if not date_str:
            continue
        rows.append(
            {
                "article_id": str(ex.get("id") or f"fnspid_{len(rows)}"),
                "source": str(ex.get("source", "fnspid")).lower(),
                "created_at": pd.to_datetime(date_str, utc=True, errors="coerce"),
                "title": str(ex.get("title", "")),
                "body": str(ex.get("body", "") or ""),
                "url": str(ex.get("url", "") or ""),
                "symbols": sym,
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = df.dropna(subset=["created_at"])
    df["bias_tier"] = df["source"].map(BIAS_BY_SOURCE).fillna(DEFAULT_BIAS).astype("string")
    for c in ("article_id", "source", "title", "body", "url", "symbols"):
        df[c] = df[c].astype("string")
    df["release_ts"] = df["created_at"]
    df["t_visible"] = df["created_at"].apply(_t_visible_for_date)
    return df[[c.name for c in NEWS_MANIFEST.columns]].reset_index(drop=True)


def write_fnspid_parquet(streaming: bool = True) -> tuple[pd.DataFrame, str]:
    df = fetch_filtered(streaming=streaming)
    if df.empty:
        return df, ""
    validate(df, NEWS_MANIFEST)
    out_path = raw_dir() / "fnspid_gold_relevant.parquet"
    df.to_parquet(out_path, compression="zstd", index=False)
    LOG.info("wrote %d FNSPID rows -> %s", len(df), out_path)
    return df, str(out_path)


if __name__ == "__main__":
    df, p = write_fnspid_parquet()
    print(f"FNSPID: {len(df)} rows -> {p}")
