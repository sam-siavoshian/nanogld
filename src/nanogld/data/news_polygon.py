"""Polygon / Massive News API — drop-in for the dropped Alpaca News module.

V4-style schema (`published_utc`, NOT `updated_at` — the V4 §2 hard rule
applies identically to any wire feed).

Spec note: this module ships GATED behind NANOGLD_POLYGON_PAID=1 by default.
Polygon's free Stocks Basic tier MAY include `/v2/reference/news`; verify
at first run. If free works, owner can flip the gate off in cli.py later.
For now the gate prevents accidentally hitting a paid endpoint.

License: Polygon's commercial use clause permits internal use on Starter
($29/mo) and up. Free tier is non-commercial use only — confirm before
deploying.
"""

from __future__ import annotations

import os
from datetime import date

import pandas as pd

from nanogld.data.schema import NEWS_MANIFEST, validate
from nanogld.data.utils import NEWS_LATENCY_SEC_ALPACA, get_logger, raw_dir

LOG = get_logger("nanogld.data.news_polygon")

BIAS_TIER = "mainstream_neutral"
SOURCE_NAME = "polygon"


def _gate_open() -> bool:
    if os.environ.get("NANOGLD_POLYGON_PAID") != "1":
        LOG.warning(
            "Polygon News skipped — NANOGLD_POLYGON_PAID not set. "
            "Set the flag if you have a paid Polygon plan or have verified "
            "the free tier includes /v2/reference/news."
        )
        return False
    return True


def _client():
    api_key = os.environ.get("POLYGON_API_KEY")
    if not api_key:
        raise RuntimeError(
            "POLYGON_API_KEY missing — populate ~/.config/nanogld/.env.paper after "
            "signing up at https://massive.com/dashboard/signup (formerly polygon.io)."
        )
    from polygon import RESTClient  # noqa: PLC0415

    return RESTClient(api_key=api_key)


def fetch_news(
    *,
    ticker: str = "GLD",
    start: date = date(2021, 4, 24),
    end: date = date(2026, 4, 24),
    per_page: int = 1000,
) -> pd.DataFrame:
    if not _gate_open():
        return pd.DataFrame()

    client = _client()
    rows: list[dict[str, object]] = []
    for art in client.list_ticker_news(
        ticker=ticker,
        published_utc_gte=start.isoformat(),
        published_utc_lte=end.isoformat(),
        order="asc",
        limit=per_page,
    ):
        publisher = getattr(art, "publisher", None)
        publisher_name = publisher.name if publisher and hasattr(publisher, "name") else "unknown"
        rows.append(
            {
                "article_id": str(getattr(art, "id", "")),
                "source": f"{SOURCE_NAME}_{publisher_name}",
                "created_at": pd.to_datetime(getattr(art, "published_utc", None), utc=True),
                "title": str(getattr(art, "title", "")),
                "body": str(getattr(art, "description", "") or ""),
                "url": str(getattr(art, "article_url", "") or ""),
                "symbols": "|".join(getattr(art, "tickers", []) or []),
                "bias_tier": BIAS_TIER,
            }
        )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = df.dropna(subset=["created_at"])
    for c in ("article_id", "source", "title", "body", "url", "symbols", "bias_tier"):
        df[c] = df[c].astype("string")
    df["release_ts"] = df["created_at"]
    df["t_visible"] = df["created_at"] + pd.Timedelta(seconds=NEWS_LATENCY_SEC_ALPACA)
    return df[[c.name for c in NEWS_MANIFEST.columns]].reset_index(drop=True)


def write_polygon_news_parquet(
    *,
    ticker: str = "GLD",
    start: date = date(2021, 4, 24),
    end: date = date(2026, 4, 24),
) -> tuple[pd.DataFrame, str]:
    df = fetch_news(ticker=ticker, start=start, end=end)
    if df.empty:
        return df, ""
    validate(df, NEWS_MANIFEST)
    out_path = raw_dir() / f"polygon_news_{ticker}.parquet"
    df.to_parquet(out_path, compression="zstd", index=False)
    LOG.info("wrote %d Polygon News rows -> %s", len(df), out_path)
    return df, str(out_path)


if __name__ == "__main__":
    df, p = write_polygon_news_parquet()
    print(f"Polygon News: {len(df)} rows -> {p}")
