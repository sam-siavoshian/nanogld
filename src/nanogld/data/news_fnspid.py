"""FNSPID Historical News Corpus — Source 12.

15.7M articles, 1999-2023, multi-source (Reuters, NASDAQ, Benzinga, Lenta, etc).
arXiv:2402.06698. **License: CC BY-NC-4.0 — NON-COMMERCIAL only** (HF dataset
card; previous spec line saying CC BY 4.0 was wrong — confirmed via Nia 2026-05).

Spec: plan/02-DATA-PIPELINE.md "Source 12".
- Filter to gold-relevant tickers + commodity miners + macro proxies.
- Date-precise only (YYYY-MM-DD); t_visible = first RTH bar of date+1.
- License: **CC BY-NC-4.0**. Gated behind NANOGLD_NONCOMMERCIAL=1 env flag
  (default off). Set the flag to enable for V1 personal/research training.
- Cite arXiv:2402.06698 in README.
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


def _noncommercial_gate_open() -> bool:
    """Return True iff NANOGLD_NONCOMMERCIAL=1 is set.

    FNSPID is CC-BY-NC-4.0. The gate prevents accidental commercial use.
    Default closed; owner sets the flag explicitly to opt in for personal /
    research training.
    """
    if os.environ.get("NANOGLD_NONCOMMERCIAL") != "1":
        LOG.warning(
            "FNSPID skipped — license is CC-BY-NC-4.0 (non-commercial). "
            "Set NANOGLD_NONCOMMERCIAL=1 to enable for personal / research training."
        )
        return False
    return True


def fetch_filtered(
    tickers: frozenset[str] = GOLD_RELEVANT_TICKERS,
    *,
    streaming: bool = False,
    num_proc: int = 4,
) -> pd.DataFrame:
    """Filter FNSPID to gold-relevant tickers. Default: non-streaming + Dataset.filter
    with num_proc=4 for parallel C-level filtering — 10× faster than the prior
    Python-iter loop.

    Gated behind NANOGLD_NONCOMMERCIAL=1 (CC-BY-NC-4.0 license).

    Args:
        tickers: which symbols to keep.
        streaming: True streams from HF (slower iter loop, no local download).
                   False downloads then filters in parallel (much faster, ~7-8 GB).
        num_proc: parallel workers when streaming=False.
    """
    if not _noncommercial_gate_open():
        return pd.DataFrame()

    from datasets import load_dataset  # noqa: PLC0415

    LOG.info("loading FNSPID streaming=%s num_proc=%d", streaming, num_proc)
    token = _ensure_token()
    ds = load_dataset(DATASET, split="train", streaming=streaming, token=token)

    if streaming:
        # Real schema (Nia 2026-05): fields are Capitalized — Stock_symbol, Date,
        # Article_title, Article, Url, Publisher, Author. Some rows have Cyrillic /
        # bad bytes that throw DatasetGenerationError mid-stream — wrap iter loop
        # in try/except per-batch to keep the run alive.
        # Early-stop heuristic: if no new rows kept for 5M consecutive examined,
        # assume we passed the gold-relevant section of the dataset and break.
        rows: list[dict[str, object]] = []
        seen = 0
        last_kept_at = 0
        # 25M chosen empirically: prior run found 52K rows after a >10M-row
        # dry section between gold-tagged shards. Tighter cutoff missed the
        # dense back-half of the dataset.
        EARLY_STOP_GAP = 25_000_000
        try:
            for ex in ds:
                seen += 1
                if seen % 500_000 == 0:
                    LOG.info("FNSPID scan: %d examined, %d kept", seen, len(rows))
                if seen - last_kept_at > EARLY_STOP_GAP and len(rows) > 0:
                    LOG.warning(
                        "FNSPID early-stop: %d rows since last keep — assuming past "
                        "gold-relevant section (kept=%d).",
                        EARLY_STOP_GAP,
                        len(rows),
                    )
                    break
                try:
                    sym = str(ex.get("Stock_symbol") or "").upper().strip()
                    if sym not in tickers:
                        continue
                    date_str = ex.get("Date")
                    if not date_str:
                        continue
                    rows.append(
                        {
                            "article_id": str(ex.get("Url") or f"fnspid_{len(rows)}"),
                            "source": str(ex.get("Publisher") or "fnspid").lower(),
                            "created_at": pd.to_datetime(date_str, utc=True, errors="coerce"),
                            "title": str(ex.get("Article_title") or ""),
                            "body": str(ex.get("Article") or ""),
                            "url": str(ex.get("Url") or ""),
                            "symbols": sym,
                        }
                    )
                    last_kept_at = seen
                except (UnicodeDecodeError, ValueError, TypeError) as e:  # noqa: BLE001
                    LOG.debug("FNSPID skip row %d: %s", seen, e)
                    continue
        except Exception as e:  # noqa: BLE001
            LOG.warning(
                "FNSPID stream halted at row %d (%s) — keeping %d rows already pulled",
                seen,
                e,
                len(rows),
            )
        LOG.info("FNSPID scan complete: %d examined, %d kept", seen, len(rows))
        df = pd.DataFrame(rows)
    else:
        # Non-streaming: vectorized filter in C, then to_pandas. Much faster.
        def _rel(ex: dict) -> bool:
            sym = str(ex.get("symbol", "")).upper().strip()
            return sym in tickers

        filtered = ds.filter(_rel, num_proc=num_proc)
        if len(filtered) == 0:
            return pd.DataFrame()
        df = filtered.to_pandas()
        df["created_at"] = pd.to_datetime(df.get("date"), utc=True, errors="coerce")
        # Conform column names + dtypes downstream-of-here
        df = df.rename(
            columns={
                "id": "article_id",
                "source": "source_raw",
            }
        )
        if "article_id" not in df.columns:
            df["article_id"] = df.index.astype(str)
        df["article_id"] = df["article_id"].astype(str).fillna("fnspid_unknown")
        df["source"] = (df.get("source_raw") or "fnspid").astype(str).str.lower()
        df["title"] = df.get("title", "").astype(str)
        df["body"] = df.get("body", "").fillna("").astype(str)
        df["url"] = df.get("url", "").fillna("").astype(str)
        df["symbols"] = df.get("symbol", "").astype(str).str.upper()
        df = df[["article_id", "source", "created_at", "title", "body", "url", "symbols"]]
    if df.empty:
        return df

    df = df.dropna(subset=["created_at"])
    # FNSPID tags same Article URL across multiple Stock_symbols (e.g. an article
    # mentioning gold + silver miners gets ~5 rows). Dedupe on article_id.
    df = df.drop_duplicates(subset=["article_id"], keep="first").reset_index(drop=True)
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
