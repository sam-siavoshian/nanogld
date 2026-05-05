"""Alpha Vantage NEWS_SENTIMENT — Source 6F.

Free 25 req/day cap → owner runs this on a daily cron and the journal at
data/raw/alpha_vantage_state.json tracks the cursor so each day picks up
where the previous left off.

History: back to 2022-03-01 (~4 years as of 2026-05).

Schema:
  time_published → created_at  (V4 hard rule applies — never `updated_at`)
  source         → bias_tier mapped per outlet
  ticker_sentiment → optional aggregated sentiment per row (kept under body
                     as JSON string for doc 03 to consume)
"""

from __future__ import annotations

import json
import os
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

from nanogld.data.schema import NEWS_MANIFEST, validate
from nanogld.data.utils import NEWS_LATENCY_SEC_ALPACA, get_logger, raw_dir

LOG = get_logger("nanogld.data.news_alpha_vantage")

API_URL = "https://www.alphavantage.co/query"
DEFAULT_TICKERS = ("GLD", "GDX", "SLV", "IAU", "GOLD", "NEM", "FNV", "AEM")
DAILY_REQ_BUDGET = 25
MAX_PER_CALL = 1000

BIAS_BY_SOURCE = {
    "bloomberg": "mainstream_neutral",
    "reuters": "mainstream_neutral",
    "marketwatch": "mainstream_neutral",
    "wsj": "mainstream_neutral",
    "ft": "mainstream_neutral",
    "cnbc": "mainstream_neutral",
    "yahoo": "aggregator_neutral",
    "motley_fool": "retail_pundit",
    "seeking_alpha": "retail_pundit",
    "benzinga": "mainstream_neutral",
    "zacks": "retail_pundit",
}
DEFAULT_BIAS = "aggregator_neutral"


def _state_path() -> Path:
    return raw_dir() / "alpha_vantage_state.json"


def _load_state() -> dict[str, object]:
    p = _state_path()
    if not p.exists():
        return {"cursor_iso": "2022-03-01T00:00:00Z", "calls_today": 0, "last_call_date": ""}
    return json.loads(p.read_text())


def _save_state(state: dict[str, object]) -> None:
    _state_path().write_text(json.dumps(state, indent=2, default=str))


def _key() -> str:
    k = os.environ.get("ALPHA_VANTAGE_API_KEY")
    if not k or "FILL_ME" in str(k):
        raise RuntimeError(
            "ALPHA_VANTAGE_API_KEY missing — sign up free at "
            "https://www.alphavantage.co/support/#api-key (email-only) and paste "
            "the key into ~/.config/nanogld/.env.paper."
        )
    return k


def _avtime(ts: datetime) -> str:
    return ts.strftime("%Y%m%dT%H%M")


def _fetch_window(time_from: datetime, tickers: tuple[str, ...]) -> dict:
    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": ",".join(tickers),
        "time_from": _avtime(time_from),
        "limit": MAX_PER_CALL,
        "apikey": _key(),
    }
    resp = requests.get(API_URL, params=params, timeout=60)
    resp.raise_for_status()
    return resp.json()


def fetch_one_day(
    *,
    tickers: tuple[str, ...] = DEFAULT_TICKERS,
    max_calls: int = DAILY_REQ_BUDGET,
) -> pd.DataFrame:
    """Run today's budget of API calls, advancing the journaled cursor.

    Each call pulls up to MAX_PER_CALL articles starting at `cursor_iso`.
    Cursor advances to the latest article's time_published returned.
    """
    state = _load_state()
    today = date.today().isoformat()
    if state.get("last_call_date") != today:
        state["calls_today"] = 0
        state["last_call_date"] = today

    rows_collected: list[dict[str, object]] = []
    while state["calls_today"] < max_calls:
        cursor = pd.Timestamp(state["cursor_iso"]).to_pydatetime()
        try:
            data = _fetch_window(cursor, tickers)
        except Exception as e:  # noqa: BLE001
            LOG.warning("AV call failed: %s", e)
            break
        state["calls_today"] += 1

        feed = data.get("feed") or []
        if not feed:
            LOG.info("AV: empty feed at cursor %s — done for now", cursor)
            break

        for art in feed:
            tp = art.get("time_published")
            if not tp:
                continue
            created_at = pd.to_datetime(tp, format="%Y%m%dT%H%M%S", utc=True, errors="coerce")
            if pd.isna(created_at):
                continue
            src = (art.get("source") or "").lower().replace(" ", "_")
            bias = BIAS_BY_SOURCE.get(src, DEFAULT_BIAS)
            tickers_field = "|".join(t.get("ticker", "") for t in art.get("ticker_sentiment", []))
            sentiment_blob = json.dumps(art.get("ticker_sentiment", []))
            rows_collected.append(
                {
                    "article_id": art.get("url") or art.get("title", ""),
                    "source": f"alpha_vantage_{src or 'unknown'}",
                    "created_at": created_at,
                    "title": art.get("title", ""),
                    "body": (art.get("summary") or "") + "\n\nSENTIMENT_JSON:" + sentiment_blob,
                    "url": art.get("url", ""),
                    "symbols": tickers_field,
                    "bias_tier": bias,
                }
            )

        # Advance cursor to the latest seen article so next call picks up after.
        latest = max(r["created_at"] for r in rows_collected[-len(feed) :])
        state["cursor_iso"] = (latest + timedelta(seconds=1)).isoformat()
        LOG.info(
            "AV call #%d: +%d articles, cursor → %s",
            state["calls_today"],
            len(feed),
            state["cursor_iso"],
        )

    _save_state(state)
    if not rows_collected:
        return pd.DataFrame()

    df = pd.DataFrame(rows_collected).drop_duplicates(subset=["article_id"]).reset_index(drop=True)
    for c in ("article_id", "source", "title", "body", "url", "symbols", "bias_tier"):
        df[c] = df[c].astype("string")
    df["release_ts"] = df["created_at"]
    df["t_visible"] = df["created_at"] + pd.Timedelta(seconds=NEWS_LATENCY_SEC_ALPACA)
    return df[[c.name for c in NEWS_MANIFEST.columns]]


def write_alpha_vantage_parquet() -> tuple[pd.DataFrame, str]:
    today_df = fetch_one_day()
    out_path = raw_dir() / "alpha_vantage_news.parquet"
    if out_path.exists():
        prev = pd.read_parquet(out_path)
        if today_df.empty:
            return prev, str(out_path)
        merged = (
            pd.concat([prev, today_df])
            .drop_duplicates(subset=["article_id"])
            .reset_index(drop=True)
        )
    else:
        merged = today_df

    if merged.empty:
        return merged, ""
    validate(merged, NEWS_MANIFEST)
    merged.to_parquet(out_path, compression="zstd", index=False)
    LOG.info("AV: +%d new rows (total %d) -> %s", len(today_df), len(merged), out_path)
    return merged, str(out_path)


if __name__ == "__main__":
    df, p = write_alpha_vantage_parquet()
    print(f"Alpha Vantage News: {len(df)} rows -> {p}")
