"""World Gold Council central-bank flows — Source 10.

V4 corrections (plan/02-DATA-PIPELINE.md "Source 10"):
- WGC is **monthly**, not quarterly.
- Direct URLs: gold.org/download/8052 (time series) + 7739 (latest reserves).
- ~2 month IMF reporting lag.
- NO PUBLIC VINTAGE ARCHIVE — must self-snapshot weekly with fetch-date keying.

We download both files weekly, store raw bytes keyed by fetch-ts + content
SHA, and emit a tidy parquet with `(country, period, fetch_ts)` as the primary
key. The fetch-ts is the vintage marker — at training time, we use the latest
fetch_ts <= bar t_visible.
"""

from __future__ import annotations

import hashlib
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from nanogld.data.schema import WGC_MANIFEST, validate
from nanogld.data.utils import UTC, get_logger, http_get_bytes, raw_dir

LOG = get_logger("nanogld.data.wgc")

WGC_QUARTERLY_TIMESERIES = "https://www.gold.org/download/8052"
WGC_LATEST_RESERVES = "https://www.gold.org/download/7739"


def _wgc_cache_dir() -> Path:
    p = raw_dir() / "wgc"
    p.mkdir(parents=True, exist_ok=True)
    return p


def fetch_and_snapshot() -> dict[str, dict[str, str]]:
    """Pull both files; keep raw bytes keyed by fetch-ts + sha.

    fetch_ts is ISO 8601 (parseable by pd.Timestamp). filename uses a
    `:`-stripped variant since macOS filesystem rejects colons.
    """
    now = datetime.now(tz=UTC)
    fetch_ts = now.isoformat().replace("+00:00", "Z")
    fname_ts = now.strftime("%Y%m%dT%H%M%SZ")
    out: dict[str, dict[str, str]] = {}
    for name, url in [
        ("quarterly_ts", WGC_QUARTERLY_TIMESERIES),
        ("latest_reserves", WGC_LATEST_RESERVES),
    ]:
        try:
            body = http_get_bytes(url)
        except Exception as e:  # noqa: BLE001
            LOG.warning("WGC fetch failed (%s): %s", name, e)
            continue
        sha = hashlib.sha256(body).hexdigest()[:16]
        # Sniff content. gold.org/download/<id> URLs return an HTML form wall
        # for direct GETs; spec line 162 flags this — needs /browse to bypass.
        if body.startswith(b"PK\x03\x04"):
            ext = ".xlsx"
        elif body.startswith(b"<!DOCTYPE") or body[:6].lower().startswith(b"<html"):
            ext = ".html"
            LOG.warning(
                "WGC %s returned HTML form-wall (%d B). Direct download blocked; "
                "owner must extract real URL via /browse (spec line 162). "
                "Snapshot kept for audit.",
                name,
                len(body),
            )
        else:
            ext = ".bin"
        path = _wgc_cache_dir() / f"{name}_{fname_ts}_{sha}{ext}"
        path.write_bytes(body)
        out[name] = {
            "path": str(path),
            "sha": sha,
            "fetch_ts": fetch_ts,
            "size_kb": str(len(body) // 1024),
        }
        LOG.info(
            "snapshotted WGC %s -> %s (%s KB, sha=%s)", name, path.name, len(body) // 1024, sha
        )
    return out


def _release_ts_for_period(period_start: pd.Timestamp) -> pd.Timestamp:
    """Conservative: 1st BD of (period_start + 2 months) at 12:00 UTC (London noon)."""
    target = period_start + pd.DateOffset(months=2)
    d = date(target.year, target.month, 1)
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return pd.Timestamp(datetime.combine(d, datetime.min.time()), tz=UTC) + pd.Timedelta(hours=12)


OWNER_QUARTERLY_PATH = Path("data/raw/wgc/quarterly_ts_owner_provided.xlsx")
OWNER_LATEST_PATH = Path("data/raw/wgc/latest_reserves_owner_provided.xlsx")


def parse_owner_quarterly_xlsx(path: Path) -> pd.DataFrame:
    """Parse owner-provided WGC quarterly time series (real Goldhub xlsx).

    Sheet 'Gold (Tonnes)': wide-form. Row 1 has 'Q1 2000', 'Q2 2000', ...
    Cols 0-1 = country names (alt-spelling pair). Rows 3+ = data per country.

    Returns long-form (country, period, holdings_tonnes).
    """
    try:
        df = pd.read_excel(path, sheet_name="Gold (Tonnes)", header=None)
    except Exception as e:  # noqa: BLE001
        LOG.warning("WGC owner xlsx parse failed: %s", e)
        return pd.DataFrame()

    # Row 1 holds quarter labels starting at col 2.
    quarter_labels = df.iloc[1, 2:].tolist()
    countries = df.iloc[2:, 0].astype(str).str.strip()
    values = df.iloc[2:, 2:].reset_index(drop=True)

    rows: list[dict[str, object]] = []
    for c_idx, country in enumerate(countries.tolist()):
        if not country or country.lower() in {"nan", "none", "world"}:
            continue
        for q_idx, qlabel in enumerate(quarter_labels):
            if not isinstance(qlabel, str):
                continue
            v = values.iat[c_idx, q_idx]
            if pd.isna(v):
                continue
            # qlabel like 'Q1 2000' -> parse to first day of quarter
            try:
                quarter, year = qlabel.split()
                m_map = {"Q1": 1, "Q2": 4, "Q3": 7, "Q4": 10}
                period = pd.Timestamp(year=int(year), month=m_map[quarter], day=1, tz="UTC")
            except Exception:  # noqa: BLE001
                continue
            rows.append({"country": country, "period": period, "holdings_tonnes": float(v)})

    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    import numpy as np  # noqa: PLC0415

    out["country"] = out["country"].astype("string")
    out["frequency"] = pd.array(["quarterly"] * len(out), dtype="string")
    out["net_purchases_tonnes"] = pd.array([np.nan] * len(out), dtype="float64")
    out["pct_total_reserves"] = pd.array([np.nan] * len(out), dtype="float64")
    out["fetch_ts"] = pd.Timestamp("2026-05-05T14:40:00+00:00", tz="UTC")
    out["source_sha"] = pd.array(["owner_provided"] * len(out), dtype="string")
    out["release_ts"] = out["period"].apply(_release_ts_for_period)
    out["t_visible"] = out["release_ts"]
    return out


def parse_latest_reserves_xlsx(path: Path) -> pd.DataFrame:
    """Best-effort parser. WGC layout drifts — sheet 0 is typically the wide-form table.
    We try a few likely sheet names + auto-detect the country column. If parse fails,
    return an empty frame; caller should re-attempt after an owner /browse audit.
    """
    try:
        x = pd.ExcelFile(path)
    except Exception as e:  # noqa: BLE001
        LOG.warning("WGC excel open failed (%s): %s", path.name, e)
        return pd.DataFrame()

    # Try a few candidate sheet names; fall back to first sheet
    sheet_candidates = [
        s for s in x.sheet_names if "monthly" in s.lower() or "quarter" in s.lower()
    ]
    sheet = sheet_candidates[0] if sheet_candidates else x.sheet_names[0]
    df = x.parse(sheet, header=None)

    # Heuristic: drop leading metadata rows; find the row that mentions "Country".
    header_idx = None
    for i, row in df.iterrows():
        if any(isinstance(v, str) and "country" in v.lower() for v in row.tolist()):
            header_idx = i
            break
    if header_idx is None:
        LOG.warning("WGC parse: no header row matching 'Country' on sheet %r", sheet)
        return pd.DataFrame()

    df.columns = df.iloc[header_idx].astype(str).str.strip()
    df = df.iloc[header_idx + 1 :].reset_index(drop=True)
    df = df.dropna(how="all")
    return df


def build_wgc_dataframe(snap: dict[str, dict[str, str]]) -> pd.DataFrame:
    """Stitch the latest snapshot into the schema. Best-effort tidy frame.

    For V1 we emit one row per (country, period, fetch_ts) for the latest_reserves
    file. The quarterly_ts file is downloaded for completeness but parsing is
    deferred until owner /browse-confirms the layout.
    """
    if "latest_reserves" not in snap:
        return pd.DataFrame()

    info = snap["latest_reserves"]
    parsed = parse_latest_reserves_xlsx(Path(info["path"]))
    if parsed.empty:
        return pd.DataFrame()

    country_col = next((c for c in parsed.columns if c and "country" in c.lower()), None)
    holdings_col = next((c for c in parsed.columns if c and "tonnes" in c.lower()), None)
    pct_col = next(
        (c for c in parsed.columns if c and "%" in str(c) and "reserves" in c.lower()), None
    )

    if country_col is None:
        return pd.DataFrame()

    fetch_ts = pd.Timestamp(info["fetch_ts"], tz=UTC)
    period = pd.Timestamp(
        datetime.now(tz=UTC).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    )
    rows: list[dict[str, object]] = []
    for _, r in parsed.iterrows():
        country = str(r[country_col]).strip()
        if not country or country.lower() in {"nan", "none"}:
            continue
        rows.append(
            {
                "country": country,
                "period": period,
                "frequency": "monthly",
                "holdings_tonnes": pd.to_numeric(r[holdings_col], errors="coerce")
                if holdings_col
                else pd.NA,
                "net_purchases_tonnes": pd.NA,  # derived later (Δ across snapshots)
                "pct_total_reserves": pd.to_numeric(r[pct_col], errors="coerce")
                if pct_col
                else pd.NA,
                "fetch_ts": fetch_ts,
                "source_sha": info["sha"],
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["country"] = df["country"].astype("string")
    df["frequency"] = df["frequency"].astype("string")
    df["source_sha"] = df["source_sha"].astype("string")
    for c in ("holdings_tonnes", "net_purchases_tonnes", "pct_total_reserves"):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")

    df["release_ts"] = df["period"].apply(_release_ts_for_period)
    df["t_visible"] = df["release_ts"]
    return df


def write_wgc_parquet() -> tuple[pd.DataFrame, str]:
    # Prefer owner-provided xlsx when present (form-walled URLs blocked direct GET).
    if OWNER_QUARTERLY_PATH.exists():
        LOG.info("WGC: using owner-provided %s", OWNER_QUARTERLY_PATH.name)
        df = parse_owner_quarterly_xlsx(OWNER_QUARTERLY_PATH)
        if not df.empty:
            validate(df, WGC_MANIFEST)
            out_path = raw_dir() / "wgc_central_bank_quarterly.parquet"
            df.to_parquet(out_path, compression="zstd", index=False)
            LOG.info("wrote %d WGC rows -> %s", len(df), out_path)
            return df, str(out_path)
        LOG.warning("owner-provided WGC parse produced 0 rows; falling back to live download")

    snap = fetch_and_snapshot()
    df = build_wgc_dataframe(snap)
    if df.empty:
        LOG.warning("WGC parse produced 0 rows; raw snapshot retained for manual inspection")
        return df, ""
    validate(df, WGC_MANIFEST)
    out_path = raw_dir() / "wgc_central_bank_monthly.parquet"
    df.to_parquet(out_path, compression="zstd", index=False)
    LOG.info("wrote %d WGC rows to %s", len(df), out_path)
    return df, str(out_path)


if __name__ == "__main__":
    df, path = write_wgc_parquet()
    if path:
        print(f"wrote {len(df)} rows to {path}")
        print(df.head(5).to_string())
    else:
        print("WGC parse incomplete — raw snapshots saved under data/raw/wgc/")
