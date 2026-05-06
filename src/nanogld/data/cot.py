"""CFTC Commitments of Traders (Disaggregated, weekly) — Source 9.

Free public CSV. Every Friday 3:30 PM ET, CFTC publishes positioning as of the
prior Tuesday 4 PM ET. We pull the disaggregated COT report (post-2017 format)
and filter to gold contract code `088691` ("GOLD - COMMODITY EXCHANGE INC.").

Spec: plan/02-DATA-PIPELINE.md "Source 9 — CFTC COT (V4 CORRECTED)".

Hard rules:
- Field names contain UNDERSCORES (V4 correction).
- Historical zip filenames are NOT stable — parse the index page.
- Holiday-Friday rolls to next NYSE session day (cot_release_ts_utc).
- 2025 government shutdown caused a multi-week gap; flag rows with
  irregular_release=True when consecutive reports are >7 days apart.
"""

from __future__ import annotations

import io
import re
import zipfile
from io import BytesIO

import pandas as pd

from nanogld.data.schema import COT_MANIFEST, validate
from nanogld.data.utils import (
    UTC,
    cot_release_ts_utc,
    get_logger,
    http_get_bytes,
    raw_dir,
)

LOG = get_logger("nanogld.data.cot")

GOLD_CONTRACT_CODE = "088691"
GOLD_CONTRACT_NAME = "GOLD - COMMODITY EXCHANGE INC."

CURRENT_YEAR_URL = "https://www.cftc.gov/dea/newcot/f_disagg.txt"
HISTORICAL_INDEX_URL = (
    "https://www.cftc.gov/MarketReports/CommitmentsofTraders/HistoricalCompressed/index.htm"
)
SOCRATA_BACKUP = "https://publicreporting.cftc.gov/resource/72hh-3qpy.csv"


def _zip_url_for_year(year: int) -> str:
    """Spec line 910: parse the index page; fall back to the documented pattern."""
    try:
        html = http_get_bytes(HISTORICAL_INDEX_URL).decode("utf-8", errors="ignore")
        pat = re.compile(rf"href=\"([^\"]*com_disagg[^\"]*{year}[^\"]*\.zip)\"", re.IGNORECASE)
        m = pat.search(html)
        if m:
            href = m.group(1)
            if href.startswith("http"):
                return href
            return "https://www.cftc.gov" + (href if href.startswith("/") else "/" + href)
    except Exception as e:  # noqa: BLE001
        LOG.warning("CFTC index parse failed (%s); falling back to documented URL pattern", e)
    return f"https://www.cftc.gov/files/dea/history/com_disagg_txt_{year}.zip"


def _parse_cot_text(buf: bytes) -> pd.DataFrame:
    """CFTC ships disaggregated reports as fixed-width-ish CSV. Pandas reads it cleanly."""
    df = pd.read_csv(io.BytesIO(buf), low_memory=False)
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]
    return df


def _filter_gold(df: pd.DataFrame) -> pd.DataFrame:
    code_col = next((c for c in df.columns if "Contract_Market_Code" in c), None)
    name_col = next((c for c in df.columns if "Market_and_Exchange_Names" in c), None)
    if code_col is None and name_col is None:
        raise ValueError(
            f"COT frame missing contract code/name columns; got {list(df.columns)[:10]}"
        )
    if code_col is not None:
        return df[df[code_col].astype(str).str.strip() == GOLD_CONTRACT_CODE].copy()
    return df[df[name_col].astype(str).str.strip().str.upper() == GOLD_CONTRACT_NAME].copy()


def _normalize_columns(g: pd.DataFrame) -> pd.DataFrame:
    """Map raw CFTC column names to our schema. Uses g's index for alignment
    (otherwise pd.Series with default RangeIndex misaligns with non-default g).
    """
    g = g.reset_index(drop=True)  # canonical index for alignment
    pick = {
        "report_date_str": next(c for c in g.columns if "Report_Date" in c),
        "oi": next((c for c in g.columns if c == "Open_Interest_All"), None),
        "mm_long": next((c for c in g.columns if c == "M_Money_Positions_Long_All"), None),
        "mm_short": next((c for c in g.columns if c == "M_Money_Positions_Short_All"), None),
        "mm_spread": next((c for c in g.columns if "M_Money_Positions_Spread" in c), None),
        "comm_long": next((c for c in g.columns if c == "Prod_Merc_Positions_Long_All"), None),
        "comm_short": next((c for c in g.columns if c == "Prod_Merc_Positions_Short_All"), None),
        "nonrept_long": next((c for c in g.columns if c == "NonRept_Positions_Long_All"), None),
        "nonrept_short": next((c for c in g.columns if c == "NonRept_Positions_Short_All"), None),
    }
    n = len(g)
    out = pd.DataFrame(
        {
            "contract_code": pd.array([GOLD_CONTRACT_CODE] * n, dtype="string"),
            "contract_name": pd.array([GOLD_CONTRACT_NAME] * n, dtype="string"),
            "report_date": pd.to_datetime(
                g[pick["report_date_str"]].values, utc=True, errors="coerce"
            ),
        }
    )
    for col in (
        "oi",
        "mm_long",
        "mm_short",
        "mm_spread",
        "comm_long",
        "comm_short",
        "nonrept_long",
        "nonrept_short",
    ):
        src = pick[col]
        if src is None:
            out[col] = pd.array([pd.NA] * n, dtype="float64")
        else:
            out[col] = pd.to_numeric(g[src].values, errors="coerce").astype("float64")
    out = out.rename(columns={"oi": "oi_open_interest"})
    return out


def fetch_cot_year(year: int) -> pd.DataFrame:
    """Pull a single historical year. Always uses historical zip — the
    current-year f_disagg.txt short-format lacks the standard column header
    row, so we rely on the year-zip endpoint exclusively (it includes the
    current year as it accumulates).
    """
    url = _zip_url_for_year(year)
    LOG.info("fetching COT %s from %s", year, url)
    try:
        body = http_get_bytes(url)
    except Exception as e:  # noqa: BLE001
        LOG.warning("COT zip fetch failed for %s (%s) — year skipped", year, e)
        return pd.DataFrame()
    try:
        with zipfile.ZipFile(BytesIO(body)) as zf:
            txt_name = next(n for n in zf.namelist() if n.lower().endswith((".txt", ".csv")))
            with zf.open(txt_name) as f:
                df = _parse_cot_text(f.read())
    except Exception as e:  # noqa: BLE001
        LOG.warning("COT zip parse failed for %s (%s) — year skipped", year, e)
        return pd.DataFrame()
    if df.empty:
        return df
    g = _filter_gold(df)
    return _normalize_columns(g) if not g.empty else g


def _flag_irregular(df: pd.DataFrame) -> pd.DataFrame:
    """Mark rows whose gap-from-prev-report is unusual (e.g. 2025 shutdown gap)."""
    df = df.sort_values("report_date").reset_index(drop=True)
    delta = df["report_date"].diff().dt.days
    df["irregular_release"] = (delta > 8) | (delta < 6)
    df.loc[0, "irregular_release"] = False  # first row, no diff
    df["irregular_release"] = df["irregular_release"].fillna(False).astype(bool)
    return df


def fetch_cot_5y(year_start: int = 2016, year_end: int = 2026) -> pd.DataFrame:
    """Concatenate window (default 10y), filter to gold, flag irregular weeks,
    set release_ts. Func name kept for backwards compatibility."""
    frames = [fetch_cot_year(y) for y in range(year_start, year_end + 1)]
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["report_date"]).drop_duplicates(subset=["report_date"])
    df = _flag_irregular(df)

    rt = df["report_date"].dt.tz_convert(UTC).dt.date.map(cot_release_ts_utc)
    df["release_ts"] = pd.to_datetime(rt, utc=True)
    df["t_visible"] = df["release_ts"]
    return df.reset_index(drop=True)


def write_cot_parquet(year_start: int = 2016, year_end: int = 2026) -> tuple[pd.DataFrame, str]:
    df = fetch_cot_5y(year_start, year_end)
    validate(df, COT_MANIFEST)
    out_path = raw_dir() / "cftc_cot_gold_weekly.parquet"
    df.to_parquet(out_path, compression="zstd", index=False)
    LOG.info(
        "wrote %d COT rows to %s (irregular: %d)", len(df), out_path, df["irregular_release"].sum()
    )
    return df, str(out_path)


if __name__ == "__main__":
    df, path = write_cot_parquet()
    print(f"wrote {len(df)} rows to {path}")
    print(df.head(3).to_string())
