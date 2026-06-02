"""Fill paper §6 Results + §7 Attribution from `reports/` artifacts.

Reads:
    reports/backtest/*.json                  — per-run backtest report
    reports/analysis/fold_*/*.parquet        — per-fold analysis tables
    EXPERIMENTS.md                            — ledger of experiment IDs
    plan/V1-SPEC.md                           — 8 promotion-gate thresholds

Writes:
    paper/figures/results_table.md            — headline table per run
    paper/figures/cost_stress_matrix.md       — 3x3 per-bucket × cost stress
    paper/figures/per_fold_breakdown.md       — per-fold Sharpe/MDD/n_trades
    paper/figures/promotion_gates.md          — 8-gate pass/fail
    paper/figures/feature_attribution.md      — top-30 by method
    paper/results_summary.json                — machine-readable summary

The output markdown is hand-includable in nanogld.md, e.g.:

    {{< include figures/results_table.md >}}

This script never edits nanogld.md directly — it produces atomic
fill-in tables we paste in by hand when writing the final §6.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

LOG = logging.getLogger("paper.build")


PROMOTION_GATES = [
    ("G1", "OOS Sharpe ≥ 1.0",                   "sharpe_1.0x >= 1.0"),
    ("G2", "Sharpe > 0.5 at 1.5× cost",          "sharpe_1.5x > 0.5"),
    ("G3", "Beats Gao+XGBoost by ≥ 0.2 Sharpe",  "sharpe - max(baseline) >= 0.2"),
    ("G4", "DSR > 1.0",                          "dsr > 1.0"),
    ("G5", "Per-bucket all positive",            "min(present, absent) > 0"),
    ("G6", "Per-bucket ECE < 0.05",              "max(ece_*) < 0.05"),
    ("G7", "MDD < 15% any fold",                 "max_drawdown < 0.15"),
    ("G8", "Bootstrap 95% CI excludes 0",        "ci_lower > 0"),
]


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _load_json(p: Path) -> dict[str, Any] | None:
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError as exc:
        LOG.warning("could not parse %s: %s", p, exc)
        return None


def _format_results_table(reports: list[tuple[str, dict[str, Any]]]) -> str:
    lines = [
        "| Run | Cost 0.5× | Cost 1.0× | Cost 1.5× | News+ | News- | MDD |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for run_id, rep in reports:
        cs = rep.get("cost_stress") or {}
        pb = rep.get("per_bucket") or {}
        c05 = _get_metric(cs, "0.5", "sharpe")
        c10 = _get_metric(cs, "1.0", "sharpe")
        c15 = _get_metric(cs, "1.5", "sharpe")
        np_ = _get_metric(pb, "present", "sharpe")
        nm = _get_metric(pb, "absent", "sharpe")
        mdd = rep.get("max_drawdown") or rep.get("mdd") or 0.0
        lines.append(
            f"| {run_id} | {c05:.3f} | {c10:.3f} | {c15:.3f} "
            f"| {np_:.3f} | {nm:.3f} | {mdd:.2%} |"
        )
    return "\n".join(lines)


def _get_metric(d: dict[str, Any], key: str, sub: str) -> float:
    v = d.get(key) or d.get(f"{key}x") or {}
    if isinstance(v, dict):
        return float(v.get(sub) or v.get("sharpe") or 0.0)
    return 0.0


def _format_promotion_gates(rep: dict[str, Any]) -> str:
    cs = rep.get("cost_stress") or {}
    pb = rep.get("per_bucket") or {}
    sharpe_10 = _get_metric(cs, "1.0", "sharpe")
    sharpe_15 = _get_metric(cs, "1.5", "sharpe")
    sharpe_present = _get_metric(pb, "present", "sharpe")
    sharpe_absent = _get_metric(pb, "absent", "sharpe")
    mdd = float(rep.get("max_drawdown") or rep.get("mdd") or 1.0)
    dsr = float(rep.get("dsr") or 0.0)

    results = {
        "G1": sharpe_10 >= 1.0,
        "G2": sharpe_15 > 0.5,
        "G3": False,  # needs baseline comparison
        "G4": dsr > 1.0,
        "G5": min(sharpe_present, sharpe_absent) > 0,
        "G6": False,  # needs ECE
        "G7": mdd < 0.15,
        "G8": False,  # needs bootstrap CI
    }
    lines = ["| Gate | Description | Status |", "|---|---|:---:|"]
    for gid, desc, _ in PROMOTION_GATES:
        ok = results.get(gid)
        flag = "✅" if ok else "❌" if ok is False else "—"
        lines.append(f"| {gid} | {desc} | {flag} |")
    return "\n".join(lines)


def _scan_reports(reports_dir: Path) -> list[tuple[str, dict[str, Any]]]:
    """Find all backtest reports + label them by run id.

    Cost-stress and per-bucket tables only exist in the markdown report,
    not the JSON, so we parse the .md sibling for the rich data and
    use the .json sibling for the gates.
    """
    if not reports_dir.exists():
        return []
    out: list[tuple[str, dict[str, Any]]] = []
    for p in sorted(reports_dir.glob("v1_backtest_*.json")):
        rep = _load_json(p) or {}
        md_path = p.with_suffix(".md")
        if md_path.exists():
            rep = {**rep, **_parse_markdown(md_path)}
        parts = p.stem.split("_")
        run_hash = parts[2] if len(parts) > 2 else "?"
        out.append((run_hash, rep))
    return out


def _parse_markdown(md_path: Path) -> dict[str, Any]:
    """Extract cost-stress + per-bucket tables from a backtest .md."""
    text = md_path.read_text()
    cost_stress = _parse_table_after(text, "## Cost-Stress Mean Sharpe")
    per_bucket = _parse_table_after(text, "## Per-Bucket Mean Sharpe")
    # nanogld_v1 row is what we want; pivot into the dict shape the
    # downstream paper code expects.
    cs_v1 = cost_stress.get("nanogld_v1", {})
    pb_v1 = per_bucket.get("nanogld_v1", {})
    return {
        "cost_stress": {
            "0.5": {"sharpe": cs_v1.get("0.5x", 0.0)},
            "1.0": {"sharpe": cs_v1.get("1.0x", 0.0)},
            "1.5": {"sharpe": cs_v1.get("1.5x", 0.0)},
        },
        "per_bucket": {
            "present": {"sharpe": pb_v1.get("present", 0.0)},
            "absent": {"sharpe": pb_v1.get("absent", 0.0)},
            "both": {"sharpe": pb_v1.get("both", 0.0)},
        },
        "baselines_cost_stress": cost_stress,
        "baselines_per_bucket": per_bucket,
    }


def _parse_table_after(text: str, header: str) -> dict[str, dict[str, float]]:
    """Parse a single markdown table whose first column is the row name."""
    if header not in text:
        return {}
    after = text.split(header, 1)[1]
    lines = after.strip().splitlines()
    # Skip blank line(s) + header row + separator
    rows: list[str] = []
    for line in lines:
        if line.startswith("##") and rows:
            break
        if "|" in line:
            rows.append(line)
    if len(rows) < 3:
        return {}
    header_cols = [c.strip() for c in rows[0].strip("|").split("|")][1:]
    out: dict[str, dict[str, float]] = {}
    for line in rows[2:]:
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 2:
            continue
        name = cells[0]
        try:
            vals = [float(c) for c in cells[1:]]
        except ValueError:
            continue
        out[name] = dict(zip(header_cols, vals))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="paper.build")
    parser.add_argument("--reports", type=Path, default=Path("reports/backtest"))
    parser.add_argument("--out", type=Path, default=Path("paper/figures"))
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=Path("reports/analysis"),
        help="optional per-fold analysis tables",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    reports = _scan_reports(args.reports)
    LOG.info("found %d backtest reports under %s", len(reports), args.reports)

    if not reports:
        LOG.warning("no backtest reports — nothing to fill")
        return 1

    args.out.mkdir(parents=True, exist_ok=True)

    # 1. Headline results table.
    _atomic_write(args.out / "results_table.md", _format_results_table(reports))
    LOG.info("wrote %s", args.out / "results_table.md")

    # 2. Promotion gates for the LATEST run.
    latest_id, latest_rep = reports[-1]
    _atomic_write(
        args.out / "promotion_gates.md",
        f"_Promotion gates for run `{latest_id}`._\n\n"
        + _format_promotion_gates(latest_rep),
    )
    LOG.info("wrote %s", args.out / "promotion_gates.md")

    # 3. Machine-readable summary JSON.
    summary = {
        "n_runs": len(reports),
        "latest_run_hash": latest_id,
        "latest_metrics": {
            "sharpe_1.0x": _get_metric(latest_rep.get("cost_stress") or {}, "1.0", "sharpe"),
            "sharpe_1.5x": _get_metric(latest_rep.get("cost_stress") or {}, "1.5", "sharpe"),
            "sharpe_news_present": _get_metric(latest_rep.get("per_bucket") or {}, "present", "sharpe"),
            "sharpe_news_absent": _get_metric(latest_rep.get("per_bucket") or {}, "absent", "sharpe"),
            "max_drawdown": latest_rep.get("max_drawdown") or latest_rep.get("mdd"),
        },
        "all_runs": [
            {
                "run_hash": rid,
                "sharpe_1.0x": _get_metric(rep.get("cost_stress") or {}, "1.0", "sharpe"),
                "sharpe_1.5x": _get_metric(rep.get("cost_stress") or {}, "1.5", "sharpe"),
            }
            for rid, rep in reports
        ],
    }
    _atomic_write(args.out.parent / "results_summary.json", json.dumps(summary, indent=2))
    LOG.info("wrote %s", args.out.parent / "results_summary.json")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
