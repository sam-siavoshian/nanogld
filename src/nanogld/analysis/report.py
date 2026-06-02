"""Aggregate all 6 attribution methods into a single markdown report.

Inputs are dict outputs from the per-method modules; this file just
formats and writes. Atomic write via tmp + os.replace.

Output:
    output_dir / "analysis_<run_hash>_<git_sha>.md"
    output_dir / "feature_importance.parquet"
    output_dir / "modality_ablation.json"
    output_dir / "attention_rollout.json"
    output_dir / "vsn_gate_distribution.json"
    output_dir / "permutation_importance.parquet"
    output_dir / "integrated_gradients.parquet"
    output_dir / "feature_groups.md"
    output_dir / "manifest.json"

Spec: plan/V1-SPEC.md §11.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import socket
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from nanogld.analysis.config import AnalysisConfig
from nanogld.analysis.feature_groups import (
    GroupRollup,
    classify_features,
    rollup_by_group,
)

LOG = logging.getLogger("nanogld.analysis.report")


def _git_sha(repo_root: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return "unknown"


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _atomic_write_json(path: Path, payload: Any) -> None:  # noqa: ANN401
    tmp = path.with_suffix(path.suffix + ".tmp")

    def _default(obj: Any) -> Any:  # noqa: ANN401
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Path):
            return str(obj)
        raise TypeError(f"unhashable {type(obj)}")

    tmp.write_text(json.dumps(payload, indent=2, default=_default), encoding="utf-8")
    os.replace(tmp, path)


def _write_parquet(path: Path, columns: dict[str, np.ndarray]) -> None:
    """Write a parquet file with the given columns.

    Falls back to CSV if pyarrow is unavailable.
    """
    try:
        import pandas as pd  # noqa: PLC0415

        df = pd.DataFrame({k: np.asarray(v).reshape(-1) for k, v in columns.items()})
        tmp = path.with_suffix(path.suffix + ".tmp")
        df.to_parquet(tmp)
        os.replace(tmp, path)
    except Exception:  # noqa: BLE001 — fall back to CSV
        import csv  # noqa: PLC0415

        rows = list(zip(*[np.asarray(v).reshape(-1).tolist() for v in columns.values()]))
        tmp = path.with_suffix(path.suffix + ".csv.tmp")
        out = path.with_suffix(".csv")
        with tmp.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(list(columns.keys()))
            w.writerows(rows)
        os.replace(tmp, out)


def _format_top_n(
    names: list[str],
    values: np.ndarray,
    n: int = 30,
    descending: bool = True,
) -> str:
    """Render a markdown table of the top-N features by `values`."""
    n = min(n, len(values))
    order = np.argsort(values)
    if descending:
        order = order[::-1]
    order = order[:n]
    lines = ["| Rank | Feature | Value |", "|---|---|---|"]
    for rank, idx in enumerate(order, start=1):
        lines.append(f"| {rank} | `{names[idx]}` | {float(values[idx]):.5f} |")
    return "\n".join(lines)


def _format_groups(rollups: list[GroupRollup]) -> str:
    lines = [
        "| Category | N features | Mean |I| | Sum |I| | Top feature | Top value |",
        "|---|---|---|---|---|---|",
    ]
    for r in rollups:
        lines.append(
            f"| {r.category} | {r.n_features} | "
            f"{r.mean_abs_importance:.5f} | {r.sum_abs_importance:.4f} | "
            f"`{r.top_feature}` | {r.top_value:+.5f} |"
        )
    return "\n".join(lines)


def _format_ablation(ablation: dict[str, dict[str, float]]) -> str:
    if not ablation:
        return "_(no ablation data)_"
    headers = ["Ablation", "Focal", "Sharpe", "Sharpe (news+)", "Sharpe (news-)"]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for name, m in ablation.items():
        lines.append(
            f"| {name} | {m['focal']:.4f} | {m['sharpe']:.3f} | "
            f"{m['sharpe_present']:.3f} | {m['sharpe_absent']:.3f} |"
        )
    return "\n".join(lines)


def _format_attention(att: dict[str, np.ndarray] | None) -> str:
    if att is None:
        return "_(skipped)_"
    mp = att["mean_per_slot"]
    pres = att["mean_present_slot"]
    abs_ = att["mean_absent_slot"]
    n_bars = int(np.asarray(att["n_bars"]).flatten()[0])
    n_present = int(np.asarray(att["n_present"]).flatten()[0])
    lines = [
        f"Aggregated over {n_bars} bars ({n_present} news-present).",
        "",
        "| Slot | Overall | News-present | News-absent |",
        "|---|---|---|---|",
    ]
    for i in range(len(mp)):
        lines.append(
            f"| {i} | {float(mp[i]):.4f} | {float(pres[i]):.4f} | {float(abs_[i]):.4f} |"
        )
    return "\n".join(lines)


def write_report(
    cfg: AnalysisConfig,
    feature_names: list[str],
    vsn: dict[str, np.ndarray] | None,
    ig: dict[str, np.ndarray] | None,
    permutation: dict[str, np.ndarray] | None,
    ablation: dict[str, dict[str, float]] | None,
    attention: dict[str, np.ndarray] | None,
    output_dir: Path | None = None,
    git_sha_override: str | None = None,
) -> Path:
    """Render and write the full analysis report. Returns the markdown path."""
    out = output_dir or cfg.output_dir
    out.mkdir(parents=True, exist_ok=True)

    repo_root = Path(__file__).resolve().parents[3]
    git_sha = git_sha_override or _git_sha(repo_root)
    run_hash = cfg.run_hash()

    # Per-method artifacts
    if vsn is not None:
        _write_parquet(
            out / "vsn_gate_distribution.parquet",
            {
                "feature_idx": np.arange(len(vsn["mean_gate"])),
                "feature_name": np.asarray(feature_names, dtype=object),
                "mean_gate": vsn["mean_gate"],
                "std_gate": vsn["std_gate"],
                "mean_present": vsn["mean_present"],
                "mean_absent": vsn["mean_absent"],
            },
        )
        _atomic_write_json(
            out / "vsn_gate_distribution.json",
            {
                "n_bars": int(vsn["n_bars"][0]),
                "n_present": int(vsn["n_present"][0]),
                "feature_count": int(len(vsn["mean_gate"])),
            },
        )
    if ig is not None:
        _write_parquet(
            out / "integrated_gradients.parquet",
            {
                "feature_idx": np.arange(len(ig["mean_abs"])),
                "feature_name": np.asarray(feature_names, dtype=object),
                "mean_abs": ig["mean_abs"],
                "mean_signed": ig["mean_signed"],
                "per_class_0": ig["per_class_mean"][0],
                "per_class_1": ig["per_class_mean"][1],
                "per_class_2": ig["per_class_mean"][2],
            },
        )
    if permutation is not None:
        _write_parquet(
            out / "permutation_importance.parquet",
            {
                "feature_idx": permutation["feature_idx"],
                "delta_focal_mean": permutation["delta_focal_mean"],
                "delta_focal_std": permutation["delta_focal_std"],
                "delta_sharpe_mean": permutation["delta_sharpe_mean"],
                "delta_sharpe_std": permutation["delta_sharpe_std"],
            },
        )
    if ablation is not None:
        _atomic_write_json(out / "modality_ablation.json", ablation)
    if attention is not None:
        _atomic_write_json(
            out / "attention_rollout.json",
            {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in attention.items()},
        )

    # Combined feature importance parquet (for downstream plotting)
    if vsn is not None and ig is not None:
        _write_parquet(
            out / "feature_importance.parquet",
            {
                "feature_idx": np.arange(len(feature_names)),
                "feature_name": np.asarray(feature_names, dtype=object),
                "category": np.asarray(
                    list(classify_features(feature_names).values()), dtype=object
                ),
                "vsn_mean_gate": vsn["mean_gate"],
                "ig_mean_abs": ig["mean_abs"],
                "ig_mean_signed": ig["mean_signed"],
            },
        )

    rollups: list[GroupRollup] = []
    if ig is not None:
        rollups = rollup_by_group(feature_names, list(ig["mean_abs"]))

    # Markdown report
    md_lines: list[str] = []
    md_lines.append(f"# nanoGLD V1 — feature attribution report")
    md_lines.append("")
    md_lines.append(f"- run_hash: `{run_hash}`")
    md_lines.append(f"- git_sha: `{git_sha}`")
    md_lines.append(f"- generated_utc: `{datetime.now(timezone.utc).isoformat()}`")
    md_lines.append(f"- hostname: `{socket.gethostname()}`")
    md_lines.append(f"- python: `{platform.python_version()}`")
    md_lines.append(f"- fold_idx: `{cfg.fold_idx}`  split: `{cfg.split}`  device: `{cfg.device}`")
    md_lines.append("")
    md_lines.append("## 1. Modality ablation (Sharpe drop per stream)")
    md_lines.append("")
    md_lines.append(_format_ablation(ablation or {}))
    md_lines.append("")
    md_lines.append("## 2. Feature-group rollup (sum |IG|)")
    md_lines.append("")
    md_lines.append(_format_groups(rollups) if rollups else "_(no IG data)_")
    md_lines.append("")
    md_lines.append("## 3. Top-30 features by VSN gate (free native importance)")
    md_lines.append("")
    if vsn is not None:
        md_lines.append(_format_top_n(feature_names, vsn["mean_gate"], n=30, descending=True))
    else:
        md_lines.append("_(skipped)_")
    md_lines.append("")
    md_lines.append("## 4. Top-30 features by |IG|")
    md_lines.append("")
    if ig is not None:
        md_lines.append(_format_top_n(feature_names, ig["mean_abs"], n=30, descending=True))
    else:
        md_lines.append("_(skipped)_")
    md_lines.append("")
    md_lines.append("## 5. Top-30 features by ΔSharpe under permutation (positive = important)")
    md_lines.append("")
    if permutation is not None:
        idxs = permutation["feature_idx"]
        names_perm = [feature_names[int(i)] for i in idxs]
        md_lines.append(
            _format_top_n(
                names_perm, permutation["delta_sharpe_mean"], n=30, descending=True
            )
        )
    else:
        md_lines.append("_(skipped — permutation budget exceeded or disabled)_")
    md_lines.append("")
    md_lines.append("## 6. Cross-attention rollout (which news slots matter)")
    md_lines.append("")
    md_lines.append(_format_attention(attention))
    md_lines.append("")
    md_lines.append("## 7. Bucket sanity (V1 invariant 18)")
    md_lines.append("")
    if vsn is not None:
        delta_pres_abs = vsn["mean_present"] - vsn["mean_absent"]
        top5_news = np.argsort(-delta_pres_abs)[:5]
        top5_quiet = np.argsort(delta_pres_abs)[:5]
        md_lines.append("Features that gain attention WHEN news is present (top 5):")
        for i in top5_news:
            md_lines.append(
                f"- `{feature_names[int(i)]}` Δ={float(delta_pres_abs[int(i)]):+.5f}"
            )
        md_lines.append("")
        md_lines.append("Features that gain attention WHEN news is absent (top 5):")
        for i in top5_quiet:
            md_lines.append(
                f"- `{feature_names[int(i)]}` Δ={float(delta_pres_abs[int(i)]):+.5f}"
            )
    md_lines.append("")
    md_lines.append("## 8. Reproducibility")
    md_lines.append("")
    md_lines.append("```yaml")
    md_lines.append(json.dumps(asdict(cfg), default=str, indent=2))
    md_lines.append("```")

    # Manifest
    manifest = {
        "run_hash": run_hash,
        "git_sha": git_sha,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "argv": list(sys.argv),
        "fold_idx": cfg.fold_idx,
        "split": cfg.split,
        "n_features": len(feature_names),
        "methods_run": {
            "vsn": vsn is not None,
            "integrated_gradients": ig is not None,
            "permutation": permutation is not None,
            "modality_ablation": ablation is not None,
            "attention_rollout": attention is not None,
        },
    }
    _atomic_write_json(out / "manifest.json", manifest)

    md_path = out / f"analysis_{run_hash}_{git_sha}.md"
    _atomic_write_text(md_path, "\n".join(md_lines) + "\n")
    LOG.info("wrote analysis report → %s", md_path)
    return md_path
