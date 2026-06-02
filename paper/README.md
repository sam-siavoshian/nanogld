# nanoGLD — research paper

Final negative-result writeup of the nanoGLD V1 / V2 / V3a experiments
and the V1-SPEC §0 Gao 2014 + XGBoost fallback baseline. Lives in
markdown so it renders on GitHub directly; a LaTeX export step is
queued for the arxiv submission below.

Layout
------

- `nanogld.md`         — canonical paper (abstract → §9 reproducibility).
- `refs.bib`           — 24-entry BibTeX bibliography.
- `outline.md`         — original section outline + word-budget targets.
- `build_paper.py`     — auto-fills `figures/results_table.md` +
                          `figures/promotion_gates.md` + `results_summary.json`
                          from every `reports/backtest/*.md` and
                          `reports/ensemble_fallback/*.md`. Idempotent.

Ship state (commit 9ff9aa4, 2026-05-23)
---------------------------------------

- §6 results filled with measured numbers: V1 (−0.78 / −1.01), V2
  (−0.85 / −1.07), V3a (0.000), ensemble (−3.19 / −4.27).
- §6.6 ship decision: ship nothing as an active strategy. Buy-and-hold
  dominates at +0.85 Sharpe net 2 bp; nanoGLD V2 fails G1 / G3 by
  −1.70 against it. The V1-SPEC §0 fallback (Gao + XGBoost ensemble)
  also fails G1 at −3.19.
- §7 attribution filled with VSN top-3 + 4-modality ablation table from
  the V2 fold-0 analysis CLI run on `val_c` (1,885 bars). IG +
  permutation columns deferred to a CPU-finish in
  `reports/analysis/fold_0_v2/` and will append in a follow-up commit.
- §8 limitations explicitly notes (a) the Mac mini 16 GB MPS allocator
  hardware ceiling that blocks V3b / V3c retrains, and (b) the
  single-split disclosure: `walk_forward.py` exists but is not wired
  into `training/train.py`, so the "4-fold" framing in §5 is reported
  honestly as one static `train` / `val` / `test` partition.

Arxiv prep (queued, requires pandoc)
------------------------------------

```bash
brew install pandoc texlive
pandoc paper/nanogld.md -o paper/nanogld.tex \
    --bibliography=paper/refs.bib \
    --csl=paper/icml.csl \
    --citeproc \
    --standalone
```

Then patch the LaTeX with the venue template (ICLR / NeurIPS / TMLR)
and push as a preprint. The markdown source is the canonical document.

Author: Saam Siavoshian.
