# SENTINEL — NeurIPS 2026 Submission

## Status

- `neurips_2026.sty` is the **official 2026 style file** (pulled from `media.neurips.cc/Conferences/NeurIPS2026/Formatting_Instructions_For_NeurIPS_2026.zip`, last revised 2026-01-29).
- `checklist.tex` follows the official checklist template, answers filled in.
- `main.pdf` builds cleanly to **12 pages** (7 body content + 1 references + 4 appendices + checklist). NeurIPS 2026 limit is 10 content pages; references/appendix/checklist do not count. Compliant.

## Track / package options

`main.tex` currently uses `\usepackage{neurips_2026}` — the default, which is equivalent to `[main]` (Main Track) with double-blind anonymization. To switch tracks, swap the option:

```latex
\usepackage{neurips_2026}                    % Main Track (default, anonymous)
\usepackage[eandd]{neurips_2026}             % Evaluations & Datasets Track
\usepackage[position]{neurips_2026}          % Position Paper Track
\usepackage[creativeai]{neurips_2026}        % Creative AI Track
\usepackage[main, final]{neurips_2026}       % Camera-ready (after acceptance)
\usepackage[preprint]{neurips_2026}          % arXiv preprint
```

## Before submitting

1. **Verify OpenReview profile** for every co-author. New profiles without an institutional email take up to 2 weeks to moderate — start this immediately if not already done.
2. **Anonymize the supplementary code snapshot:** scrub `austinjin1`, `bcheng`, "Stockholm Junior Water Prize", and the `/home/bcheng/...` checkpoint paths visible in some `results/*.json` files.
3. **Verify the checklist** — every answer in `checklist.tex` should still be accurate after any further edits.

## Build

```bash
cd paper
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

## Files

- `main.tex` — paper source
- `neurips_2026.sty` — style placeholder (replace with official)
- `references.bib` — bibliography
- `figures/` — generated PDF figures
- `make_figures.py` — script to (re)generate figures from `../results/*.json`
