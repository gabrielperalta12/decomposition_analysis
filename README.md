# Decomposition Analysis in Python

A six-notebook graduate course covering descriptive, distributional, index-number, cooperative-game, machine-learning, and causal approaches to “drivers.” Every notebook includes LaTeX derivations, executable diagnostics, growth-marketing cases, an econometric interpretation, limitations and robustness checks, exercises, and a scholarly “what came next” roadmap.

## Setup with uv

```bash
uv sync
uv run jupyter lab
```

The environment was initialized with `uv init` and is pinned in `uv.lock`. The notebooks are already executed, so their tables and figures are visible when opened.

Markdown versions of every lesson—including executed outputs and generated figures—are available from [`markdown/README.md`](markdown/README.md).

## Course sequence

1. `00_field_map.ipynb` — taxonomy and corrected genealogy
2. `01_rates_kitagawa_dasgupta.ipynb` — demographic rate decomposition
3. `02_index_numbers_lmdi_pvm_sda.ipynb` — index and identity methods
4. `03_oaxaca_reweighting_rif.ipynb` — mean and distribution gaps
5. `04_shapley_anova_ml.ipynb` — axiomatic and predictive attribution
6. `05_econometrics_causality_driver_trees.ipynb` — causal boundaries and driver claims

## Rebuild executed notebooks

```bash
uv run python scripts/build_course.py
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb
uv run jupyter nbconvert --to markdown --output-dir=markdown notebooks/*.ipynb
```

The original research briefs remain in `prompt1.md` and `prompt_driver.md`.
