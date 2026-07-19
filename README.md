# Decomposition Analysis in Python

A six-notebook graduate course covering descriptive, distributional, index-number, cooperative-game, machine-learning, and causal approaches to “drivers.” Every notebook includes LaTeX derivations, executable diagnostics, growth-marketing cases, an econometric interpretation, limitations and robustness checks, exercises, and a scholarly “what came next” roadmap.

## Setup with uv

```bash
uv sync
uv run jupyter lab
```

The environment was initialized with `uv init` and is pinned in `uv.lock`. The notebooks are already executed, so their tables and figures are visible when opened.

Markdown versions of every lesson—including executed outputs and generated figures—are available from [`markdown/README.md`](markdown/README.md).

## Production implementation of Notebook 01

The validated Python API is in `scripts/rate_decomposition.py`. It implements:

- two-period Kitagawa and the one-variable Chevan category report;
- order-dependent stepwise replacement;
- symmetric Das Gupta/Shorrocks all-orders allocation, exact or Monte Carlo;
- hierarchical Owen allocation for predeclared factor groups;
- direct and chained multiperiod Kitagawa;
- explicit policies for segments that enter or leave with an unidentified zero-weight rate.

These methods allocate an observed change; they do **not** identify causal effects.

Run a two-period CSV (`segment,w0,w1,r0,r1`):

```bash
uv run python -m scripts.run_rate_decomposition kitagawa examples/rates_two_period.csv
```

Run a panel CSV (`period,segment,weight,rate`):

```bash
uv run python -m scripts.run_rate_decomposition multiperiod examples/rates_multiperiod.csv
```

For an entrant whose baseline weight and rate are both absent, choose the convention explicitly:

```bash
uv run python -m scripts.run_rate_decomposition kitagawa input.csv \
  --missing-rate-policy separate
```

Use `reference` plus `--reference-rate` when a defensible reference rate exists. The total change remains observed, but the split between labels can depend on that convention.

The factor-replacement methods accept any deterministic business identity:

```python
from scripts.rate_decomposition import (
    das_gupta_decomposition,
    hierarchical_owen_decomposition,
    shorrocks_decomposition,
    stepwise_decomposition,
)

base = {"traffic": 1_000, "cvr": 0.04, "aov": 50}
final = {"traffic": 1_200, "cvr": 0.05, "aov": 48}
revenue = lambda x: x["traffic"] * x["cvr"] * x["aov"]

ordered = stepwise_decomposition(base, final, revenue, ["traffic", "cvr", "aov"])
symmetric = das_gupta_decomposition(base, final, revenue)
shorrocks = shorrocks_decomposition(base, final, revenue)
hierarchical = hierarchical_owen_decomposition(
    base,
    final,
    revenue,
    groups={"acquisition": ["traffic"], "economics": ["cvr", "aov"]},
)
```

`Das Gupta` and `Shorrocks` intentionally share the same all-orders engine here: with the same value function and coalition convention they produce the same symmetric allocation. The aliases preserve the interpretation used in the notebook.

Run the production test suite with:

```bash
uv run python -m unittest discover -s tests -v
```

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
