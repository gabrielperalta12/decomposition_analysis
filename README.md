# Decomposition Analysis in Python

An eight-notebook graduate course covering descriptive, distributional, index-number, cooperative-game, machine-learning, causal, pattern-mining, and end-to-end Growth workflows. Every notebook includes LaTeX derivations, executable diagnostics, growth-marketing cases, an econometric interpretation, limitations and robustness checks, exercises, and references.

## Setup with uv

```bash
uv sync
uv run jupyter lab
```

The environment was initialized with `uv init` and is pinned in `uv.lock`. The notebooks are already executed, so their tables and figures are visible when opened.

Markdown versions of every lesson—including executed outputs and generated figures—are available from [`markdown/README.md`](markdown/README.md).

## Production implementation of Notebook 01

The validated Python API is in `scripts/rate_decomposition.py`. It implements:

- two-period Kitagawa, the one-variable Chevan category report, and the exact two-variable Chevan–Sutherland refinement;
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

## Production implementation of the Notebook 04 route

The API in `scripts/shapley_decomposition.py` keeps the Shapley allocation operator separate from the scientific definition of the coalition value. It includes:

- exact generic Shapley allocation and Shorrocks activation/neutralization;
- Biewen pure main and interaction effects through Möbius inversion;
- Israeli decomposition of linear-regression $R^2$;
- Zhao channel and ordered-touchpoint attribution for observed journeys;
- CF-Shapley driven by an explicit unit- or timestamp-specific counterfactual oracle.

```python
import numpy as np

from scripts.shapley_decomposition import (
    biewen_interactions,
    counterfactual_shapley,
    israeli_r2_shapley,
    shorrocks_decomposition,
    zhao_journey_attribution,
)

biewen = biewen_interactions(base, final, revenue)
shorrocks = shorrocks_decomposition(base, final, revenue)

# This equality is a tested identity: Shapley divides each pure interaction
# equally among the factors that participate in it.
np.testing.assert_allclose(
    biewen.shapley_from_interactions.loc[shorrocks.contributions.index],
    shorrocks.contributions,
)
```

`counterfactual_shapley` does not estimate a causal model. It allocates values returned by the supplied counterfactual oracle. The caller remains responsible for the causal graph, structural equations, reference intervention, identification assumptions, and uncertainty.

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
7. `06_subgroup_discovery_pattern_mining.ipynb` — CART, CHAID, subgroup discovery, and RuleFit for locating change
8. `07_complete_growth_workflow.ipynb` — KPI monitoring, exact decomposition, HCD localization, experiment, and economics

## Rebuild executed notebooks

```bash
uv run python scripts/build_course.py
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb
uv run jupyter nbconvert --to markdown --output-dir=markdown notebooks/*.ipynb
```

The original research briefs remain in `prompt1.md` and `prompt_driver.md`.
