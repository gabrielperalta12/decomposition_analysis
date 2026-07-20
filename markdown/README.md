# Decomposition Analysis — Markdown Course

This directory contains the publication-friendly Markdown exports of the executed Python course. Each lesson includes formal derivations, growth-marketing applications, an econometric point of view, limitations, robustness checks, and the authors or papers that later extended the method. Tables appear inline and generated figures are stored in the adjacent `_files` directories.

## Course sequence

1. [What decomposition analysis is](00_field_map.md)
2. [Rate decomposition: Kitagawa, Das Gupta, and replacement](01_rates_kitagawa_dasgupta.md)
3. [Index numbers, LMDI, PVM, shift–share, and SDA](02_index_numbers_lmdi_pvm_sda.md)
4. [Oaxaca–Blinder and distributional decompositions](03_oaxaca_reweighting_rif.md)
5. [Shapley, Aumann–Shapley, functional ANOVA, and SHAP](04_shapley_anova_ml.md)
6. [Econometrics, causal inference, and defensible driver claims](05_econometrics_causality_driver_trees.md)
7. [Subgroup Discovery and Pattern Mining](06_subgroup_discovery_pattern_mining.md)
8. [Complete Growth workflow](07_complete_growth_workflow.md)

## Rebuild

```bash
uv run jupyter nbconvert --to markdown --output-dir=markdown notebooks/*.ipynb
```

Run the notebooks before exporting if their code has changed so the Markdown contains current outputs.
