# 00 — What decomposition analysis is

**Graduate course: Decomposition Analysis in Python**

## Learning objectives

- Distinguish an accounting identity from a statistical estimand.
- Place the major traditions in a defensible genealogy.
- Use a taxonomy based on the object being decomposed.

## How to use this notebook

This opening lesson gives the vocabulary used throughout the course. Read it before choosing a method: methods that all produce a waterfall chart may answer fundamentally different questions.

### Guiding question

**What exactly is being split, and what kind of claim can the resulting components support?**

### Minimal prerequisites

Only weighted averages and the idea of a before/after comparison are required. Later notebooks add regression, index numbers, game theory, and causal inference.

### Notation used here

- $P_0,P_1$: baseline and comparison populations or empirical distributions.
- $T(P)$: the quantity of interest, such as a mean, rate, total, quantile, or prediction.
- $\Delta_T$: the observed contrast to be decomposed.
- $C_k$: the contribution assigned to component $k$.

The core lesson is simple: **a decomposition is defined by its target and allocation rule, not by the chart used to display it.**

## A family of frameworks, not one unified discipline

A decomposition maps a total, change, gap, distribution, or model prediction into labelled components. The shared requirement is usually **efficiency**: $\sum_j C_j=\Delta$. It does not, by itself, identify causes.

The proposed single lineage in the research brief is historically misleading. Several branches developed partly independently: demographic standardization (Kitagawa, Das Gupta); index numbers and Divisia methods; regional shift–share; labor-market mean/distribution decompositions; input–output structural decomposition; cooperative-game allocations; and statistical/ML function explanations. SHAP borrows Shapley axioms, but is not a descendant of Oaxaca–Blinder or LMDI.

| Object | Representative methods | Typical claim |
|---|---|---|
| aggregate rate | Kitagawa, Das Gupta, stepwise replacement | mix/rate contribution |
| mean or distribution gap | Oaxaca–Blinder, reweighting, RIF | composition/structure gap |
| aggregate identity | PVM, Laspeyres, Fisher, LMDI, SDA | factor contribution |
| value function | Shapley, Aumann–Shapley | axiom-based allocation |
| fitted function | functional ANOVA, SHAP, PDP/ALE | predictive explanation |
| potential outcomes / SCM | DiD, IV, RD, DML | causal effect under assumptions |

## Formal vocabulary: object, contrast, rule, estimand

Let $T(P)$ be a target functional of a population or empirical distribution $P$. A comparison is

$$\Delta_T = T(P_1)-T(P_0).$$

A decomposition rule $\mathcal A$ maps $(P_0,P_1,T)$ into contributions $C_1,\ldots,C_K$. **Efficiency** requires

$$\sum_{k=1}^K C_k=\Delta_T.$$

Efficiency is an accounting property. It does not imply that $C_k$ is unique, consistently estimated, policy-invariant, or causal. Identification asks whether the observed-data law determines the target. Estimation asks how a sample is used to approximate it. Attribution asks how a known total is allocated. These are different operations.


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from itertools import permutations, combinations
pd.options.display.float_format = '{:,.4f}'.format
plt.style.use('seaborn-v0_8-whitegrid')
rng = np.random.default_rng(42)
```


```python
taxonomy = pd.DataFrame({
    'method': ['Kitagawa', 'Oaxaca–Blinder', 'LMDI', 'Shapley', 'SHAP',
               'Difference-in-Differences'],
    'decomposes': ['rate change', 'mean gap', 'aggregate change', 'value change',
                   'prediction', 'potential-outcome contrast'],
    'causal_by_itself': [False, False, False, False, False, False],
    'key_choice': ['two-period average', 'reference coefficients', 'log-mean path',
                   'coalition value', 'background distribution', 'parallel trends'],
})
taxonomy
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>method</th>
      <th>decomposes</th>
      <th>causal_by_itself</th>
      <th>key_choice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Kitagawa</td>
      <td>rate change</td>
      <td>False</td>
      <td>two-period average</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Oaxaca–Blinder</td>
      <td>mean gap</td>
      <td>False</td>
      <td>reference coefficients</td>
    </tr>
    <tr>
      <th>2</th>
      <td>LMDI</td>
      <td>aggregate change</td>
      <td>False</td>
      <td>log-mean path</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Shapley</td>
      <td>value change</td>
      <td>False</td>
      <td>coalition value</td>
    </tr>
    <tr>
      <th>4</th>
      <td>SHAP</td>
      <td>prediction</td>
      <td>False</td>
      <td>background distribution</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Difference-in-Differences</td>
      <td>potential-outcome contrast</td>
      <td>False</td>
      <td>parallel trends</td>
    </tr>
  </tbody>
</table>
</div>



## How to read the taxonomy output

Each row is defined by the **object decomposed**, not by industry. The `causal_by_itself` column is false even for Difference-in-Differences because an estimator name alone is insufficient: DiD becomes causally interpretable only when parallel trends and the rest of its design assumptions are credible. The final column identifies the choice or assumption that an analyst must make visible.

## Econometrics point of view

An econometric analysis should declare an estimand before choosing an estimator. A decomposition can operate on observed statistics, fitted conditional means, counterfactual distributions, or identified causal effects. Only the last case inherits a causal interpretation, and only under the assumptions that identify those effects.

| Layer | Example target | Main uncertainty | Defensible verb |
|---|---|---|---|
| accounting | $R_1-R_0$ | measurement/revisions | contributed |
| descriptive statistical | $T(\hat P_1)-T(\hat P_0)$ | sampling | associated/composed |
| predictive | $f(x)-E[f(X)]$ | generalization/model | predicted/attributed |
| causal | $E[Y(1)-Y(0)]$ | identification + sampling | caused |

### Growth-marketing use case

For $\text{Revenue}=\text{Traffic}\times\text{CVR}\times\text{AOV}$, attribution answers how the realized revenue change is distributed across these factors. It does **not** answer what revenue would have been if a campaign manager had intervened on traffic while holding the rest of the system at its post-intervention equilibrium.

## Limitations and failure modes

- **Non-uniqueness:** interactions admit multiple exact allocations.
- **Baseline dependence:** changing period, control group, reference model, or background distribution changes contributions.
- **Aggregation bias:** segment-level decompositions may reverse after aggregation (Simpson-type behavior).
- **Generated-regressor uncertainty:** fitted rates, coefficients, or propensities must carry estimation error forward.
- **No policy invariance:** accounting identities need not survive interventions because other factors respond.
- **Semantic overreach:** “driver” often conflates arithmetic contribution, predictive relevance, and causal effect.

## What came next

Kitagawa (1955) was generalized to multiple factors by **Prithwis Das Gupta (1978, 1993)**. Mean decompositions by **Oaxaca and Blinder (1973)** were extended from means toward counterfactual distributions by **Juhn, Murphy & Pierce (1993)** and **DiNardo, Fortin & Lemieux (1996)**, and toward unconditional quantiles by **Firpo, Fortin & Lemieux (2009)**. Cooperative-game allocations developed by **Shapley (1953)** later informed **Shorrocks (2013)** for distributional decomposition and **Lundberg & Lee (2017)** for model explanations. These are convergences around allocation principles, not one linear genealogy.

## Takeaways and bridge to Notebook 01

1. Always name the target, contrast, and allocation rule.
2. Exact contributions need not be unique or causal.
3. Choose a method by the mathematical object, not by dashboard terminology.

Notebook 01 now specializes this framework to the simplest important object: an aggregate rate formed as a weighted average of segment rates.

### Exercise

For a statement such as “revenue was driven by price,” identify the total, comparison, allocation rule, uncertainty, and causal assumptions. Rewrite it as a defensible descriptive claim.

## Interpretation checklist

1. State the mathematical identity or estimand.
2. Verify exactness numerically.
3. Separate description, prediction, and causation.
4. Report reference population/path/order.
5. Quantify sampling uncertainty when inputs are estimated.

## References

- Kitagawa, E. M. (1955). Components of a difference between two rates. *JASA*, 50, 1168–1194. https://doi.org/10.1080/01621459.1955.10501299
- Das Gupta, P. (1993). *Standardization and Decomposition of Rates*. U.S. Census Bureau.
- Shorrocks, A. F. (2013). Decomposition procedures for distributional analysis. *Journal of Economic Inequality*, 11, 99–126. https://doi.org/10.1007/s10888-011-9214-z
- Pearl, J. (2009). *Causality* (2nd ed.). Cambridge University Press.
- Juhn, C., Murphy, K. M., & Pierce, B. (1993). Wage inequality and the rise in returns to skill. *Journal of Political Economy*, 101, 410–442. https://doi.org/10.1086/261881
- Fortin, N., Lemieux, T., & Firpo, S. (2011). Decomposition methods in economics. In *Handbook of Labor Economics*, Vol. 4A, 1–102. https://doi.org/10.1016/S0169-7218(11)00407-2
- Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *NeurIPS 30*, 4765–4774. https://papers.nips.cc/paper/7062
