# 05 — Econometrics, causal inference, and defensible driver claims

**Graduate course: Decomposition Analysis in Python**

## Learning objectives

- Distinguish contribution, association, prediction, and causal effect.
- Demonstrate confounding in a regression.
- Build a KPI driver tree without overstating causality.

## Roadmap and notation

### Guiding question

What evidence is required to move from “this factor contributed to the observed change” to “changing this factor would change the outcome”?

### Prerequisites

Basic regression and Notebook 00's distinction between descriptive, predictive, and causal claims.

### Symbols

- $D\in\{0,1\}$: treatment or intervention indicator.
- $Y(1),Y(0)$: potential outcomes under treatment and control.
- $X$: pre-treatment covariates.
- $ATE$: average treatment effect; $ATT$: effect for treated units.
- $do(X=x)$: intervention notation in a structural causal model.

The simulated example makes confounding visible. The method table then links practical growth questions to the assumptions needed for causal identification.

## Different questions, different estimands

An identity decomposition explains how an observed arithmetic change is allocated. Regression estimates conditional associations unless a design and assumptions identify a causal estimand. Potential-outcome methods target contrasts such as $E[Y(1)-Y(0)]$; structural causal models encode interventions $do(X=x)$.

| Claim | Minimum object | Typical requirement |
|---|---|---|
| contribution | identity/value function | explicit allocation rule |
| association | joint distribution/model | specification and sampling assumptions |
| prediction | out-of-sample loss | representative validation |
| causal effect | potential outcomes/SCM | exchangeability or valid design |

OLS, fixed effects, DiD, synthetic control, matching, IV, RD, DML, causal forests, and BSTS do not become causal merely by name. Each needs its own identifying assumptions. A driver tree is a semantic/accounting graph; it becomes causal only when its edges are supported by a causal model and identification strategy.

## Econometric estimands behind common methods

For treatment $D$, outcome $Y$, and potential outcomes $Y(1),Y(0)$,

$$ATE=E[Y(1)-Y(0)],\qquad ATT=E[Y(1)-Y(0)\mid D=1].$$

Under conditional exchangeability, $(Y(1),Y(0))\perp D\mid X$, positivity, and consistency,

$$ATE=E_X\{E[Y\mid D=1,X]-E[Y\mid D=0,X]\}.$$

DiD instead uses parallel trends; IV identifies a local effect under relevance, independence, exclusion, and monotonicity; RD identifies a local cutoff effect under continuity/no precise manipulation. Fixed effects remove time-invariant additive confounding, not time-varying confounding. DML reduces regularization bias through orthogonal scores and cross-fitting, but still requires causal identification.


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
n=3000
quality=rng.normal(size=n); campaign=(rng.random(n)<1/(1+np.exp(-1.4*quality))).astype(int)
sales=10+3*quality+1.5*campaign+rng.normal(size=n)
naive=sm.OLS(sales,sm.add_constant(campaign)).fit()
adjusted=sm.OLS(sales,sm.add_constant(np.c_[campaign,quality])).fit()
pd.Series({'true treatment effect':1.5,'naive association':naive.params[1],'adjusted coefficient':adjusted.params[1]})
```




    true treatment effect   1.5000
    naive association       4.5556
    adjusted coefficient    1.4338
    dtype: float64




```python
criteria=pd.DataFrame({
'question':['What changed arithmetically?','What predicts Y?','What would Y be under intervention?'],
'tool':['LMDI/PVM/Shapley','validated ML/regression','RCT, DiD, IV, RD, SCM, etc.'],
'safe wording':['contributed under rule R','predictively associated','caused under assumptions A']})
criteria
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
      <th>question</th>
      <th>tool</th>
      <th>safe wording</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>What changed arithmetically?</td>
      <td>LMDI/PVM/Shapley</td>
      <td>contributed under rule R</td>
    </tr>
    <tr>
      <th>1</th>
      <td>What predicts Y?</td>
      <td>validated ML/regression</td>
      <td>predictively associated</td>
    </tr>
    <tr>
      <th>2</th>
      <td>What would Y be under intervention?</td>
      <td>RCT, DiD, IV, RD, SCM, etc.</td>
      <td>caused under assumptions A</td>
    </tr>
  </tbody>
</table>
</div>



## Reading the simulation

The data-generating process sets the treatment effect to 1.5. The naive regression estimates about 4.56 because high-quality units are more likely to receive the campaign and quality also raises sales. After controlling for the simulated confounder, the coefficient is about 1.43; the remaining difference from 1.5 is ordinary sampling noise.

This is a teaching example in which the confounder is observed and correctly modeled. In real growth data, adjustment is credible only if the required confounders are measured, pre-treatment, and modeled with adequate overlap. A close adjusted estimate in this simulation is not a general endorsement of regression adjustment.

## Growth-marketing designs

| Question | Preferred design | Key threat |
|---|---|---|
| incremental conversions from ads | geo/user randomized holdout | interference, noncompliance |
| lifecycle email effect | randomized send/holdout | triggered eligibility, spillovers |
| bid-policy rollout | staggered experiment or credible DiD | heterogeneous timing, anticipation |
| threshold-based offer | RD | manipulation around cutoff |
| channel incrementality with auction instrument | IV only if exclusion is credible | direct effects of instrument |
| heterogeneous treatment effects | causal forest after identification | overlap, multiple testing |

A driver tree helps define measurement identities and candidate interventions. A DAG documents confounders, mediators, colliders, and selection. An experiment or identification strategy estimates causal edges. Keep these artifacts connected but conceptually separate.


```python
# Omitted-variable-bias sensitivity in the simulation
rows=[]
for strength in [0.0,.5,1.0,1.4,2.0]:
    q=rng.normal(size=n)
    d_=(rng.random(n)<1/(1+np.exp(-strength*q))).astype(int)
    y_=10+3*q+1.5*d_+rng.normal(size=n)
    naive_=sm.OLS(y_,sm.add_constant(d_)).fit().params[1]
    adjusted_=sm.OLS(y_,sm.add_constant(np.c_[d_,q])).fit().params[1]
    rows.append((strength,naive_,adjusted_))
pd.DataFrame(rows,columns=['selection_strength','naive','adjusted']).set_index('selection_strength')
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
      <th>naive</th>
      <th>adjusted</th>
    </tr>
    <tr>
      <th>selection_strength</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0.0000</th>
      <td>1.4856</td>
      <td>1.5268</td>
    </tr>
    <tr>
      <th>0.5000</th>
      <td>2.9417</td>
      <td>1.4998</td>
    </tr>
    <tr>
      <th>1.0000</th>
      <td>4.0162</td>
      <td>1.5356</td>
    </tr>
    <tr>
      <th>1.4000</th>
      <td>4.5926</td>
      <td>1.4432</td>
    </tr>
    <tr>
      <th>2.0000</th>
      <td>5.0706</td>
      <td>1.4849</td>
    </tr>
  </tbody>
</table>
</div>



## Limitations and robustness by design

- **OLS/GLM:** functional form, omitted variables, measurement error, simultaneity; use residual diagnostics and sensitivity analysis, not causal language by default.
- **Panel fixed effects:** remaining time-varying confounding, dynamic bias, clustered dependence; cluster at the assignment/shock level.
- **DiD:** inspect pre-trends, treatment timing, anticipation, spillovers, and negative weighting under heterogeneous effects.
- **Matching/propensity scores:** cannot fix unmeasured confounding; check overlap and covariate balance, not propensity-model fit alone.
- **IV:** weak instruments and exclusion violations; report first stage and weak-IV-robust inference.
- **RD:** bandwidth/specification sensitivity, manipulation, and local external validity.
- **DML/causal forests:** overlap and identification still dominate; cross-fitting is not a cure for bad controls.
- **BSTS/synthetic control:** donor contamination, post-selection, unstable pre-period fit, and limited placebo units.

## What came next

**Rubin (1974)** formalized potential-outcome reasoning; **Rosenbaum & Rubin (1983)** developed propensity-score design. **Pearl (1995, 2009)** formalized causal graphs and intervention calculus. Modern robust estimation includes **Imbens & Lemieux (2008)** for RD practice, **Abadie, Diamond & Hainmueller (2010)** for synthetic control, **Chernozhukov et al. (2018)** for DML, and **Athey, Tibshirani & Wager (2019)** for generalized random forests. For staggered DiD with heterogeneous effects, **Callaway & Sant'Anna (2021)** and **Sun & Abraham (2021)** repair failures of naive two-way fixed-effects summaries.

## Final takeaways

1. Contribution, association, prediction, and causation are different claims.
2. A driver tree organizes identities and hypotheses; it does not identify causal edges.
3. Every causal estimator depends on design-specific assumptions.
4. Growth decisions should combine descriptive monitoring, predictive validation, and causal experimentation without conflating their outputs.

The recommended final deliverable is three coordinated artifacts: a decomposition for monitoring, a predictive model for forecasting or targeting, and a causal design for intervention decisions.

### Capstone

For a conversion-rate decline, produce (i) a Kitagawa mix/rate decomposition, (ii) a predictive model with held-out error, and (iii) a causal design for one actionable lever. Keep the three conclusions separate.

## Interpretation checklist

1. State the mathematical identity or estimand.
2. Verify exactness numerically.
3. Separate description, prediction, and causation.
4. Report reference population/path/order.
5. Quantify sampling uncertainty when inputs are estimated.

## References

- Rubin, D. B. (1974). Estimating causal effects of treatments in randomized and nonrandomized studies. *Journal of Educational Psychology*, 66, 688–701. https://doi.org/10.1037/h0037350
- Pearl, J. (2009). *Causality* (2nd ed.). Cambridge University Press.
- Angrist, J. D., & Pischke, J.-S. (2009). *Mostly Harmless Econometrics*. Princeton University Press.
- Chernozhukov, V. et al. (2018). Double/debiased machine learning. *The Econometrics Journal*, 21, C1–C68. https://doi.org/10.1111/ectj.12097
- Athey, S., & Imbens, G. W. (2016). Recursive partitioning for heterogeneous causal effects. *PNAS*, 113, 7353–7360. https://doi.org/10.1073/pnas.1510489113
- Rosenbaum, P. R., & Rubin, D. B. (1983). The central role of the propensity score in observational studies for causal effects. *Biometrika*, 70, 41–55. https://doi.org/10.1093/biomet/70.1.41
- Imbens, G. W., & Lemieux, T. (2008). Regression discontinuity designs: A guide to practice. *Journal of Econometrics*, 142, 615–635. https://doi.org/10.1016/j.jeconom.2007.05.001
- Abadie, A., Diamond, A., & Hainmueller, J. (2010). Synthetic control methods for comparative case studies. *JASA*, 105, 493–505. https://doi.org/10.1198/jasa.2009.ap08746
- Athey, S., Tibshirani, J., & Wager, S. (2019). Generalized random forests. *Annals of Statistics*, 47, 1148–1178. https://doi.org/10.1214/18-AOS1709
- Callaway, B., & Sant'Anna, P. H. C. (2021). Difference-in-differences with multiple time periods. *Journal of Econometrics*, 225, 200–230. https://doi.org/10.1016/j.jeconom.2020.12.001
- Sun, L., & Abraham, S. (2021). Estimating dynamic treatment effects in event studies with heterogeneous treatment effects. *Journal of Econometrics*, 225, 175–199. https://doi.org/10.1016/j.jeconom.2020.09.006
