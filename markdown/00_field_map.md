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

## Descriptive versus causal counterfactuals

This distinction is subtle because both approaches construct a world that was not observed exactly as stated. The decisive difference is not the word *counterfactual*. It is the **scientific question**, the **meaning of the hypothetical world**, and the assumptions required to connect that world to reality.

### 1. A descriptive counterfactual: hold a component fixed

Suppose

$$\text{Sales}=\text{Traffic}\times\text{Conversion rate}.$$

The observed periods are

| Period | Traffic | Conversion rate | Sales |
|---|---:|---:|---:|
| 0 | 1,000 | 10% | 100 |
| 1 | 1,200 | 10% | 120 |

A decomposition may construct the hybrid scenario

$$\widetilde Y=1{,}000\times 10\%=100,$$

which combines baseline traffic with the comparison-period conversion rate. Because conversion did not change, all 20 additional sales are allocated to traffic.

This hypothetical answers:

> How would the accounting identity evaluate if one observed component were held at its baseline value while the other component took its comparison-period value?

Everything used to construct the hybrid state—1,000 visits and a 10% conversion rate—was observed somewhere in the two periods. The analyst is rearranging observed factor values inside a known identity. No treatment, assignment mechanism, intervention, or missing potential outcome has been specified.

The conclusion is therefore descriptive:

> Under this decomposition rule, the observed increase of 20 sales is allocated entirely to the traffic component.

It does **not** imply that a campaign caused traffic to increase, that increasing traffic deliberately would leave conversion fixed, or that another 200 visits would generate exactly 20 incremental sales under intervention.

### 2. A causal counterfactual: compare interventions

Now define $A=1$ as launching a campaign and $A=0$ as not launching it. Let

$$Y(1)=\text{sales under the campaign},\qquad
Y(0)=\text{sales without the campaign}.$$

For the same business and period, the causal effect is

$$\tau=Y(1)-Y(0).$$

If the campaign was launched, we observe $Y=Y(1)=120$ but do not observe $Y(0)$. If it was not launched, we observe $Y(0)$ but not $Y(1)$. This is the fundamental problem of causal inference: for the same unit at the same time, only one potential outcome is observed.

The causal question is:

> How many sales would this same business have generated in the same period under the alternative intervention—no campaign?

Unlike the descriptive hybrid, $Y(0)$ is not obtained merely by substituting an observed traffic value into the sales identity. Without the campaign, competitor behavior, channel auctions, user composition, conversion, prices, and seasonality might all differ. The counterfactual must be identified using a research design and assumptions—for example randomization, conditional exchangeability, parallel trends, an instrument, or a credible synthetic control.

### 3. What is held fixed?

| Dimension | Descriptive decomposition | Causal inference |
|---|---|---|
| Main question | How is an observed difference allocated? | What would change under an intervention? |
| Hypothetical object | Hybrid combination of factor values | Potential outcome or interventional distribution |
| Typical notation | $F(x_{11},x_{20})$ | $Y(1),Y(0)$ or $P(Y\mid do(A=a))$ |
| What is held fixed | Components by accounting convention | Unit/population definition and intervention contrast |
| Other variables | Often fixed mechanically or changed by a path rule | May respond downstream to the intervention |
| Missing-data problem | Usually no missing potential outcome is defined | One potential outcome per unit is fundamentally unobserved |
| Primary requirement | Valid identity and explicit allocation rule | Identification assumptions plus an estimator |
| Typical conclusion | “Traffic contributed 20 sales under rule $R$” | “The campaign caused $\tau$ incremental sales under assumptions $A$” |

The phrase “same business” in a causal question does not mean every measured variable is frozen. It means we compare well-defined interventions for the same target unit or population. Mediators such as traffic and conversion may change as part of the treatment's total effect.

### 4. Netflix example

Observed watch hours increase from 100 to 120 after an algorithm change.

A descriptive decomposition might allocate the increase as:

- +8 hours from more active users;
- +7 hours from more time per active user;
- +5 hours from better observed retention.

This exactly accounts for the 20-hour change under a stated driver identity or allocation rule. It does not show that the algorithm produced any of those component changes.

The causal estimand instead compares

$$E[Y(\text{new algorithm})-Y(\text{old algorithm})].$$

Estimating it requires the missing outcome that would have occurred for comparable users under the algorithm not received. An A/B test could identify this contrast under random assignment, consistency, limited interference, and correct outcome measurement.

### 5. Pricing example

A CFO asks: “Which components account for the profit increase?” A Price–Volume–Mix decomposition can allocate the observed difference to price, volume, mix, cost, and interactions.

A CEO asks: “Did our new pricing strategy cause the profit increase?” This requires

$$\text{Profit}(\text{new policy})-\text{Profit}(\text{old policy}).$$

Holding observed volume fixed while substituting the new price is not generally the causal counterfactual, because demand, mix, competitor prices, and customer selection can respond to price. The descriptive price component is therefore not a price elasticity or a policy effect.

### 6. A practical classification rule

When a paper says “counterfactual,” ask:

1. **Is this a hybrid scenario formed by replacing or holding fixed components inside an identity or fitted model?**  
   It is primarily a descriptive attribution unless additional causal structure is supplied.
2. **Is this an alternative world defined by a treatment or intervention?**  
   It is a causal estimand only if the intervention, target population, identification assumptions, and estimator are explicit.
3. **Could downstream variables respond to the intervention?**  
   If yes, mechanically freezing them may block part of the effect or create an incoherent intervention.
4. **What evidence identifies the unobserved world?**  
   If the answer is only “the decomposition is exact,” causality has not been established.

### 7. Important nuance

Some modern decomposition methods use statistically estimated counterfactual distributions—for example reweighting one group's covariate distribution to resemble another's. These are more than simple accounting substitutions, but they are still not automatically causal. Their interpretation depends on whether the reweighted distribution is merely descriptive or is identified as the distribution that would arise under a well-defined intervention.

Therefore, “descriptive” does not mean useless or unsophisticated. A descriptive counterfactual can be exact, decision-relevant, and mathematically rigorous. It simply answers a different question from a causal counterfactual.

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
3. A descriptive hybrid holds factors fixed by convention; a causal counterfactual compares well-defined interventions and requires identification.
4. Choose a method by the mathematical object, not by dashboard terminology.

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
- Holland, P. W. (1986). Statistics and causal inference. *JASA*, 81, 945–960. https://doi.org/10.1080/01621459.1986.10478354
- Imbens, G. W., & Rubin, D. B. (2015). *Causal Inference for Statistics, Social, and Biomedical Sciences*. Cambridge University Press. https://doi.org/10.1017/CBO9781139025751
- Hernán, M. A., & Robins, J. M. (2020). *Causal Inference: What If*. Chapman & Hall/CRC. https://www.hsph.harvard.edu/miguel-hernan/causal-inference-book/
