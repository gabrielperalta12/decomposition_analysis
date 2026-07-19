# 01 — Rate decomposition: Kitagawa, Das Gupta, and replacement

**Graduate course: Decomposition Analysis in Python**

## Learning objectives

- Derive the two-factor rate decomposition.
- Recognize path dependence with three or more factors.
- Implement symmetric and stepwise allocations.

## Roadmap and notation

### Guiding question

When an aggregate conversion rate changes, how much is associated with a different mix of users and how much with different conversion rates inside the same segments?

### Prerequisites

Weighted averages and percentage-point changes. Notebook 00's distinction between contribution and causation is assumed.

### Symbols

| Symbol | Meaning |
|---|---|
| $g$ | segment, such as channel or customer type |
| $t\in\{0,1\}$ | baseline and comparison periods |
| $w_{gt}$ | share of observations in segment $g$; $\sum_gw_{gt}=1$ |
| $r_{gt}$ | rate inside segment $g$ |
| $R_t$ | aggregate rate $\sum_gw_{gt}r_{gt}$ |
| $C_w,C_r$ | mix and within-segment-rate contributions |

We first establish the identity, then calculate it, interpret the output, and finally test whether the answer changes when segments are aggregated.

## Setup

For group $g$, the aggregate rate is $R_t=\sum_g w_{gt}r_{gt}$. Kitagawa's symmetric allocation is
$$C_w=\sum_g(w_{g1}-w_{g0})(r_{g1}+r_{g0})/2,$$
$$C_r=\sum_g(r_{g1}-r_{g0})(w_{g1}+w_{g0})/2.$$
Then $C_w+C_r=R_1-R_0$ exactly. These are composition and within-group-rate components—not causal effects. Das Gupta generalized standardization to several factors. Stepwise replacement is exact but generally order-dependent; averaging all orders produces a Shapley-style symmetric allocation.

## Derivation and interaction allocation

For one segment, write $w_1=w_0+\Delta w$ and $r_1=r_0+\Delta r$:

$$\Delta(wr)=r_0\Delta w+w_0\Delta r+\Delta w\Delta r.$$

Kitagawa splits the interaction equally:

$$C_w=\Delta w\left(r_0+\frac{\Delta r}{2}\right),\qquad
C_r=\Delta r\left(w_0+\frac{\Delta w}{2}\right).$$

Summing over $g$ yields exactness. The half-interaction rule is symmetric under exchanging periods, but it is an allocation convention—not an empirical discovery. With $K$ changing factors, stepwise replacement produces $K!$ paths; averaging their marginal increments is order-invariant and closely related to the Shapley value.


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
# One row per stable customer segment; shares sum to one in each period.
d = pd.DataFrame({
    'segment': ['New', 'Returning', 'Enterprise'],
    'w0': [.50, .35, .15],
    'w1': [.42, .38, .20],
    'r0': [.08, .18, .31],
    'r1': [.10, .17, .34],
})

# Kitagawa's symmetric mix and rate contributions, calculated by segment.
d['mix'] = (d.w1 - d.w0) * (d.r1 + d.r0) / 2
d['rate'] = (d.r1 - d.r0) * (d.w1 + d.w0) / 2

R0 = (d.w0 * d.r0).sum()
R1 = (d.w1 * d.r1).sum()
summary = pd.Series({
    'R0': R0,
    'R1': R1,
    'change': R1 - R0,
    'mix': d['mix'].sum(),
    'rate': d['rate'].sum(),
    'error': (R1 - R0) - d[['mix', 'rate']].to_numpy().sum(),
})
d, summary
```




    (      segment     w0     w1     r0     r1     mix    rate
     0         New 0.5000 0.4200 0.0800 0.1000 -0.0072  0.0092
     1   Returning 0.3500 0.3800 0.1800 0.1700  0.0053 -0.0036
     2  Enterprise 0.1500 0.2000 0.3100 0.3400  0.0163  0.0053,
     R0        0.1495
     R1        0.1746
     change    0.0251
     mix       0.0143
     rate      0.0108
     error    -0.0000
     dtype: float64)




```python
segment_contributions = d.set_index('segment')[['mix', 'rate']]
ax = segment_contributions.plot.bar(color=['#4C78A8', '#F58518'])
ax.axhline(0, color='black', linewidth=.8)
ax.set_ylabel('contribution to aggregate rate change')
plt.show()
```


    
![png](01_rates_kitagawa_dasgupta_files/01_rates_kitagawa_dasgupta_6_0.png)
    


## Reading the worked example

The aggregate rate rises from 0.1495 to 0.1746, a change of **0.0251**, or **2.51 percentage points**. The method allocates 1.43 pp to mix and 1.08 pp to within-segment rates. The near-zero `error` verifies the identity numerically.

At segment level, New users contribute negatively through mix because their traffic share falls, but positively through rate because their CVR improves. Enterprise contributes positively through both channels. Positive does not mean “good intervention”; it means positive contribution to the observed aggregate change under this rule.

## Growth-marketing case: conversion rate

Let $w_{gt}$ be the traffic share of channel $g$ and $r_{gt}$ its conversion rate. Then total CVR is $R_t=\sum_g w_{gt}r_{gt}$.

- **Mix contribution:** traffic moved toward channels with higher/lower CVR.
- **Rate contribution:** within-channel CVR changed.
- **Actionable diagnostic:** split further by device, geography, landing page, or cohort, but check sparse cells and post-treatment segmentation.

Example claim: “Total CVR rose 1.18 pp; under the Kitagawa two-period rule, +0.34 pp is allocated to channel mix and +0.84 pp to within-channel rates.” This is precise and descriptive. “The new campaign caused +0.84 pp” is not supported without an experiment or credible quasi-experiment.


```python
# Sensitivity to aggregation: collapse Returning and Enterprise
fine = d.copy()
fine['n0'] = fine.w0 * 10_000
fine['n1'] = fine.w1 * 12_000
collapsed = pd.DataFrame({
    'segment':['New','Established'],
    'w0':[fine.loc[0,'w0'], fine.loc[1:,'w0'].sum()],
    'w1':[fine.loc[0,'w1'], fine.loc[1:,'w1'].sum()],
    'r0':[fine.loc[0,'r0'], np.average(fine.loc[1:,'r0'], weights=fine.loc[1:,'w0'])],
    'r1':[fine.loc[0,'r1'], np.average(fine.loc[1:,'r1'], weights=fine.loc[1:,'w1'])],
})
collapsed['mix']=(collapsed.w1-collapsed.w0)*(collapsed.r1+collapsed.r0)/2
collapsed['rate']=(collapsed.r1-collapsed.r0)*(collapsed.w1+collapsed.w0)/2
pd.DataFrame({'fine':[d['mix'].sum(),d['rate'].sum()],
              'collapsed':[collapsed['mix'].sum(),collapsed['rate'].sum()]},
             index=['mix','rate'])
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
      <th>fine</th>
      <th>collapsed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>mix</th>
      <td>0.0143</td>
      <td>0.0107</td>
    </tr>
    <tr>
      <th>rate</th>
      <td>0.0108</td>
      <td>0.0144</td>
    </tr>
  </tbody>
</table>
</div>



## Limitations, robustness, and inference

- Shares must sum to one in each period and segments must be defined consistently.
- New/disappearing categories and small cells require explicit handling.
- Results depend on segmentation; always repeat at plausible aggregation levels.
- Observed rate changes combine treatment, seasonality, selection, composition within cells, and noise.
- For estimated rates, use a stratified bootstrap or delta method; exactness conditional on estimates is not zero sampling variance.
- If channel is affected by treatment, conditioning on it can create post-treatment bias in a causal analysis.

## What came next

**Das Gupta (1978)** extended the logic to several compositional factors. **Chevan & Sutherland (2009)** revisited Das Gupta and supplied a general algorithmic treatment. In applications with many interacting factors, averaging stepwise-replacement orders leads naturally to the axiomatic decomposition presented by **Shorrocks (2013)**.

## Takeaways and bridge to Notebook 02

1. Kitagawa separates mix from within-segment rate changes exactly.
2. The equal interaction split is symmetric but conventional.
3. Segmentation and sampling uncertainty can materially change the story.
4. Use causal language only with a separate design.

Notebook 02 moves from weighted rates to totals generated by multiplicative business identities.

### Exercise

Bootstrap observations within segments, recompute the decomposition, and form percentile intervals. Why is algebraic exactness not the same as statistical certainty?

## Interpretation checklist

1. State the mathematical identity or estimand.
2. Verify exactness numerically.
3. Separate description, prediction, and causation.
4. Report reference population/path/order.
5. Quantify sampling uncertainty when inputs are estimated.

## References

- Kitagawa, E. M. (1955). *JASA*, 50, 1168–1194. https://doi.org/10.1080/01621459.1955.10501299
- Das Gupta, P. (1978). A general method of decomposing a difference between two rates into several components. *Demography*, 15, 99–112. https://doi.org/10.2307/2060493
- Chevan, A., & Sutherland, M. (2009). Revisiting Das Gupta. *Demography*, 46, 429–449. https://doi.org/10.1353/dem.0.0051
- Das Gupta, P. (1993). *Standardization and Decomposition of Rates: A User's Manual*. U.S. Census Bureau.
- Shorrocks, A. F. (2013). Decomposition procedures for distributional analysis. *Journal of Economic Inequality*, 11, 99–126. https://doi.org/10.1007/s10888-011-9214-z
