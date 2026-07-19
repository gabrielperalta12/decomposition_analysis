# 03 — Oaxaca–Blinder and distributional decompositions

**Graduate course: Decomposition Analysis in Python**

## Learning objectives

- Decompose a mean gap into composition and coefficient terms.
- See why the reference structure matters.
- Connect reweighting and RIF regressions to distributional statistics.

## Roadmap and notation

### Guiding question

When two groups have different average outcomes, how much of the gap is associated with different observed characteristics and how much with different fitted outcome structures?

### Prerequisites

Linear regression and sample means. Familiarity with causal inference is helpful but not required.

### Symbols

| Symbol | Meaning |
|---|---|
| $A,B$ | comparison groups |
| $Y$ | outcome, for example LTV or log wage |
| $X$ | observed characteristics, including a constant |
| $\bar X_g$ | group-$g$ covariate means |
| $\hat\beta_g$ | group-$g$ OLS coefficients |
| $\beta^*$ | chosen reference coefficient structure |

The decomposition is first an algebraic identity built from fitted regressions. A causal reading requires additional identification assumptions that are examined later.

## Mean-gap decomposition

With group means $\bar X_A,\bar X_B$ and linear coefficients $\hat\beta_A,\hat\beta_B$,
$$\bar Y_A-\bar Y_B=(\bar X_A-\bar X_B)'\hat\beta_B+\bar X_A'(\hat\beta_A-\hat\beta_B).$$
The first term is often called composition/endowment and the second structure/coefficient. Labels such as “explained” and “unexplained” do not establish explanation, discrimination, or causality. Results depend on the reference coefficients, included covariates, functional form, common support, and selection.

DiNardo–Fortin–Lemieux reweights an entire distribution using density ratios. Firpo–Fortin–Lemieux regress the recentered influence function (RIF) to decompose unconditional quantiles and other distributional statistics.

## Threefold decomposition and identification

Adding and subtracting $\bar X_B'\hat\beta_A$ yields a threefold form:

$$\bar Y_A-\bar Y_B=
(\bar X_A-\bar X_B)'\hat\beta_B
+\bar X_B'(\hat\beta_A-\hat\beta_B)
+(\bar X_A-\bar X_B)'(\hat\beta_A-\hat\beta_B).$$

These are endowment, coefficient, and interaction terms. The twofold form chooses a nondiscriminatory/reference coefficient vector $\beta^*$:

$$\Delta=(\bar X_A-\bar X_B)'\beta^*
+\bar X_A'(\hat\beta_A-\beta^*)
+\bar X_B'(\beta^*-\hat\beta_B).$$

The reference choice is not innocuous. A causal counterfactual additionally needs overlap, consistency, and a conditional exchangeability or structural assumption; OB algebra alone supplies none of these.


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
import statsmodels.api as sm
n=1500; g=rng.binomial(1,.48,n); experience=np.clip(rng.normal(8+2*g,3,n),0,None); education=rng.normal(15+.7*g,1.5,n)
y=2+0.09*experience+0.14*education+g*(.18+.025*experience)+rng.normal(0,.45,n)
df=pd.DataFrame({'y':y,'group':g,'experience':experience,'education':education})
fits={k:sm.OLS(z.y,sm.add_constant(z[['experience','education']])).fit() for k,z in df.groupby('group')}
xbar={k:np.r_[1,z[['experience','education']].mean()] for k,z in df.groupby('group')}
b0,b1=fits[0].params.to_numpy(),fits[1].params.to_numpy(); gap=df[df.group==1].y.mean()-df[df.group==0].y.mean()
composition=(xbar[1]-xbar[0])@b0; structure=xbar[1]@(b1-b0)
pd.Series({'mean gap':gap,'composition (group 0 reference)':composition,'structure':structure,'identity error':gap-composition-structure})
```




    mean gap                          0.7113
    composition (group 0 reference)   0.2661
    structure                         0.4451
    identity error                    0.0000
    dtype: float64




```python
comp_alt=(xbar[1]-xbar[0])@b1; struct_alt=xbar[0]@(b1-b0)
pd.DataFrame({'group 0 reference':[composition,structure], 'group 1 reference':[comp_alt,struct_alt]}, index=['composition','structure'])
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
      <th>group 0 reference</th>
      <th>group 1 reference</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>composition</th>
      <td>0.2661</td>
      <td>0.3008</td>
    </tr>
    <tr>
      <th>structure</th>
      <td>0.4451</td>
      <td>0.4104</td>
    </tr>
  </tbody>
</table>
</div>



## Reading the worked example

The first output verifies that the mean gap equals composition plus structure under group 0's coefficients. The second output changes the reference to group 1. Both decompositions reproduce the same total gap, but the allocation changes. This is the index-number problem in observable form.

The correct conclusion is therefore conditional: “using group 0 (or group 1) as the reference response structure.” The structure component should not be renamed discrimination, campaign quality, or treatment effect without a separate identification argument.

## Growth-marketing case: cohort LTV and conversion gaps

Compare organic ($A$) and paid-social ($B$) users. Let $Y$ be 90-day LTV or activation, and $X$ include device, country, signup week, acquisition creative, and pre-acquisition intent proxies.

- The composition term asks how much of the mean gap is associated with different observed user mixes under a reference response surface.
- The structure term captures different fitted mappings from $X$ to $Y$ plus omitted variables, misspecification, selection, and potentially treatment effects.
- Do not call structure “campaign quality” or “incrementality.” Paid acquisition changes who appears in the sample and may change downstream experiences.

For binary conversion, linear probability OB is transparent but may predict outside $[0,1]$; nonlinear decompositions require an averaging/path rule.


```python
# Bootstrap uncertainty for the twofold OB components
def ob_once(data):
    fs={k:sm.OLS(z.y,sm.add_constant(z[['experience','education']])).fit()
        for k,z in data.groupby('group')}
    xb={k:np.r_[1,z[['experience','education']].mean()] for k,z in data.groupby('group')}
    b0_,b1_=fs[0].params.to_numpy(),fs[1].params.to_numpy()
    return np.array([(xb[1]-xb[0])@b0_, xb[1]@(b1_-b0_)])
boots=[]
for _ in range(200):
    sample=pd.concat([z.sample(len(z),replace=True,random_state=int(rng.integers(1e9)))
                      for _,z in df.groupby('group')])
    boots.append(ob_once(sample))
ci=pd.DataFrame(np.quantile(boots,[.025,.5,.975],axis=0).T,
                index=['composition','structure'],columns=['p2.5','median','p97.5'])
ci
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
      <th>p2.5</th>
      <th>median</th>
      <th>p97.5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>composition</th>
      <td>0.2240</td>
      <td>0.2657</td>
      <td>0.3095</td>
    </tr>
    <tr>
      <th>structure</th>
      <td>0.4004</td>
      <td>0.4445</td>
      <td>0.4952</td>
    </tr>
  </tbody>
</table>
</div>



## Limitations and robustness

- **Index-number problem:** results change with the reference coefficient vector.
- **Omitted variables/endogeneity:** the coefficient component absorbs much more than a structural effect.
- **Support:** extrapolation occurs when covariate distributions do not overlap; trim/report overlap diagnostics.
- **Path dependence:** nonlinear and detailed decompositions depend on ordering or normalization.
- **Selection:** acquisition and outcome observation may both be selected.
- **Inference:** bootstrap the entire workflow, including reweighting/model fitting; cluster when observations share campaigns, geographies, or time shocks.
- Run specification curves over reference group, covariate set, link function, trimming threshold, and cohort window.

## What came next

**Juhn, Murphy & Pierce (1993)** decomposed distributional changes using residual ranks. **DiNardo, Fortin & Lemieux (1996)** constructed counterfactual densities via reweighting. **Machado & Mata (2005)** used quantile regression for counterfactual distributions. **Firpo, Fortin & Lemieux (2009)** introduced RIF regression for unconditional distributional statistics, and **Fortin, Lemieux & Firpo (2011)** unified the modern decomposition toolkit and clarified identification.

## Takeaways and bridge to Notebook 04

1. Oaxaca–Blinder splits a fitted mean gap into composition and structure.
2. Reference coefficients, support, specification, and selection matter.
3. Bootstrap uncertainty and report sensitivity across credible specifications.
4. Distributional extensions answer questions beyond the mean.

Notebook 04 replaces reference-coefficient choices with an axiomatic rule that averages marginal contributions over coalitions.

### Exercise

Add an interaction and nonlinear term to the data-generating process. Compare linear OB results with a flexible outcome model and discuss specification dependence.

## Interpretation checklist

1. State the mathematical identity or estimand.
2. Verify exactness numerically.
3. Separate description, prediction, and causation.
4. Report reference population/path/order.
5. Quantify sampling uncertainty when inputs are estimated.

## References

- Oaxaca, R. (1973). Male–female wage differentials in urban labor markets. *International Economic Review*, 14, 693–709. https://doi.org/10.2307/2525981
- Blinder, A. S. (1973). Wage discrimination: Reduced form and structural estimates. *Journal of Human Resources*, 8, 436–455. https://doi.org/10.2307/144855
- DiNardo, J., Fortin, N. M., & Lemieux, T. (1996). Labor market institutions and the distribution of wages. *Econometrica*, 64, 1001–1044. https://doi.org/10.2307/2171954
- Firpo, S., Fortin, N. M., & Lemieux, T. (2009). Unconditional quantile regressions. *Econometrica*, 77, 953–973. https://doi.org/10.3982/ECTA6822
- Juhn, C., Murphy, K. M., & Pierce, B. (1993). Wage inequality and the rise in returns to skill. *Journal of Political Economy*, 101, 410–442. https://doi.org/10.1086/261881
- Machado, J. A. F., & Mata, J. (2005). Counterfactual decomposition of changes in wage distributions using quantile regression. *Journal of Applied Econometrics*, 20, 445–465. https://doi.org/10.1002/jae.788
- Fortin, N., Lemieux, T., & Firpo, S. (2011). Decomposition methods in economics. In *Handbook of Labor Economics*, Vol. 4A, 1–102. https://doi.org/10.1016/S0169-7218(11)00407-2
