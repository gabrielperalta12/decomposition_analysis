# 06 — Subgroup Discovery and Pattern Mining: where does the change occur?

**Graduate course: Decomposition Analysis in Python**

## Learning objectives

- Define conditional change as an estimand rather than an informal “driver.”
- Distinguish generic decision trees, CART, CHAID, subgroup discovery, and RuleFit.
- Discover candidate segments on training data and evaluate them honestly on holdout data.
- Quantify support, effect size, uncertainty, multiplicity, and stability.
- Separate descriptive change localization from heterogeneous causal effects.

## Guiding question and notation

The previous notebooks answer *how an aggregate change can be allocated*. This lesson asks a different question:

> **Where in the population is the observed change concentrated?**

Let $T\in\{0,1\}$ identify the baseline and comparison periods, $Y$ be an outcome such as conversion, and $X$ contain only segment characteristics available in both periods. The descriptive conditional change is

$$\tau(x)=E[Y\mid T=1,X=x]-E[Y\mid T=0,X=x].$$

This is not automatically a treatment effect. Period is not a manipulable treatment, and customers, acquisition policies, prices, seasonality, or measurement may differ between periods.

For repeated cross-sections with $p=P(T=1)$, define the transformed outcome

$$Z_i=Y_i\left(\frac{T_i}{p}-\frac{1-T_i}{1-p}\right).$$

When period propensity is constant within the sampled population,

$$E[Z\mid X=x]=\tau(x).$$

### Proof of the transformed-outcome identity

Condition on $X=x$ and use $P(T=1\mid X=x)=p$:

$$\begin{aligned}
E[Z\mid X=x]
&=E\left[\frac{TY}{p}-\frac{(1-T)Y}{1-p}\,\middle|\,X=x\right]\\
&=\frac{P(T=1\mid x)}{p}E[Y\mid T=1,x]
-\frac{P(T=0\mid x)}{1-p}E[Y\mid T=0,x]\\
&=E[Y\mid T=1,x]-E[Y\mid T=0,x]\\
&=\tau(x).
\end{aligned}$$

The identity explains the target but also reveals its weakness. For binary $Y$, $Z$ takes values (Y/p) or (-Y/(1-p)), so its variance grows when $p$ approaches zero or one. With unequal period propensity, replace $p$ by $e(X)=P(T=1\mid X)$, diagnose overlap, and preferably use an augmented/cross-fitted pseudo-outcome to reduce noise. None of these statistical improvements makes calendar period a causal treatment.

Thus a regression tree or rule ensemble fitted to $Z$ searches directly for heterogeneous **descriptive changes**.


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from itertools import combinations
from scipy import sparse, stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor, export_text

pd.options.display.float_format = '{:,.4f}'.format
rng = np.random.default_rng(2026)
plt.style.use('seaborn-v0_8-whitegrid')
```

## Synthetic growth-marketing case

We observe two independent monthly samples. Conversion improves mainly for high-intent Paid Search visitors on mobile, deteriorates for low-intent Social traffic, and changes slightly with tenure. Because the data-generating process is known, we can judge whether each method recovers the planted patterns.

In real work, discovery never reveals “truth” this cleanly. The simulation is a unit test for reasoning, not evidence that a method controls business confounding.


```python
n = 10_000
df = pd.DataFrame({
    'period': rng.integers(0, 2, n),
    'channel': rng.choice(['Paid Search', 'Organic', 'Social'], n,
                          p=[.38, .37, .25]),
    'device': rng.choice(['Mobile', 'Desktop'], n, p=[.68, .32]),
    'market': rng.choice(['Lima', 'Mexico City', 'Bogota'], n,
                         p=[.35, .38, .27]),
    'tenure_months': rng.gamma(2.2, 5.0, n).clip(0, 36),
    'sessions_30d': rng.poisson(3.5, n),
})

base_logit = (
    -3.0
    + .18 * (df.channel == 'Paid Search')
    + .35 * (df.device == 'Desktop')
    + .055 * df.sessions_30d
    + .012 * df.tenure_months
)
true_change_logit = (
    .10
    + .85 * ((df.channel == 'Paid Search') &
             (df.device == 'Mobile') & (df.sessions_30d >= 4))
    - .55 * ((df.channel == 'Social') & (df.sessions_30d <= 2))
    + .20 * ((df.market == 'Lima') & (df.tenure_months >= 12))
)
prob = 1 / (1 + np.exp(-(base_logit + df.period * true_change_logit)))
df['converted'] = rng.binomial(1, prob)

overall = df.groupby('period').converted.agg(['mean', 'size'])
overall.loc['change', 'mean'] = overall.loc[1, 'mean'] - overall.loc[0, 'mean']
overall
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
      <th>mean</th>
      <th>size</th>
    </tr>
    <tr>
      <th>period</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0716</td>
      <td>4,997.0000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0995</td>
      <td>5,003.0000</td>
    </tr>
    <tr>
      <th>change</th>
      <td>0.0279</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



## Honest discovery/evaluation split

Using the same observations to search thousands of rules and report the best one produces winner's-curse bias. We therefore:

1. split observations before searching;
2. learn partitions or rules only on the discovery sample;
3. freeze their definitions;
4. estimate period-specific rates and changes on holdout data;
5. report support and uncertainty, not only rank.

For a fixed subgroup $S$, an approximate standard error for the difference between two binary rates is

$$SE(\widehat\Delta_S)=\sqrt{
\frac{\hat r_{S1}(1-\hat r_{S1})}{n_{S1}}+
\frac{\hat r_{S0}(1-\hat r_{S0})}{n_{S0}}}.$$

This interval is valid for a prespecified subgroup under ordinary sampling assumptions. Selection makes discovery-sample intervals optimistic; holdout evaluation limits that problem.


```python
train, holdout = train_test_split(
    df, test_size=.40, random_state=42, stratify=df['period']
)
p_train = train.period.mean()
train = train.copy()
holdout = holdout.copy()
train['change_target'] = train.converted * (
    train.period / p_train - (1 - train.period) / (1 - p_train)
)

def change_report(data, group_col):
    table = data.groupby([group_col, 'period']).converted.agg(['mean', 'size']).unstack()
    table.columns = [f'{stat}_t{period}' for stat, period in table.columns]
    table['change'] = table['mean_t1'] - table['mean_t0']
    table['se'] = np.sqrt(
        table.mean_t1 * (1-table.mean_t1) / table.size_t1
        + table.mean_t0 * (1-table.mean_t0) / table.size_t0
    )
    table['ci_low'] = table.change - 1.96 * table.se
    table['ci_high'] = table.change + 1.96 * table.se
    return table.sort_values('change', ascending=False)

train.shape, holdout.shape
```




    ((6000, 8), (4000, 7))



## 1. Decision trees: the broad family

A decision tree recursively partitions feature space. At each internal node it chooses a condition such as `sessions_30d ≤ 3.5`; terminal leaves estimate a local target. “Decision tree” is the family name, not a unique algorithm.

Trees are attractive for change localization because they produce mutually exclusive, collectively exhaustive segments. Their main weakness is instability: small data changes can alter early splits and therefore the entire tree.

### The mathematical object learned by a tree

A tree $\mathcal T$ partitions the feature space into leaves $A_1,\ldots,A_L$ and estimates a piecewise-constant function

$$\widehat\tau_{\mathcal T}(x)=
\sum_{\ell=1}^{L}\widehat\tau_\ell\mathbf 1\{x\in A_\ell\},
\qquad
\widehat\tau_\ell=\frac{1}{n_\ell}\sum_{i:X_i\in A_\ell}Z_i.$$

Each observation belongs to exactly one leaf. This makes tree leaves suitable for an additive reporting frontier, unlike overlapping rules. But the fitted leaf mean predicts conditional change; it is not yet the leaf's contribution to the aggregate KPI. For a rate KPI, the exact contribution is

$$c(A_\ell)=w_1(A_\ell)r_1(A_\ell)-w_0(A_\ell)r_0(A_\ell).$$

A small leaf may have a large local change $\widehat\tau_\ell$ but a small aggregate contribution, while a broad leaf with modest change can dominate the total.

### “Decision tree” versus named algorithms

| Algorithm/family | Typical split | Typical target | Characteristic |
|---|---|---|---|
| ID3 | multiway categorical | classification | information gain |
| C4.5 | categorical and threshold | classification | gain ratio and pruning |
| CART | binary | regression/classification | SSE, Gini, or deviance; cost-complexity pruning |
| CHAID | multiway categorical | categorical/ordinal | significance tests and category merging |
| Model-based trees | parameter-instability split | fitted local model | coefficients can vary by leaf |

Saying “we used a decision tree” is therefore underspecified. Report the target, split loss, categorical encoding, stopping rule, pruning rule, missing-value policy, and evaluation design.

### Greedy recursion and its consequences

At node $A$, the algorithm evaluates a restricted set of candidate conditions, commits to the best immediate split, and recurses. It does not generally revisit an early choice after seeing later splits. Consequently, a weak main effect can hide a strong deep interaction; correlated variables can substitute for one another; one-hot encoding can turn a naturally multiway categorical question into sequential binary questions; and a locally optimal tree need not be the globally optimal partition.

### CART is a specific decision-tree algorithm

Classification and Regression Trees (CART) uses binary splits. For a regression target $Z$, a candidate split $(j,s)$ minimizes within-child squared error:

$$\mathcal L(j,s)=
\sum_{i:X_{ij}\le s}(Z_i-\bar Z_L)^2+
\sum_{i:X_{ij}>s}(Z_i-\bar Z_R)^2.$$

Equivalently, CART maximizes impurity reduction

$$G(j,s)=I(A)-\frac{n_L}{n_A}I(A_L)-\frac{n_R}{n_A}I(A_R),$$

where $I(A)=n_A^{-1}\sum_{i\in A}(Z_i-\bar Z_A)^2$ for regression. A large gain means the child means are more homogeneous under squared loss. It does **not** imply statistical significance, stability, causal identification, or economic materiality.

### Closed-form CART gain

The ANOVA identity for one binary split is

$$\underbrace{\sum_{i\in A}(Z_i-\bar Z_A)^2}_{SSE_A}
=\underbrace{SSE_L+SSE_R}_{\text{within children}}
+\underbrace{\frac{n_Ln_R}{n_A}(\bar Z_L-\bar Z_R)^2}_{\text{between children}}.$$

Therefore the reduction in SSE is exactly

$$\boxed{\Delta SSE=
\frac{n_Ln_R}{n_A}(\bar Z_L-\bar Z_R)^2}.$$

This formula gives the intuition behind CART. A split is attractive when child means differ, but the factor $n_Ln_R/n_A$ penalizes extremely unbalanced children. Dividing by $n_A$ gives the normalized impurity gain. It also shows why a huge local change in a tiny subgroup may lose to a smaller contrast covering more observations.

**Proof sketch.** Write $Z_i-\bar Z_A=(Z_i-\bar Z_h)+(\bar Z_h-\bar Z_A)$ inside child $h\in\{L,R\}$. The cross term vanishes because residuals sum to zero within each child. Summing the two remaining between-child terms and using $n_L(\bar Z_L-\bar Z_A)+n_R(\bar Z_R-\bar Z_A)=0$ yields the boxed expression.

The fully grown tree is typically regularized through depth, minimum leaf size, or cost-complexity pruning:

$$R_\alpha(\mathcal T)=R(\mathcal T)+\alpha|\mathcal T|.$$

Large leaves improve precision but can hide narrow patterns; small leaves increase discovery power at the cost of variance and false findings.


```python
features = ['channel', 'device', 'market', 'tenure_months', 'sessions_30d']
categorical = ['channel', 'device', 'market']
numeric = ['tenure_months', 'sessions_30d']
encoder = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical),
    ('num', 'passthrough', numeric),
])
X_train = encoder.fit_transform(train[features])
X_holdout = encoder.transform(holdout[features])
feature_names = encoder.get_feature_names_out()

cart = DecisionTreeRegressor(
    max_depth=3, min_samples_leaf=300, random_state=42
).fit(X_train, train.change_target)
print(export_text(cart, feature_names=list(feature_names), decimals=3))

train['cart_leaf'] = cart.apply(X_train)
holdout['cart_leaf'] = cart.apply(X_holdout)
cart_holdout = change_report(holdout, 'cart_leaf')
cart_holdout
```

    |--- cat__channel_Paid Search <= 0.500
    |   |--- num__sessions_30d <= 2.500
    |   |   |--- cat__channel_Organic <= 0.500
    |   |   |   |--- value: [0.017]
    |   |   |--- cat__channel_Organic >  0.500
    |   |   |   |--- value: [-0.060]
    |   |--- num__sessions_30d >  2.500
    |   |   |--- num__tenure_months <= 18.549
    |   |   |   |--- value: [0.028]
    |   |   |--- num__tenure_months >  18.549
    |   |   |   |--- value: [-0.031]
    |--- cat__channel_Paid Search >  0.500
    |   |--- num__sessions_30d <= 3.500
    |   |   |--- num__sessions_30d <= 1.500
    |   |   |   |--- value: [0.077]
    |   |   |--- num__sessions_30d >  1.500
    |   |   |   |--- value: [0.018]
    |   |--- num__sessions_30d >  3.500
    |   |   |--- cat__device_Desktop <= 0.500
    |   |   |   |--- value: [0.177]
    |   |   |--- cat__device_Desktop >  0.500
    |   |   |   |--- value: [0.035]
    





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
      <th>mean_t0</th>
      <th>mean_t1</th>
      <th>size_t0</th>
      <th>size_t1</th>
      <th>change</th>
      <th>se</th>
      <th>ci_low</th>
      <th>ci_high</th>
    </tr>
    <tr>
      <th>cart_leaf</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>13</th>
      <td>0.0866</td>
      <td>0.1870</td>
      <td>254</td>
      <td>246</td>
      <td>0.1004</td>
      <td>0.0305</td>
      <td>0.0406</td>
      <td>0.1601</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.0439</td>
      <td>0.1078</td>
      <td>114</td>
      <td>102</td>
      <td>0.0640</td>
      <td>0.0362</td>
      <td>-0.0070</td>
      <td>0.1350</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.0873</td>
      <td>0.1148</td>
      <td>126</td>
      <td>122</td>
      <td>0.0275</td>
      <td>0.0383</td>
      <td>-0.0476</td>
      <td>0.1025</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0.1134</td>
      <td>0.1354</td>
      <td>97</td>
      <td>96</td>
      <td>0.0220</td>
      <td>0.0475</td>
      <td>-0.0711</td>
      <td>0.1151</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0649</td>
      <td>0.0866</td>
      <td>262</td>
      <td>231</td>
      <td>0.0217</td>
      <td>0.0240</td>
      <td>-0.0253</td>
      <td>0.0687</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0.0772</td>
      <td>0.0909</td>
      <td>311</td>
      <td>308</td>
      <td>0.0137</td>
      <td>0.0223</td>
      <td>-0.0300</td>
      <td>0.0574</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.0721</td>
      <td>0.0682</td>
      <td>680</td>
      <td>719</td>
      <td>-0.0039</td>
      <td>0.0137</td>
      <td>-0.0307</td>
      <td>0.0229</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0774</td>
      <td>0.0621</td>
      <td>155</td>
      <td>177</td>
      <td>-0.0153</td>
      <td>0.0281</td>
      <td>-0.0704</td>
      <td>0.0398</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Verify the CART gain identity for a candidate Paid Search split.
left = train.loc[train.channel.ne('Paid Search'), 'change_target']
right = train.loc[train.channel.eq('Paid Search'), 'change_target']
parent = train.change_target
sse_parent = ((parent-parent.mean())**2).sum()
sse_children = ((left-left.mean())**2).sum() + ((right-right.mean())**2).sum()
gain_direct = sse_parent-sse_children
gain_closed_form = (
    len(left)*len(right)/len(parent)*(left.mean()-right.mean())**2
)
pd.Series({
    'left mean': left.mean(),
    'right mean': right.mean(),
    'direct SSE gain': gain_direct,
    'closed-form gain': gain_closed_form,
    'identity error': gain_direct-gain_closed_form,
})
```




    left mean          0.0048
    right mean         0.0784
    direct SSE gain    7.6866
    closed-form gain   7.6866
    identity error     0.0000
    dtype: float64




```python
# Overfitting diagnostic: compare a deep tree with the regularized tree.
deep_cart = DecisionTreeRegressor(
    min_samples_leaf=20, random_state=42
).fit(X_train, train.change_target)
pd.DataFrame({
    'model': ['regularized CART', 'deep CART'],
    'leaves': [cart.get_n_leaves(), deep_cart.get_n_leaves()],
    'train MSE': [mean_squared_error(train.change_target, cart.predict(X_train)),
                  mean_squared_error(train.change_target, deep_cart.predict(X_train))],
    'holdout proxy MSE': [mean_squared_error(
        holdout.converted * (holdout.period/holdout.period.mean()
        - (1-holdout.period)/(1-holdout.period.mean())), cart.predict(X_holdout)),
        mean_squared_error(
        holdout.converted * (holdout.period/holdout.period.mean()
        - (1-holdout.period)/(1-holdout.period.mean())),
        deep_cart.predict(X_holdout))],
})
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
      <th>model</th>
      <th>leaves</th>
      <th>train MSE</th>
      <th>holdout proxy MSE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>regularized CART</td>
      <td>8</td>
      <td>0.3370</td>
      <td>0.3436</td>
    </tr>
    <tr>
      <th>1</th>
      <td>deep CART</td>
      <td>228</td>
      <td>0.3021</td>
      <td>0.3798</td>
    </tr>
  </tbody>
</table>
</div>



### Cost-complexity pruning in practice

CART's weakest-link pruning first grows a large tree and produces a nested sequence of subtrees indexed by $\alpha$. Cross-validation should choose among those subtrees. The smallest validation error is not the only defensible choice: the **one-standard-error rule** selects the simplest tree whose validation loss is statistically indistinguishable from the minimum.

The next diagnostic evaluates a compact grid of pruning values on the untouched holdout pseudo-outcome only for teaching. In a real honest workflow, tune $\alpha$ inside the discovery sample and reserve holdout solely for final evaluation. Otherwise the holdout quietly becomes another training set.


```python
holdout_p = holdout.period.mean()
holdout_target = holdout.converted * (
    holdout.period/holdout_p - (1-holdout.period)/(1-holdout_p)
)
pruning_path = deep_cart.cost_complexity_pruning_path(
    X_train, train.change_target
)
alpha_grid = np.unique(np.quantile(pruning_path.ccp_alphas, np.linspace(0, 1, 12)))
pruning_rows = []
for alpha in alpha_grid:
    candidate_tree = DecisionTreeRegressor(
        ccp_alpha=float(alpha), min_samples_leaf=20, random_state=42
    ).fit(X_train, train.change_target)
    pruning_rows.append({
        'ccp_alpha': alpha,
        'leaves': candidate_tree.get_n_leaves(),
        'train MSE': mean_squared_error(
            train.change_target, candidate_tree.predict(X_train)
        ),
        'holdout proxy MSE': mean_squared_error(
            holdout_target, candidate_tree.predict(X_holdout)
        ),
    })
pruning_diagnostic = pd.DataFrame(pruning_rows)
pruning_diagnostic.sort_values('holdout proxy MSE').head(8)
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
      <th>ccp_alpha</th>
      <th>leaves</th>
      <th>train MSE</th>
      <th>holdout proxy MSE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>11</th>
      <td>0.0013</td>
      <td>1</td>
      <td>0.3408</td>
      <td>0.3427</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.0004</td>
      <td>25</td>
      <td>0.3260</td>
      <td>0.3593</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.0002</td>
      <td>53</td>
      <td>0.3189</td>
      <td>0.3675</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.0001</td>
      <td>93</td>
      <td>0.3118</td>
      <td>0.3714</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.0001</td>
      <td>116</td>
      <td>0.3088</td>
      <td>0.3743</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.0001</td>
      <td>139</td>
      <td>0.3063</td>
      <td>0.3760</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.0001</td>
      <td>156</td>
      <td>0.3049</td>
      <td>0.3772</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0001</td>
      <td>173</td>
      <td>0.3037</td>
      <td>0.3778</td>
    </tr>
  </tbody>
</table>
</div>



## 2. CHAID: categorical, multiway, significance-driven splitting

CHAID (Chi-squared Automatic Interaction Detection) differs from CART in three important ways:

- it was designed around categorical predictors and outcomes;
- it merges statistically similar predictor categories before splitting;
- it can create multiway rather than only binary splits.

At a node, classical CHAID uses Pearson's statistic

$$X^2=\sum_{a,b}\frac{(O_{ab}-E_{ab})^2}{E_{ab}}$$

and adjusts category-merging/split decisions for multiple comparisons, commonly with Bonferroni corrections. Continuous predictors must first be binned, so cutpoints are substantive tuning choices.

### Classical CHAID node algorithm

For every predictor at the current node:

1. cross-tabulate predictor categories against the categorical target;
2. for each eligible category pair, test whether their target distributions differ;
3. merge the least different pair when its adjusted $p$-value exceeds the merge threshold;
4. repeat merging until no eligible pair remains;
5. optionally split an ordinal category that was previously merged if separation becomes significant;
6. compute the adjusted significance of the remaining multiway association;
7. split on the predictor with the smallest adjusted $p$-value, if below the split threshold;
8. recurse until depth, sample-size, or significance stopping rules bind.

For unordered predictors with $J$ categories there are initially $\binom{J}{2}$ pairwise comparisons. Bonferroni adjustment protects against this local search by replacing $p$ with approximately $\min(1,mp)$ for $m$ comparisons. It does not automatically correct the full adaptivity of all nodes, variables, bins, and pruning choices.

### Applying CHAID to *change*

Classical CHAID of `converted` on customer characteristics finds groups with different **levels**, not necessarily different **period changes**. To target change, the node model must include period and test a period-by-category interaction. For categorical predictor $G$:

$$\operatorname{logit}P(Y=1\mid T,G)
=\beta_0+\beta_TT+\sum_{g>1}\beta_g\mathbf1(G=g)
+\sum_{g>1}\gamma_gT\mathbf1(G=g).$$

The null $H_0:\gamma_2=\cdots=\gamma_J=0$ says the period contrast is homogeneous across categories. Rejecting it localizes heterogeneity but does not identify why it occurred.

### Expected-count and binning diagnostics

Pearson's approximation becomes unreliable with small expected counts. Merge sparse categories, use an exact or Monte Carlo test where appropriate, or stop splitting. For continuous features, results can change materially with initial bins; report the binning rule and test sensitivity to quantile, business, and monotonic bins.

### Why Pearson's statistic has a chi-square reference

Under independence of row variable (A) and column variable (B), the fitted expected count is

$$E_{ab}=\frac{n_{a+}n_{+b}}{n}.$$

The standardized residuals $(O_{ab}-E_{ab})/\sqrt{E_{ab}}$ are asymptotically normal, but row and column totals impose linear constraints. An (R\times C) table has (RC) cells and (R+C-1) independent marginal constraints, leaving

$$RC-(R+C-1)=(R-1)(C-1)$$

degrees of freedom. Hence, under regularity conditions,

$$X^2=\sum_{a=1}^{R}\sum_{b=1}^{C}
\frac{(O_{ab}-E_{ab})^2}{E_{ab}}
\overset{a}{\sim}\chi^2_{(R-1)(C-1)}.$$

The likelihood-ratio statistic used below is

$$G^2=2\{\ell(\text{interaction model})-\ell(\text{additive model})\},$$

and is asymptotically chi-square with degrees of freedom equal to the number of added interaction parameters. For (J) categories and binary period, this difference is typically (J-1). Pearson and likelihood-ratio tests become asymptotically equivalent, though they can differ in sparse samples.

Below is a **CHAID-style first-node diagnostic**, not a full CHAID implementation. For each categorical feature it compares a logistic model containing period and category main effects with one also containing their interaction. The likelihood-ratio test asks whether descriptive period change varies across categories.


```python
def chaid_style_screen(data, variables):
    rows = []
    for variable in variables:
        reduced = smf.logit(
            f'converted ~ period + C({variable})', data=data
        ).fit(disp=False)
        full = smf.logit(
            f'converted ~ period * C({variable})', data=data
        ).fit(disp=False)
        lr = 2 * (full.llf - reduced.llf)
        df_diff = int(full.df_model - reduced.df_model)
        rows.append({
            'candidate split': variable,
            'LR chi2': lr,
            'df': df_diff,
            'raw p': stats.chi2.sf(lr, df_diff),
        })
    result = pd.DataFrame(rows).sort_values('raw p')
    result['Bonferroni p'] = np.minimum(1, result['raw p'] * len(result))
    return result

chaid_screen = chaid_style_screen(train, categorical)
chaid_screen
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
      <th>candidate split</th>
      <th>LR chi2</th>
      <th>df</th>
      <th>raw p</th>
      <th>Bonferroni p</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>channel</td>
      <td>20.9043</td>
      <td>2</td>
      <td>0.0000</td>
      <td>0.0001</td>
    </tr>
    <tr>
      <th>2</th>
      <td>market</td>
      <td>3.3312</td>
      <td>2</td>
      <td>0.1891</td>
      <td>0.5672</td>
    </tr>
    <tr>
      <th>1</th>
      <td>device</td>
      <td>0.4980</td>
      <td>1</td>
      <td>0.4804</td>
      <td>1.0000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Make the winning candidate substantively interpretable.
winning_chaid_variable = chaid_screen.iloc[0]['candidate split']
chaid_category_changes = (
    train.groupby([winning_chaid_variable, 'period'])
    .converted.agg(['mean', 'size']).unstack()
)
chaid_category_changes.columns = [
    f'{stat}_t{period}' for stat, period in chaid_category_changes.columns
]
chaid_category_changes['change'] = (
    chaid_category_changes.mean_t1-chaid_category_changes.mean_t0
)
chaid_category_changes.sort_values('change', ascending=False)
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
      <th>mean_t0</th>
      <th>mean_t1</th>
      <th>size_t0</th>
      <th>size_t1</th>
      <th>change</th>
    </tr>
    <tr>
      <th>channel</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Paid Search</th>
      <td>0.0664</td>
      <td>0.1447</td>
      <td>1144</td>
      <td>1147</td>
      <td>0.0783</td>
    </tr>
    <tr>
      <th>Social</th>
      <td>0.0563</td>
      <td>0.0801</td>
      <td>746</td>
      <td>749</td>
      <td>0.0238</td>
    </tr>
    <tr>
      <th>Organic</th>
      <td>0.0803</td>
      <td>0.0723</td>
      <td>1108</td>
      <td>1106</td>
      <td>-0.0080</td>
    </tr>
  </tbody>
</table>
</div>



### Reading the CHAID-style output

The interaction screen answers whether change differs somewhere across the categories of a variable. It does not say every category differs, nor does its $p$-value measure effect size. Read it jointly with category rates, denominators, absolute changes, and a commercially relevant threshold.

The demonstration stops after the first node and does not implement CHAID's iterative category merging, multiway recursion, ordinal restrictions, or full multiplicity bookkeeping. Calling it “CHAID-style interaction screening” is deliberate. A production CHAID analysis should use a tested implementation and disclose its merge/split thresholds.

## 3. Subgroup Discovery: explicit search for interesting rules

Subgroup Discovery is a pattern-mining task: search a language of human-readable descriptions $S$ for subsets whose target distribution is unusual. Unlike a tree, discovered rules may overlap and need not cover the whole population.

A common quality function balances size and exceptionality. For change localization, one useful choice is

$$Q(S)=\left(\frac{n_S}{n}\right)^\alpha
\left|\widehat\Delta_S-\widehat\Delta\right|,$$

where support controls reliability and $\alpha$ controls the preference for broad versus narrow rules. Other choices include weighted relative accuracy, likelihood ratios, unusualness, or lower confidence bounds.

### Description language and search lattice

A rule is a conjunction such as

$$S(x)=\mathbf1\{\text{channel=Paid Search}\}
\mathbf1\{\text{device=Mobile}\}
\mathbf1\{\text{sessions}\ge4\}.$$

The description language determines what can be discovered. A refinement operator adds one condition at a time, producing a lattice from general to specific rules. Exhaustive enumeration grows combinatorially, so algorithms use beam search, branch-and-bound, evolutionary search, Monte Carlo search, or exceptional-model trees.

Beam search retains only the best $B$ rules at each depth. It is fast and interpretable but can discard a mediocre parent whose refinement would have been excellent. Branch-and-bound is exact only when the quality measure has a valid optimistic upper bound.

### Quality measures answer different questions

Let $s=P(X\in S)$, $\Delta_S$ be subgroup change, and $\Delta$ global change.

| Quality | Formula | Preference |
|---|---|---|
| unusualness | $s^\alpha|\Delta_S-\Delta|$ | coverage balanced with exceptional change |
| signed impact | $s(\Delta_S-\Delta)$ | broad positive deviations |
| absolute KPI mass | $|w_1r_1-w_0r_0|$ | contribution to aggregate movement |
| standardized score | $|\Delta_S-\Delta|/SE$ | precision, often favors moderate large samples |
| lower confidence bound | $|\Delta_S-\Delta|-zSE$ | conservative materiality |

No score is universally correct. An unusual subgroup need not contribute much to the aggregate KPI, and the largest contribution need not have the most exceptional local change. Choose the score from the decision problem, then test sensitivity.

### Covariance intuition for signed unusualness

Let (W=\mathbf1\{X\in S\}) and let (Z) be a pseudo-outcome whose conditional mean is the period change. Then

$$\begin{aligned}
\operatorname{Cov}(W,Z)
&=E[WZ]-E[W]E[Z]\\
&=sE[Z\mid W=1]-sE[Z]\\
&=s(\Delta_S-\Delta).
\end{aligned}$$

Thus signed impact is exactly the covariance between rule membership and change target. Absolute unusualness uses its magnitude, possibly with (s^\alpha) instead of (s). This explains both its appeal and limitation: it measures association between membership and change, not a causal effect or an additive contribution to the observed KPI.

### Search multiplicity

Even if every subgroup has the same population change, the largest empirical quality among thousands of candidates will be positive. A permutation calibration repeats the **entire search** under a null created by shuffling period labels and records the maximum score:

$$M^{(b)}=\max_{S\in\mathcal D}Q^{(b)}(S).$$

The empirical tail probability compares the observed maximum with the null distribution of maxima, not with the null distribution of one prespecified rule. This controls a family-level search statistic for the exact candidate language and permutation scheme used.

### Redundancy and overlap

Top rules are often near-duplicates. For membership sets $S_a,S_b$, use Jaccard overlap

$$J(S_a,S_b)=\frac{|S_a\cap S_b|}{|S_a\cup S_b|}.$$

One can suppress a new rule when its overlap with a higher-ranked rule exceeds a threshold, or optimize a diverse top-$k$ set. Because overlapping rules reuse observations, their changes or quality scores cannot be summed into an aggregate decomposition. Convert them into disjoint atoms or use a separate reconciliation rule when conservation is required.

The code below uses a compact beam search over one- and two-condition conjunctions. It is deliberately transparent: production systems should also canonicalize duplicate rules, control the search budget, correct multiplicity, and evaluate frozen rules on fresh data.


```python
conditions = {
    'channel=Paid Search': train.channel.eq('Paid Search'),
    'channel=Organic': train.channel.eq('Organic'),
    'channel=Social': train.channel.eq('Social'),
    'device=Mobile': train.device.eq('Mobile'),
    'device=Desktop': train.device.eq('Desktop'),
    'market=Lima': train.market.eq('Lima'),
    'sessions>=4': train.sessions_30d.ge(4),
    'sessions<=2': train.sessions_30d.le(2),
    'tenure>=12': train.tenure_months.ge(12),
}

def period_change(data, mask):
    selected = data.loc[mask]
    rates = selected.groupby('period').converted.mean()
    if len(rates) < 2:
        return np.nan
    return rates.loc[1] - rates.loc[0]

global_change = period_change(train, pd.Series(True, index=train.index))
candidates = []
items = list(conditions.items())
for depth in [1, 2]:
    for combo in combinations(items, depth):
        names, masks = zip(*combo)
        mask = np.logical_and.reduce(masks)
        support = mask.mean()
        if mask.sum() < 250 or train.loc[mask, 'period'].nunique() < 2:
            continue
        delta = period_change(train, mask)
        quality = np.sqrt(support) * abs(delta - global_change)
        candidates.append({
            'rule': ' AND '.join(names), 'support_train': support,
            'change_train': delta, 'quality_train': quality,
        })

discovered = pd.DataFrame(candidates).sort_values(
    'quality_train', ascending=False
).drop_duplicates('rule').head(8)
discovered
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
      <th>rule</th>
      <th>support_train</th>
      <th>change_train</th>
      <th>quality_train</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>12</th>
      <td>channel=Paid Search AND sessions&gt;=4</td>
      <td>0.1755</td>
      <td>0.1193</td>
      <td>0.0362</td>
    </tr>
    <tr>
      <th>9</th>
      <td>channel=Paid Search AND device=Mobile</td>
      <td>0.2617</td>
      <td>0.1033</td>
      <td>0.0360</td>
    </tr>
    <tr>
      <th>19</th>
      <td>channel=Organic AND sessions&lt;=2</td>
      <td>0.1175</td>
      <td>-0.0573</td>
      <td>0.0309</td>
    </tr>
    <tr>
      <th>14</th>
      <td>channel=Paid Search AND tenure&gt;=12</td>
      <td>0.1357</td>
      <td>0.1137</td>
      <td>0.0298</td>
    </tr>
    <tr>
      <th>0</th>
      <td>channel=Paid Search</td>
      <td>0.3818</td>
      <td>0.0783</td>
      <td>0.0281</td>
    </tr>
    <tr>
      <th>17</th>
      <td>channel=Organic AND market=Lima</td>
      <td>0.1292</td>
      <td>-0.0444</td>
      <td>0.0278</td>
    </tr>
    <tr>
      <th>1</th>
      <td>channel=Organic</td>
      <td>0.3690</td>
      <td>-0.0080</td>
      <td>0.0248</td>
    </tr>
    <tr>
      <th>15</th>
      <td>channel=Organic AND device=Mobile</td>
      <td>0.2567</td>
      <td>-0.0122</td>
      <td>0.0228</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Calibrate the maximum search score under a shuffled-period null.
search_masks = []
for depth in [1, 2]:
    for combo in combinations(items, depth):
        mask = np.logical_and.reduce([item[1].to_numpy() for item in combo])
        if mask.sum() >= 250:
            search_masks.append(mask)

def maximum_quality_for_period(period_vector):
    y_values = train.converted.to_numpy()
    global_delta = y_values[period_vector == 1].mean() - y_values[period_vector == 0].mean()
    scores = []
    for mask in search_masks:
        subgroup_period = period_vector[mask]
        subgroup_y = y_values[mask]
        if np.unique(subgroup_period).size < 2:
            continue
        delta = (
            subgroup_y[subgroup_period == 1].mean()
            - subgroup_y[subgroup_period == 0].mean()
        )
        scores.append(np.sqrt(mask.mean())*abs(delta-global_delta))
    return max(scores)

permutation_rng = np.random.default_rng(123)
observed_max_quality = discovered.quality_train.max()
null_maxima = np.array([
    maximum_quality_for_period(
        permutation_rng.permutation(train.period.to_numpy())
    )
    for _ in range(200)
])
permutation_diagnostic = pd.Series({
    'observed maximum quality': observed_max_quality,
    'null 95% maximum': np.quantile(null_maxima, .95),
    'family-wise permutation p': (
        1+np.sum(null_maxima >= observed_max_quality)
    )/(len(null_maxima)+1),
})
permutation_diagnostic
```




    observed maximum quality    0.0362
    null 95% maximum            0.0221
    family-wise permutation p   0.0050
    dtype: float64




```python
# Freeze the top rules and estimate them on holdout observations.
def evaluate_rule_text(data, rule):
    mask = pd.Series(True, index=data.index)
    mapping = {
        'channel=Paid Search': data.channel.eq('Paid Search'),
        'channel=Organic': data.channel.eq('Organic'),
        'channel=Social': data.channel.eq('Social'),
        'device=Mobile': data.device.eq('Mobile'),
        'device=Desktop': data.device.eq('Desktop'),
        'market=Lima': data.market.eq('Lima'),
        'sessions>=4': data.sessions_30d.ge(4),
        'sessions<=2': data.sessions_30d.le(2),
        'tenure>=12': data.tenure_months.ge(12),
    }
    for part in rule.split(' AND '):
        mask &= mapping[part]
    selected = data.loc[mask]
    rates = selected.groupby('period').converted.agg(['mean', 'size'])
    delta = rates.loc[1, 'mean'] - rates.loc[0, 'mean']
    se = np.sqrt(sum(rates['mean'] * (1-rates['mean']) / rates['size']))
    return pd.Series({'support_holdout': mask.mean(), 'change_holdout': delta,
                      'ci_low': delta-1.96*se, 'ci_high': delta+1.96*se})

honest_rules = discovered.join(
    discovered.rule.apply(lambda rule: evaluate_rule_text(holdout, rule))
)
honest_rules
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
      <th>rule</th>
      <th>support_train</th>
      <th>change_train</th>
      <th>quality_train</th>
      <th>support_holdout</th>
      <th>change_holdout</th>
      <th>ci_low</th>
      <th>ci_high</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>12</th>
      <td>channel=Paid Search AND sessions&gt;=4</td>
      <td>0.1755</td>
      <td>0.1193</td>
      <td>0.0362</td>
      <td>0.1732</td>
      <td>0.0785</td>
      <td>0.0281</td>
      <td>0.1289</td>
    </tr>
    <tr>
      <th>9</th>
      <td>channel=Paid Search AND device=Mobile</td>
      <td>0.2617</td>
      <td>0.1033</td>
      <td>0.0360</td>
      <td>0.2700</td>
      <td>0.0635</td>
      <td>0.0265</td>
      <td>0.1005</td>
    </tr>
    <tr>
      <th>19</th>
      <td>channel=Organic AND sessions&lt;=2</td>
      <td>0.1175</td>
      <td>-0.0573</td>
      <td>0.0309</td>
      <td>0.1232</td>
      <td>0.0217</td>
      <td>-0.0253</td>
      <td>0.0687</td>
    </tr>
    <tr>
      <th>14</th>
      <td>channel=Paid Search AND tenure&gt;=12</td>
      <td>0.1357</td>
      <td>0.1137</td>
      <td>0.0298</td>
      <td>0.1412</td>
      <td>0.0089</td>
      <td>-0.0444</td>
      <td>0.0622</td>
    </tr>
    <tr>
      <th>0</th>
      <td>channel=Paid Search</td>
      <td>0.3818</td>
      <td>0.0783</td>
      <td>0.0281</td>
      <td>0.3820</td>
      <td>0.0504</td>
      <td>0.0197</td>
      <td>0.0811</td>
    </tr>
    <tr>
      <th>17</th>
      <td>channel=Organic AND market=Lima</td>
      <td>0.1292</td>
      <td>-0.0444</td>
      <td>0.0278</td>
      <td>0.1237</td>
      <td>0.0179</td>
      <td>-0.0312</td>
      <td>0.0670</td>
    </tr>
    <tr>
      <th>1</th>
      <td>channel=Organic</td>
      <td>0.3690</td>
      <td>-0.0080</td>
      <td>0.0248</td>
      <td>0.3633</td>
      <td>0.0053</td>
      <td>-0.0214</td>
      <td>0.0319</td>
    </tr>
    <tr>
      <th>15</th>
      <td>channel=Organic AND device=Mobile</td>
      <td>0.2567</td>
      <td>-0.0122</td>
      <td>0.0228</td>
      <td>0.2522</td>
      <td>0.0028</td>
      <td>-0.0284</td>
      <td>0.0340</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Quantify discovery optimism and redundancy among the selected rules.
honest_rules = honest_rules.assign(
    absolute_shrinkage=lambda x: x.change_train.abs()-x.change_holdout.abs()
)

def mask_for_rule(data, rule):
    mapping = {
        'channel=Paid Search': data.channel.eq('Paid Search'),
        'channel=Organic': data.channel.eq('Organic'),
        'channel=Social': data.channel.eq('Social'),
        'device=Mobile': data.device.eq('Mobile'),
        'device=Desktop': data.device.eq('Desktop'),
        'market=Lima': data.market.eq('Lima'),
        'sessions>=4': data.sessions_30d.ge(4),
        'sessions<=2': data.sessions_30d.le(2),
        'tenure>=12': data.tenure_months.ge(12),
    }
    mask = pd.Series(True, index=data.index)
    for part in rule.split(' AND '):
        mask &= mapping[part]
    return mask

rule_masks = {
    rule: mask_for_rule(holdout, rule).to_numpy()
    for rule in honest_rules.rule.head(6)
}
jaccard = pd.DataFrame(index=rule_masks, columns=rule_masks, dtype=float)
for left_name, left_mask in rule_masks.items():
    for right_name, right_mask in rule_masks.items():
        union = np.logical_or(left_mask, right_mask).sum()
        jaccard.loc[left_name, right_name] = (
            np.logical_and(left_mask, right_mask).sum()/union if union else np.nan
        )
honest_rules[['rule', 'change_train', 'change_holdout', 'absolute_shrinkage']], jaccard
```




    (                                     rule  change_train  change_holdout  \
     12    channel=Paid Search AND sessions>=4        0.1193          0.0785   
     9   channel=Paid Search AND device=Mobile        0.1033          0.0635   
     19        channel=Organic AND sessions<=2       -0.0573          0.0217   
     14     channel=Paid Search AND tenure>=12        0.1137          0.0089   
     0                     channel=Paid Search        0.0783          0.0504   
     17        channel=Organic AND market=Lima       -0.0444          0.0179   
     1                         channel=Organic       -0.0080          0.0053   
     15      channel=Organic AND device=Mobile       -0.0122          0.0028   
     
         absolute_shrinkage  
     12              0.0408  
     9               0.0398  
     19              0.0356  
     14              0.1048  
     0               0.0279  
     17              0.0265  
     1               0.0027  
     15              0.0094  ,
                                            channel=Paid Search AND sessions>=4  \
     channel=Paid Search AND sessions>=4                                 1.0000   
     channel=Paid Search AND device=Mobile                               0.3928   
     channel=Organic AND sessions<=2                                     0.0000   
     channel=Paid Search AND tenure>=12                                  0.2517   
     channel=Paid Search                                                 0.4535   
     channel=Organic AND market=Lima                                     0.0000   
     
                                            channel=Paid Search AND device=Mobile  \
     channel=Paid Search AND sessions>=4                                   0.3928   
     channel=Paid Search AND device=Mobile                                 1.0000   
     channel=Organic AND sessions<=2                                       0.0000   
     channel=Paid Search AND tenure>=12                                    0.3171   
     channel=Paid Search                                                   0.7068   
     channel=Organic AND market=Lima                                       0.0000   
     
                                            channel=Organic AND sessions<=2  \
     channel=Paid Search AND sessions>=4                             0.0000   
     channel=Paid Search AND device=Mobile                           0.0000   
     channel=Organic AND sessions<=2                                 1.0000   
     channel=Paid Search AND tenure>=12                              0.0000   
     channel=Paid Search                                             0.0000   
     channel=Organic AND market=Lima                                 0.2182   
     
                                            channel=Paid Search AND tenure>=12  \
     channel=Paid Search AND sessions>=4                                0.2517   
     channel=Paid Search AND device=Mobile                              0.3171   
     channel=Organic AND sessions<=2                                    0.0000   
     channel=Paid Search AND tenure>=12                                 1.0000   
     channel=Paid Search                                                0.3698   
     channel=Organic AND market=Lima                                    0.0000   
     
                                            channel=Paid Search  \
     channel=Paid Search AND sessions>=4                 0.4535   
     channel=Paid Search AND device=Mobile               0.7068   
     channel=Organic AND sessions<=2                     0.0000   
     channel=Paid Search AND tenure>=12                  0.3698   
     channel=Paid Search                                 1.0000   
     channel=Organic AND market=Lima                     0.0000   
     
                                            channel=Organic AND market=Lima  
     channel=Paid Search AND sessions>=4                             0.0000  
     channel=Paid Search AND device=Mobile                           0.0000  
     channel=Organic AND sessions<=2                                 0.2182  
     channel=Paid Search AND tenure>=12                              0.0000  
     channel=Paid Search                                             0.0000  
     channel=Organic AND market=Lima                                 1.0000  )



### Reading the Subgroup Discovery diagnostics

`absolute_shrinkage > 0` indicates that the discovered magnitude became smaller on holdout—a direct view of winner's curse. Negative values can occur by chance and are not proof of anti-overfitting. The Jaccard matrix reveals when several impressive rows describe almost the same people.

For confirmatory use, freeze the rule list before holdout, report all frozen rules rather than only survivors, and use simultaneous or multiplicity-adjusted inference when making a family of claims. For exploratory use, label the table as hypothesis generation.

## 4. RuleFit: a sparse model built from tree rules

RuleFit first grows a tree ensemble to generate nonlinear rules, then combines rule indicators with linear terms in a sparse regression:

$$\hat f(x)=\beta_0+\sum_{j=1}^{p}\beta_j l_j(x)
+\sum_{k=1}^{K}\alpha_k r_k(x),$$

where $l_j(x)$ are winsorized/scaled linear features and $r_k(x)\in\{0,1\}$ are rules extracted from tree nodes or leaves. An $L_1$ penalty selects a small subset:

$$\min_{\beta,\alpha}\frac1n\sum_i
(Z_i-\hat f(X_i))^2+\lambda(\|\beta\|_1+\|\alpha\|_1).$$

Compared with one CART tree, RuleFit is usually more stable and flexible; compared with an unrestricted boosting model, it produces an inspectable sparse rule list. Correlated rules can substitute for each other, so coefficient rank is not a causal or uniquely identified importance ordering.

### Original RuleFit construction

Friedman and Popescu's procedure has three conceptual stages:

1. fit a stochastic tree ensemble, often varying tree sizes so both low- and higher-order interactions appear;
2. extract a rule from every non-root tree node, not only terminal leaves;
3. combine rule indicators with winsorized linear terms and fit a sparse linear model.

For continuous feature $x_j$, the linear term is winsorized at declared quantiles and scaled so its typical variation is comparable with a rule. A common representation is

$$l_j(x_j)=0.4\frac{x_j^*-\bar x_j^*}{sd(x_j^*)},$$

where $x_j^*$ is winsorized. The factor (0.4) approximates the average standard deviation of a binary rule under typical support, reducing arbitrary penalty differences between linear and rule terms.

### Global and local rule importance

For rule $r_k$ with coefficient $\alpha_k$ and empirical support $s_k$, global importance is

$$I_k=|\alpha_k|\sqrt{s_k(1-s_k)}.$$

This corrects raw coefficient magnitude for how much the binary rule varies. A local contribution at observation $x$ is

$$I_k(x)=\alpha_k\{r_k(x)-s_k\}.$$

Because (r_k\) is Bernoulli with variance (s_k(1-s_k)),

$$\operatorname{Var}[I_k(X)]
=\alpha_k^2s_k(1-s_k),$$

so (I_k=|\alpha_k|\sqrt{s_k(1-s_k)}) is exactly the standard deviation of the rule's centered contribution across the sample. This is why a huge coefficient attached to an almost-always-true rule can have modest global importance.

### Why the Lasso produces zeros

Under the convention

$$\min_\theta \frac{1}{2n}\|Z-H\theta\|_2^2+\lambda\|\theta\|_1,$$

the Karush–Kuhn–Tucker condition for a coefficient to remain zero is

$$\left|\frac1nH_k^\top(Z-H\widehat\theta)\right|\le\lambda.$$

In words, a rule enters only when its correlation with the current residual exceeds the penalty. When two rules are highly correlated, fitting one can reduce the other's residual correlation below the threshold. This gives a statistical explanation for coefficient instability among near-duplicate rules.

These are predictive-model diagnostics. They do not conserve an observed period change, and correlated rules can redistribute coefficients when the sample or penalty changes.

### Tuning and validation

Tune ensemble size/depth, subsampling, winsorization, and the Lasso penalty within discovery folds. Evaluate the final rule model on untouched data. In change discovery, use time-aware folds when seasonality or drift matters; random folds can make unstable rules appear reproducible.

The implementation below is pedagogical and RuleFit-inspired: gradient-boosted leaf rules plus scaled encoded linear terms followed by `LassoCV`. A production implementation should use repeated validation, explicit winsorization, rule deduplication, stability selection, and a supported RuleFit library.


```python
gb = GradientBoostingRegressor(
    n_estimators=40, max_depth=2, min_samples_leaf=120,
    learning_rate=.05, random_state=42
).fit(X_train, train.change_target)

train_leaves = gb.apply(X_train).astype(int).reshape(len(train), -1)
holdout_leaves = gb.apply(X_holdout).astype(int).reshape(len(holdout), -1)
leaf_encoder = OneHotEncoder(handle_unknown='ignore')
R_train = leaf_encoder.fit_transform(train_leaves)
R_holdout = leaf_encoder.transform(holdout_leaves)

scaler = StandardScaler()
L_train = scaler.fit_transform(X_train)
L_holdout = scaler.transform(X_holdout)
design_train = sparse.hstack([sparse.csr_matrix(L_train), R_train], format='csr')
design_holdout = sparse.hstack([sparse.csr_matrix(L_holdout), R_holdout], format='csr')

rulefit = LassoCV(cv=5, random_state=42, max_iter=20_000).fit(
    design_train, train.change_target
)
rule_coefs = rulefit.coef_[L_train.shape[1]:]

def describe_leaf(tree, target_leaf, names):
    # Recover the conjunction leading to one terminal leaf.
    def walk(node, path):
        if node == target_leaf:
            return path
        if tree.children_left[node] == tree.children_right[node]:
            return None
        feature = names[tree.feature[node]]
        threshold = tree.threshold[node]
        left = walk(tree.children_left[node], path + [f'{feature} <= {threshold:.3g}'])
        if left is not None:
            return left
        return walk(tree.children_right[node], path + [f'{feature} > {threshold:.3g}'])
    return ' AND '.join(walk(0, []) or [])

rule_descriptions = []
for estimator_number, leaf_ids in enumerate(leaf_encoder.categories_):
    tree = gb.estimators_[estimator_number, 0].tree_
    rule_descriptions.extend(
        describe_leaf(tree, leaf_id, feature_names) for leaf_id in leaf_ids
    )

rule_support = np.asarray(R_train.mean(axis=0)).ravel()
top_rules = pd.DataFrame({
    'rule': rule_descriptions,
    'coefficient': rule_coefs,
    'support': rule_support,
})
top_rules = top_rules.loc[top_rules.coefficient.ne(0)].assign(
    rule_importance=lambda x: (
        x.coefficient.abs()*np.sqrt(x.support*(1-x.support))
    )
).sort_values('rule_importance', ascending=False).head(12)

pd.Series({
    'selected linear terms': np.count_nonzero(rulefit.coef_[:L_train.shape[1]]),
    'selected leaf rules': np.count_nonzero(rule_coefs),
    'train MSE': mean_squared_error(train.change_target,
                                    rulefit.predict(design_train)),
    'holdout proxy MSE': mean_squared_error(
        holdout.converted * (holdout.period/holdout.period.mean()
        - (1-holdout.period)/(1-holdout.period.mean())),
        rulefit.predict(design_holdout)),
}), top_rules
```




    (selected linear terms    5.0000
     selected leaf rules     63.0000
     train MSE                0.3341
     holdout proxy MSE        0.3460
     dtype: float64,
                                                       rule  coefficient  support  \
     47   cat__channel_Paid Search > 0.5 AND cat__device...       0.0718   0.2617   
     3    cat__channel_Paid Search > 0.5 AND num__sessio...       0.0592   0.1755   
     49   num__sessions_30d <= 3.5 AND num__sessions_30d...      -0.0406   0.5058   
     66   cat__channel_Organic > 0.5 AND num__sessions_3...      -0.0522   0.1175   
     117  num__tenure_months <= 29.9 AND num__tenure_mon...       0.0678   0.0602   
     131  cat__channel_Organic <= 0.5 AND num__tenure_mo...       0.0310   0.2772   
     43   num__sessions_30d > 3.5 AND num__tenure_months...       0.0689   0.0370   
     120  cat__channel_Paid Search <= 0.5 AND num__tenur...      -0.0676   0.0367   
     51   num__sessions_30d > 3.5 AND num__tenure_months...       0.0318   0.1442   
     56   num__sessions_30d <= 3.5 AND num__tenure_month...      -0.0538   0.0443   
     55   cat__channel_Organic > 0.5 AND num__tenure_mon...      -0.0419   0.0468   
     65   cat__channel_Organic <= 0.5 AND num__tenure_mo...       0.0361   0.0643   
     
          rule_importance  
     47            0.0315  
     3             0.0225  
     49            0.0203  
     66            0.0168  
     117           0.0161  
     131           0.0139  
     43            0.0130  
     120           0.0127  
     51            0.0112  
     56            0.0111  
     55            0.0089  
     65            0.0089  )




```python
# Verify that support-adjusted importance is the rule contribution's SD.
top_index = int(top_rules.index[0])
top_rule_values = R_train[:, top_index].toarray().ravel()
top_coefficient = rule_coefs[top_index]
top_support = top_rule_values.mean()
centered_rule_contribution = top_coefficient*(top_rule_values-top_support)
pd.Series({
    'formula importance': abs(top_coefficient)*np.sqrt(
        top_support*(1-top_support)
    ),
    'empirical contribution SD': centered_rule_contribution.std(ddof=0),
    'identity error': (
        abs(top_coefficient)*np.sqrt(top_support*(1-top_support))
        - centered_rule_contribution.std(ddof=0)
    ),
})
```




    formula importance          0.0315
    empirical contribution SD   0.0315
    identity error              0.0000
    dtype: float64



### Reading the RuleFit output

The table ranks selected leaf rules by support-adjusted importance, not merely by coefficient size. A positive coefficient means the rule raises the fitted change target conditional on all other selected terms; it is not the rule's standalone period change. Inspect the raw baseline/comparison rates for any rule before giving it business meaning.

This demonstration differs from canonical RuleFit in four ways: it extracts terminal leaves only, uses a fixed shallow boosting depth, scales encoded linear columns without explicit winsorization, and does not deduplicate logically equivalent rules. Those simplifications keep the mechanism visible but should be removed or documented in production research.

### Why RuleFit and Subgroup Discovery can disagree

Subgroup Discovery scores each rule largely on its own. RuleFit estimates coefficients jointly, so one rule can absorb variation that another overlapping rule would explain marginally. CART, meanwhile, forces a single exhaustive partition. Disagreement is therefore expected and useful: it exposes dependence on the search space and estimand rather than revealing which algorithm found the one “true” segment.

## Method comparison

| Method | Search structure | Split/selection criterion | Output | Main risk |
|---|---|---|---|---|
| Decision tree | recursive partition | algorithm-dependent | exhaustive leaves | instability |
| CART | binary recursive partition | impurity/SSE reduction plus pruning | one tree | greedy cutpoints and overfit |
| CHAID | categorical multiway partition | adjusted chi-square tests and category merging | one significance-driven tree | binning, low expected counts, repeated testing |
| Subgroup Discovery | overlapping rule search | support × unusualness quality | ranked local patterns | huge search space and redundancy |
| RuleFit | ensemble-generated rules + linear terms | predictive loss with sparsity penalty | sparse additive rule model | correlated rules and coefficient instability |

The methods are complementary. CART asks for one partition of everyone. CHAID emphasizes categorical interactions and inferential screening. Subgroup Discovery seeks exceptional, possibly overlapping niches. RuleFit trades the simplicity of one tree for a more stable sparse ensemble.

## Econometric point of view

### Description versus causal heterogeneity

The conditional period contrast

$$E[Y\mid T=1,X=x]-E[Y\mid T=0,X=x]$$

answers where outcomes changed among observed populations. A conditional average treatment effect instead requires potential outcomes:

$$\tau_{\mathrm{causal}}(x)=E[Y(1)-Y(0)\mid X=x].$$

Equating them requires a well-defined intervention plus identification assumptions such as random assignment or conditional exchangeability, positivity, consistency, and no interference. If `period` is merely calendar time, those assumptions usually fail.

For experiments, causal trees and causal forests modify splitting and estimation to target treatment-effect heterogeneity. Ordinary CART on outcomes, a period interaction screen, or RuleFit on observational change does not become causal because it discovers an intuitive segment.

### Robust workflow

1. Declare outcome, periods, population, features, and minimum commercially relevant change.
2. Exclude post-period or post-treatment features that create leakage.
3. Reserve a final untouched evaluation sample or future time window.
4. Search with minimum support in **both** periods.
5. Report baseline rate, comparison rate, difference, interval, and counts.
6. Correct or disclose multiplicity; use false-discovery control when testing many fixed candidates.
7. Test stability across seeds, time windows, cutpoints, and nearby rule definitions.
8. Translate replicated descriptive patterns into hypotheses for experiments or quasi-experiments.

## Limitations

- Adaptive rules exaggerate effects unless evaluated honestly.
- Rare subgroups can show extreme but noisy changes.
- Missing categories and changing measurement definitions can masquerade as patterns.
- Trees are discontinuous: observations around a cutpoint may receive very different labels.
- CHAID p-values depend on bins, expected cell counts, merging rules, and the search procedure.
- RuleFit coefficients are conditional on a large correlated dictionary and are not unique causal contributions.
- Overlapping Subgroup Discovery rules cannot be summed into an aggregate decomposition.
- A subgroup can be predictively useful but operationally unactionable or ethically inappropriate.

## Exercises

1. Change the CART minimum leaf size from 50 to 800. Plot discovered change against holdout change.
2. Reverse the discovery and holdout time windows. Which rules remain stable?
3. Extend the beam search to depth three and quantify the winner's curse.
4. Replace the quality score with a lower confidence bound. How does ranking change?
5. Bin tenure three different ways before the CHAID-style screen.
6. Simulate a campaign randomized within period and compare descriptive period heterogeneity with causal treatment heterogeneity.
7. Add channel-mix drift without any within-cell change. Determine which methods confuse composition with conditional performance.

## Takeaways

1. Pattern mining localizes change; it does not allocate the aggregate change like Kitagawa or Shapley.
2. CART, CHAID, Subgroup Discovery, and RuleFit encode different search spaces and notions of an interesting subgroup.
3. Support, holdout replication, uncertainty, multiplicity, and stability belong in every report.
4. “The change is concentrated here” is descriptive. “This segment responds to intervention” requires a causal design.

## References

- Morgan, J. N., & Sonquist, J. A. (1963). Problems in the analysis of survey data, and a proposal. *Journal of the American Statistical Association*, 58, 415–434.
- Kass, G. V. (1980). An exploratory technique for investigating large quantities of categorical data. *Applied Statistics*, 29, 119–127. https://doi.org/10.2307/2986296
- Breiman, L., Friedman, J. H., Olshen, R. A., & Stone, C. J. (1984). *Classification and Regression Trees*. Wadsworth.
- Klösgen, W. (1996). Explora: A multipattern and multistrategy discovery assistant. In *Advances in Knowledge Discovery and Data Mining*.
- Wrobel, S. (1997). An algorithm for multi-relational discovery of subgroups. In *PKDD 1997*.
- Friedman, J. H., & Popescu, B. E. (2008). Predictive learning via rule ensembles. *Annals of Applied Statistics*, 2, 916–954. https://doi.org/10.1214/07-AOAS148
- Herrera, F., Carmona, C. J., González, P., & del Jesus, M. J. (2011). An overview on subgroup discovery. *Knowledge and Information Systems*, 29, 495–525. https://doi.org/10.1007/s10115-010-0356-2
- Athey, S., & Imbens, G. (2016). Recursive partitioning for heterogeneous causal effects. *PNAS*, 113, 7353–7360. https://doi.org/10.1073/pnas.1510489113
- Wager, S., & Athey, S. (2018). Estimation and inference of heterogeneous treatment effects using random forests. *JASA*, 113, 1228–1242. https://doi.org/10.1080/01621459.2017.1319839
