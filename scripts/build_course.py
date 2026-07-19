"""Build and execute the decomposition-analysis notebook course."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks"
OUT.mkdir(exist_ok=True)

SETUP = """import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from itertools import permutations, combinations
pd.options.display.float_format = '{:,.4f}'.format
plt.style.use('seaborn-v0_8-whitegrid')
rng = np.random.default_rng(42)"""

def md(text): return nbf.v4.new_markdown_cell(text.strip())
def code(text): return nbf.v4.new_code_cell(text.strip())

def lesson(title, objectives, theory, cells, refs):
    return [
        md(f"# {title}\n\n**Graduate course: Decomposition Analysis in Python**\n\n## Learning objectives\n\n{objectives}"),
        md(theory), code(SETUP), *cells,
        md("## Interpretation checklist\n\n1. State the mathematical identity or estimand.\n2. Verify exactness numerically.\n3. Separate description, prediction, and causation.\n4. Report reference population/path/order.\n5. Quantify sampling uncertainty when inputs are estimated."),
        md("## References\n\n" + refs),
    ]

notebooks = {}

notebooks['00_field_map.ipynb'] = lesson(
"00 — What decomposition analysis is",
"- Distinguish an accounting identity from a statistical estimand.\n- Place the major traditions in a defensible genealogy.\n- Use a taxonomy based on the object being decomposed.",
r"""## A family of frameworks, not one unified discipline

A decomposition maps a total, change, gap, distribution, or model prediction into labelled components. The shared requirement is usually **efficiency**: $\sum_j C_j=\Delta$. It does not, by itself, identify causes.

The proposed single lineage in the research brief is historically misleading. Several branches developed partly independently: demographic standardization (Kitagawa, Das Gupta); index numbers and Divisia methods; regional shift–share; labor-market mean/distribution decompositions; input–output structural decomposition; cooperative-game allocations; and statistical/ML function explanations. SHAP borrows Shapley axioms, but is not a descendant of Oaxaca–Blinder or LMDI.

| Object | Representative methods | Typical claim |
|---|---|---|
| aggregate rate | Kitagawa, Das Gupta, stepwise replacement | mix/rate contribution |
| mean or distribution gap | Oaxaca–Blinder, reweighting, RIF | composition/structure gap |
| aggregate identity | PVM, Laspeyres, Fisher, LMDI, SDA | factor contribution |
| value function | Shapley, Aumann–Shapley | axiom-based allocation |
| fitted function | functional ANOVA, SHAP, PDP/ALE | predictive explanation |
| potential outcomes / SCM | DiD, IV, RD, DML | causal effect under assumptions |""",
[
code("""taxonomy = pd.DataFrame({
    'method': ['Kitagawa', 'Oaxaca–Blinder', 'LMDI', 'Shapley', 'SHAP',
               'Difference-in-Differences'],
    'decomposes': ['rate change', 'mean gap', 'aggregate change', 'value change',
                   'prediction', 'potential-outcome contrast'],
    'causal_by_itself': [False, False, False, False, False, False],
    'key_choice': ['two-period average', 'reference coefficients', 'log-mean path',
                   'coalition value', 'background distribution', 'parallel trends'],
})
taxonomy"""),
md("### Exercise\n\nFor a statement such as “revenue was driven by price,” identify the total, comparison, allocation rule, uncertainty, and causal assumptions. Rewrite it as a defensible descriptive claim.")],
"""- Kitagawa, E. M. (1955). Components of a difference between two rates. *JASA*, 50, 1168–1194. https://doi.org/10.1080/01621459.1955.10501299
- Das Gupta, P. (1993). *Standardization and Decomposition of Rates*. U.S. Census Bureau.
- Shorrocks, A. F. (2013). Decomposition procedures for distributional analysis. *Journal of Economic Inequality*, 11, 99–126. https://doi.org/10.1007/s10888-011-9214-z
- Pearl, J. (2009). *Causality* (2nd ed.). Cambridge University Press.""")

notebooks['01_rates_kitagawa_dasgupta.ipynb'] = lesson(
"01 — Rate decomposition: Kitagawa, Das Gupta, and replacement",
"- Derive the two-factor rate decomposition.\n- Recognize path dependence with three or more factors.\n- Implement stepwise, Das Gupta, and Shorrocks all-orders allocations.\n- Interpret Chevan–Sutherland category-level refinements.\n- Handle entrant/exit segments and compare direct with chained multiperiod decompositions.",
r"""## Kitagawa identity: intuition

For group $g$, the aggregate rate is $R_t=\sum_g w_{gt}r_{gt}$. Kitagawa's symmetric allocation is
$$C_w=\sum_g(w_{g1}-w_{g0})(r_{g1}+r_{g0})/2,$$
$$C_r=\sum_g(r_{g1}-r_{g0})(w_{g1}+w_{g0})/2.$$
Then $C_w+C_r=R_1-R_0$ exactly. These are composition and within-group-rate components—not causal effects. Das Gupta generalized standardization to several factors. Stepwise replacement is exact but generally order-dependent; averaging all orders produces a Shapley-style symmetric allocation.""",
[
code("""# One row per stable customer segment; shares sum to one in each period.
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
d, summary"""),
code("""segment_contributions = d.set_index('segment')[['mix', 'rate']]
ax = segment_contributions.plot.bar(color=['#4C78A8', '#F58518'])
ax.axhline(0, color='black', linewidth=.8)
ax.set_ylabel('contribution to aggregate rate change')
plt.show()"""),
md("""### Exercises

1. Bootstrap observations within segments and form percentile intervals for Kitagawa components. Why is exactness not statistical certainty?
2. Build a channel × device cross-classification and produce Chevan–Sutherland-style category contributions. Identify offsetting categories.
3. Apply the generic Shorrocks function to a four-factor funnel and approximate contributions by sampling permutations rather than enumerating $4!$ orders.
4. Add one entrant and one disappearing segment. Compare a separate entry/exit component with three reference-rate assumptions.
5. Simulate 12 monthly periods. Compare direct annual and chained monthly components, including gross positive and negative contributions.
6. Design a hierarchical Shorrocks/Owen grouping for acquisition and product factors. Explain why flat and grouped allocations may differ.""")],
"""- Kitagawa, E. M. (1955). *JASA*, 50, 1168–1194. https://doi.org/10.1080/01621459.1955.10501299
- Das Gupta, P. (1978). A general method of decomposing a difference between two rates into several components. *Demography*, 15, 99–112. https://doi.org/10.2307/2060493
- Chevan, A., & Sutherland, M. (2009). Revisiting Das Gupta: Refinement and extension of standardization and decomposition. *Demography*, 46, 429–449. https://doi.org/10.1353/dem.0.0060""")

notebooks['02_index_numbers_lmdi_pvm_sda.ipynb'] = lesson(
"02 — Index numbers, LMDI, PVM, shift–share, and SDA",
"- Derive and compare Laspeyres, Paasche, and Fisher indexes.\n- Prove additive LMDI exactness and implement additive and multiplicative interpretations.\n- Distinguish price, volume, genuine mix, and interaction conventions.\n- Derive classical shift–share and interpret its benchmark-relative residual.\n- Derive a symmetric two-factor structural decomposition using the Leontief model.",
r"""## Overview: identities and allocation rules

If emissions $E=\sum_i Q_i I_i$, the additive LMDI uses the logarithmic mean $L(a,b)=(a-b)/(\log a-\log b)$:
$$\Delta E_Q=\sum_iL(E_{i1},E_{i0})\log(Q_{i1}/Q_{i0}),\quad
\Delta E_I=\sum_iL(E_{i1},E_{i0})\log(I_{i1}/I_{i0}).$$
It is exact for positive values and has no residual. PVM applies the same identity logic to revenue $\sum_i p_iq_i$, but common business conventions allocate the interaction differently. Laspeyres uses base weights; Fisher is the geometric mean of Laspeyres and Paasche indexes. Shift–share partitions regional change into national, industry-mix, and regional-competitive terms. SDA applies polar/average or all-permutation rules to input–output identities.""",
[
code("""x = pd.DataFrame({
    'product': ['A', 'B', 'C'],
    'q0': [100, 80, 50],
    'q1': [115, 70, 66],
    'i0': [2.0, 3.0, 5.0],
    'i1': [1.8, 3.4, 4.6],
})
x['E0'] = x.q0 * x.i0
x['E1'] = x.q1 * x.i1

def L(a,b):
    # Logarithmic mean, using its continuous value when a equals b.
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return np.where(np.isclose(a, b), a, (a - b) / (np.log(a) - np.log(b)))

log_weights = L(x.E1, x.E0)
x['activity'] = log_weights * np.log(x.q1 / x.q0)
x['intensity'] = log_weights * np.log(x.i1 / x.i0)

lmdi_summary = pd.Series({
    'observed': x.E1.sum() - x.E0.sum(),
    'activity': x.activity.sum(),
    'intensity': x.intensity.sum(),
    'residual': (x.E1 - x.E0 - x.activity - x.intensity).sum(),
})
x, lmdi_summary"""),
code("""p0 = np.array([10., 16., 25.])
p1 = np.array([11., 15., 27.])
q0 = x.q0.to_numpy()
q1 = x.q1.to_numpy()

price = (p1 - p0) * q0
volume = p0 * (q1 - q0)
interaction = (p1 - p0) * (q1 - q0)

pvm = pd.DataFrame({
    'product': x['product'],
    'price': price,
    'volume': volume,
    'interaction': interaction,
}).set_index('product')
pvm.plot.bar()
plt.ylabel('revenue contribution')
plt.show()"""),
md("""### Exercises

1. Verify Fisher's time-reversal test by swapping periods and multiplying the forward and reverse indexes.
2. Implement multiplicative LMDI for the growth funnel and verify that the factor indexes multiply to $Y_1/Y_0$.
3. Decompose product revenue into total volume, product mix, price, and interactions using two alternative conventions.
4. Change the shift–share portfolio benchmark and base year. Which component is most sensitive, and why?
5. Calculate both SDA polar decompositions before averaging them. Explain which factor receives the interaction in each path.
6. For a paid-marketing question, write separately an exact accounting decomposition and a causal estimand for incrementality.""")],
"""- Diewert, W. E. (1976). Exact and superlative index numbers. *Journal of Econometrics*, 4, 115–145. https://doi.org/10.1016/0304-4076(76)90009-9
- Ang, B. W. (2005). The LMDI approach to decomposition analysis. *Energy Policy*, 33, 867–871. https://doi.org/10.1016/j.enpol.2003.10.010
- Dunn, E. S. (1960). A statistical and analytical technique for regional analysis. *Papers of the RSA*, 6, 97–112.
- Rose, A., & Casler, S. (1996). Input–output structural decomposition analysis. *Economic Systems Research*, 8, 33–62. https://doi.org/10.1080/09535319600000003""")

notebooks['03_oaxaca_reweighting_rif.ipynb'] = lesson(
"03 — Oaxaca–Blinder and distributional decompositions",
"- Decompose a mean gap into composition and coefficient terms.\n- See why the reference structure matters.\n- Connect reweighting and RIF regressions to distributional statistics.",
r"""## Mean-gap decomposition

With group means $\bar X_A,\bar X_B$ and linear coefficients $\hat\beta_A,\hat\beta_B$,
$$\bar Y_A-\bar Y_B=(\bar X_A-\bar X_B)'\hat\beta_B+\bar X_A'(\hat\beta_A-\hat\beta_B).$$
The first term is often called composition/endowment and the second structure/coefficient. Labels such as “explained” and “unexplained” do not establish explanation, discrimination, or causality. Results depend on the reference coefficients, included covariates, functional form, common support, and selection.

DiNardo–Fortin–Lemieux reweights an entire distribution using density ratios. Firpo–Fortin–Lemieux regress the recentered influence function (RIF) to decompose unconditional quantiles and other distributional statistics.""",
[
code("""import statsmodels.api as sm
n=1500; g=rng.binomial(1,.48,n); experience=np.clip(rng.normal(8+2*g,3,n),0,None); education=rng.normal(15+.7*g,1.5,n)
y=2+0.09*experience+0.14*education+g*(.18+.025*experience)+rng.normal(0,.45,n)
df=pd.DataFrame({'y':y,'group':g,'experience':experience,'education':education})
fits={k:sm.OLS(z.y,sm.add_constant(z[['experience','education']])).fit() for k,z in df.groupby('group')}
xbar={k:np.r_[1,z[['experience','education']].mean()] for k,z in df.groupby('group')}
b0,b1=fits[0].params.to_numpy(),fits[1].params.to_numpy(); gap=df[df.group==1].y.mean()-df[df.group==0].y.mean()
composition=(xbar[1]-xbar[0])@b0; structure=xbar[1]@(b1-b0)
pd.Series({'mean gap':gap,'composition (group 0 reference)':composition,'structure':structure,'identity error':gap-composition-structure})"""),
code("""comp_alt=(xbar[1]-xbar[0])@b1; struct_alt=xbar[0]@(b1-b0)
pd.DataFrame({'group 0 reference':[composition,structure], 'group 1 reference':[comp_alt,struct_alt]}, index=['composition','structure'])"""),
md("### Exercise\n\nAdd an interaction and nonlinear term to the data-generating process. Compare linear OB results with a flexible outcome model and discuss specification dependence.")],
"""- Oaxaca, R. (1973). Male–female wage differentials in urban labor markets. *International Economic Review*, 14, 693–709. https://doi.org/10.2307/2525981
- Blinder, A. S. (1973). Wage discrimination: Reduced form and structural estimates. *Journal of Human Resources*, 8, 436–455. https://doi.org/10.2307/144855
- DiNardo, J., Fortin, N. M., & Lemieux, T. (1996). Labor market institutions and the distribution of wages. *Econometrica*, 64, 1001–1044. https://doi.org/10.2307/2171954
- Firpo, S., Fortin, N. M., & Lemieux, T. (2009). Unconditional quantile regressions. *Econometrica*, 77, 953–973. https://doi.org/10.3982/ECTA6822""")

notebooks['04_shapley_anova_ml.ipynb'] = lesson(
"04 — Shapley, Aumann–Shapley, functional ANOVA, and SHAP",
"- Derive exact Shapley values from random-order marginal contributions.\n- Prove efficiency and interpret the Shapley axioms.\n- Derive Aumann–Shapley as a line-integral allocation.\n- Construct functional ANOVA and variance components.\n- Derive SHAP for model predictions and separate predictive from causal attribution.",
r"""## Cooperative-game allocation

For value function $v(S)$ and $p$ players,
$$\phi_j=\sum_{S\subseteq N\setminus\{j\}}\frac{|S|!(p-|S|-1)!}{p!}[v(S\cup\{j\})-v(S)].$$
The allocation is efficient, symmetric, additive, and assigns zero to null players. Aumann–Shapley is a continuous-path analogue. Functional ANOVA decomposes a square-integrable function into main and interaction functions relative to an input distribution.

SHAP applies Shapley values to model predictions. The coalition value depends on a background distribution and on whether “missing” features are integrated conditionally or interventionally. Therefore a SHAP value is a model-and-background-specific predictive attribution, not automatically a causal effect.""",
[
code("""features=['traffic','conversion','price']
base={'traffic':1000.,'conversion':.04,'price':50.}; current={'traffic':1200.,'conversion':.05,'price':48.}
def revenue(z): return z['traffic']*z['conversion']*z['price']
def value(S):
    z={k:(current[k] if k in S else base[k]) for k in features}; return revenue(z)-revenue(base)
phi={j:0. for j in features}
for order in permutations(features):
    S=set()
    for j in order:
        phi[j]+=(value(S|{j})-value(S))/np.math.factorial(len(features)); S.add(j)
pd.Series({**phi,'allocated':sum(phi.values()),'observed':revenue(current)-revenue(base)})""".replace('np.math.factorial','__import__("math").factorial')),
code("""pd.Series(phi).plot.bar(color=['#4C78A8','#54A24B','#E45756']); plt.axhline(0,color='black',lw=.8); plt.ylabel('Shapley revenue contribution'); plt.show()"""),
md("""### Exercises

1. Change the revenue baseline and recompute discrete Shapley values. Which axioms remain true, and which business interpretation changes?
2. Replace the straight Aumann–Shapley path with a path that changes traffic first and price last. Compare contributions and verify efficiency.
3. Modify the functional ANOVA example so $X_2$ is correlated with $X_1$. Show numerically why the classical variance components no longer add cleanly.
4. Change the SHAP background mean for the linear model. Verify local accuracy and explain why the feature attributions move.
5. For one growth intervention, define separately a descriptive Shapley game, a predictive SHAP game, and a causal estimand.""")],
"""- Shapley, L. S. (1953). A value for n-person games. In *Contributions to the Theory of Games II*. Princeton University Press.
- Aumann, R. J., & Shapley, L. S. (1974). *Values of Non-Atomic Games*. Princeton University Press.
- Hoeffding, W. (1948). A class of statistics with asymptotically normal distribution. *Annals of Mathematical Statistics*, 19, 293–325.
- Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *NeurIPS 30*, 4765–4774. https://papers.nips.cc/paper/7062""")

notebooks['05_econometrics_causality_driver_trees.ipynb'] = lesson(
"05 — Econometrics, causal inference, and defensible driver claims",
"- Distinguish contribution, association, prediction, and causal effect.\n- Demonstrate confounding in a regression.\n- Build a KPI driver tree without overstating causality.",
r"""## Different questions, different estimands

An identity decomposition explains how an observed arithmetic change is allocated. Regression estimates conditional associations unless a design and assumptions identify a causal estimand. Potential-outcome methods target contrasts such as $E[Y(1)-Y(0)]$; structural causal models encode interventions $do(X=x)$.

| Claim | Minimum object | Typical requirement |
|---|---|---|
| contribution | identity/value function | explicit allocation rule |
| association | joint distribution/model | specification and sampling assumptions |
| prediction | out-of-sample loss | representative validation |
| causal effect | potential outcomes/SCM | exchangeability or valid design |

OLS, fixed effects, DiD, synthetic control, matching, IV, RD, DML, causal forests, and BSTS do not become causal merely by name. Each needs its own identifying assumptions. A driver tree is a semantic/accounting graph; it becomes causal only when its edges are supported by a causal model and identification strategy.""",
[
code("""n=3000
quality=rng.normal(size=n); campaign=(rng.random(n)<1/(1+np.exp(-1.4*quality))).astype(int)
sales=10+3*quality+1.5*campaign+rng.normal(size=n)
naive=sm.OLS(sales,sm.add_constant(campaign)).fit()
adjusted=sm.OLS(sales,sm.add_constant(np.c_[campaign,quality])).fit()
pd.Series({'true treatment effect':1.5,'naive association':naive.params[1],'adjusted coefficient':adjusted.params[1]})"""),
code("""criteria=pd.DataFrame({
'question':['What changed arithmetically?','What predicts Y?','What would Y be under intervention?'],
'tool':['LMDI/PVM/Shapley','validated ML/regression','RCT, DiD, IV, RD, SCM, etc.'],
'safe wording':['contributed under rule R','predictively associated','caused under assumptions A']})
criteria"""),
md("### Capstone\n\nFor a conversion-rate decline, produce (i) a Kitagawa mix/rate decomposition, (ii) a predictive model with held-out error, and (iii) a causal design for one actionable lever. Keep the three conclusions separate.")],
"""- Rubin, D. B. (1974). Estimating causal effects of treatments in randomized and nonrandomized studies. *Journal of Educational Psychology*, 66, 688–701. https://doi.org/10.1037/h0037350
- Pearl, J. (2009). *Causality* (2nd ed.). Cambridge University Press.
- Angrist, J. D., & Pischke, J.-S. (2009). *Mostly Harmless Econometrics*. Princeton University Press.
- Chernozhukov, V. et al. (2018). Double/debiased machine learning. *The Econometrics Journal*, 21, C1–C68. https://doi.org/10.1111/ectj.12097
- Athey, S., & Imbens, G. W. (2016). Recursive partitioning for heterogeneous causal effects. *PNAS*, 113, 7353–7360. https://doi.org/10.1073/pnas.1510489113""")

# Deepening blocks are kept separate from the introductory narrative so each
# lesson has the same scholarly structure and can be expanded independently.
ENRICHMENTS = {
'00_field_map.ipynb': [
md(r"""## Formal vocabulary: object, contrast, rule, estimand

Let $T(P)$ be a target functional of a population or empirical distribution $P$. A comparison is

$$\Delta_T = T(P_1)-T(P_0).$$

A decomposition rule $\mathcal A$ maps $(P_0,P_1,T)$ into contributions $C_1,\ldots,C_K$. **Efficiency** requires

$$\sum_{k=1}^K C_k=\Delta_T.$$

Efficiency is an accounting property. It does not imply that $C_k$ is unique, consistently estimated, policy-invariant, or causal. Identification asks whether the observed-data law determines the target. Estimation asks how a sample is used to approximate it. Attribution asks how a known total is allocated. These are different operations."""),
md(r"""## Econometrics point of view

An econometric analysis should declare an estimand before choosing an estimator. A decomposition can operate on observed statistics, fitted conditional means, counterfactual distributions, or identified causal effects. Only the last case inherits a causal interpretation, and only under the assumptions that identify those effects.

| Layer | Example target | Main uncertainty | Defensible verb |
|---|---|---|---|
| accounting | $R_1-R_0$ | measurement/revisions | contributed |
| descriptive statistical | $T(\hat P_1)-T(\hat P_0)$ | sampling | associated/composed |
| predictive | $f(x)-E[f(X)]$ | generalization/model | predicted/attributed |
| causal | $E[Y(1)-Y(0)]$ | identification + sampling | caused |

### Growth-marketing use case

For $\text{Revenue}=\text{Traffic}\times\text{CVR}\times\text{AOV}$, attribution answers how the realized revenue change is distributed across these factors. It does **not** answer what revenue would have been if a campaign manager had intervened on traffic while holding the rest of the system at its post-intervention equilibrium."""),
md(r"""## Descriptive versus causal counterfactuals

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

Therefore, “descriptive” does not mean useless or unsophisticated. A descriptive counterfactual can be exact, decision-relevant, and mathematically rigorous. It simply answers a different question from a causal counterfactual."""),
md(r"""## Limitations and failure modes

- **Non-uniqueness:** interactions admit multiple exact allocations.
- **Baseline dependence:** changing period, control group, reference model, or background distribution changes contributions.
- **Aggregation bias:** segment-level decompositions may reverse after aggregation (Simpson-type behavior).
- **Generated-regressor uncertainty:** fitted rates, coefficients, or propensities must carry estimation error forward.
- **No policy invariance:** accounting identities need not survive interventions because other factors respond.
- **Semantic overreach:** “driver” often conflates arithmetic contribution, predictive relevance, and causal effect.

## What came next

Kitagawa (1955) was generalized to multiple factors by **Prithwis Das Gupta (1978, 1993)**. Mean decompositions by **Oaxaca and Blinder (1973)** were extended from means toward counterfactual distributions by **Juhn, Murphy & Pierce (1993)** and **DiNardo, Fortin & Lemieux (1996)**, and toward unconditional quantiles by **Firpo, Fortin & Lemieux (2009)**. Cooperative-game allocations developed by **Shapley (1953)** later informed **Shorrocks (2013)** for distributional decomposition and **Lundberg & Lee (2017)** for model explanations. These are convergences around allocation principles, not one linear genealogy.""")],

'01_rates_kitagawa_dasgupta.ipynb': [
md(r"""## Kitagawa: derivation and proof of exactness

We want to decompose the change in the aggregate rate

$$R_t=\sum_{g=1}^{G}w_{gt}r_{gt},\qquad \sum_{g=1}^{G}w_{gt}=1.$$

Define $\Delta w_g=w_{g1}-w_{g0}$ and $\Delta r_g=r_{g1}-r_{g0}$. For one segment,

$$w_{g1}r_{g1}-w_{g0}r_{g0}
=(w_{g0}+\Delta w_g)(r_{g0}+\Delta r_g)-w_{g0}r_{g0}.$$

Expanding the product gives

$$w_{g1}r_{g1}-w_{g0}r_{g0}
=r_{g0}\Delta w_g+w_{g0}\Delta r_g+\Delta w_g\Delta r_g.$$

The first term is a pure mix change evaluated at the baseline rate; the second is a pure rate change evaluated at the baseline share; the third is an interaction because both quantities changed. Kitagawa allocates half of that interaction to each component:

$$C_{w,g}=r_{g0}\Delta w_g+\frac12\Delta w_g\Delta r_g
=\Delta w_g\frac{r_{g0}+r_{g1}}{2},$$

$$C_{r,g}=w_{g0}\Delta r_g+\frac12\Delta w_g\Delta r_g
=\Delta r_g\frac{w_{g0}+w_{g1}}{2}.$$

Adding the two components for one segment recovers its exact observed change:

$$C_{w,g}+C_{r,g}
=r_{g0}\Delta w_g+w_{g0}\Delta r_g+\Delta w_g\Delta r_g
=w_{g1}r_{g1}-w_{g0}r_{g0}.$$

Finally, summing over all segments proves the aggregate identity:

$$\boxed{R_1-R_0=\sum_g C_{w,g}+\sum_g C_{r,g}}.$$

The 50/50 interaction split makes the decomposition symmetric under reversal of periods. It is a defensible convention, not a uniquely identified scientific effect."""),
md(r"""## Growth-marketing case: conversion rate

Let $w_{gt}$ be the traffic share of channel $g$ and $r_{gt}$ its conversion rate. Then total CVR is $R_t=\sum_g w_{gt}r_{gt}$.

- **Mix contribution:** traffic moved toward channels with higher/lower CVR.
- **Rate contribution:** within-channel CVR changed.
- **Actionable diagnostic:** split further by device, geography, landing page, or cohort, but check sparse cells and post-treatment segmentation.

Example claim for the worked data: “Total CVR rose 2.51 pp; under the Kitagawa two-period rule, +1.43 pp is allocated to segment mix and +1.08 pp to within-segment rates.” This is precise and descriptive. “The new campaign caused +1.08 pp” is not supported without an experiment or credible quasi-experiment."""),
code(r"""# Aggregation check: collapse Returning and Enterprise.
# The collapsed rate must be share-weighted separately in each period.
fine = d.copy()
collapsed = pd.DataFrame({
    'segment': ['New', 'Established'],
    'w0': [fine.loc[0, 'w0'], fine.loc[1:, 'w0'].sum()],
    'w1': [fine.loc[0, 'w1'], fine.loc[1:, 'w1'].sum()],
    'r0': [fine.loc[0, 'r0'],
           np.average(fine.loc[1:, 'r0'], weights=fine.loc[1:, 'w0'])],
    'r1': [fine.loc[0, 'r1'],
           np.average(fine.loc[1:, 'r1'], weights=fine.loc[1:, 'w1'])],
})
collapsed['mix'] = (collapsed.w1-collapsed.w0)*(collapsed.r1+collapsed.r0)/2
collapsed['rate'] = (collapsed.r1-collapsed.r0)*(collapsed.w1+collapsed.w0)/2

comparison = pd.DataFrame({
    'fine segmentation': [R0, R1, R1-R0, d['mix'].sum(), d['rate'].sum()],
    'collapsed segmentation': [
        (collapsed.w0*collapsed.r0).sum(),
        (collapsed.w1*collapsed.r1).sum(),
        (collapsed.w1*collapsed.r1).sum()-(collapsed.w0*collapsed.r0).sum(),
        collapsed['mix'].sum(),
        collapsed['rate'].sum(),
    ],
}, index=['R0','R1','total change','mix','rate'])
comparison"""),
md(r"""## Is the result consistent when segments are aggregated?

There are two different consistency questions.

### 1. Is the aggregate rate preserved? Yes—if aggregation is done correctly

For a block $H$ containing several fine segments, define

$$w_{Ht}=\sum_{g\in H}w_{gt},\qquad
r_{Ht}=\frac{\sum_{g\in H}w_{gt}r_{gt}}{w_{Ht}}.$$

Then $w_{Ht}r_{Ht}=\sum_{g\in H}w_{gt}r_{gt}$. Therefore $R_0$, $R_1$, and $R_1-R_0$ are exactly unchanged after collapsing segments. A simple unweighted average of rates would fail this property.

### 2. Are the mix and rate components preserved? Generally no

The fine decomposition evaluates every $\Delta w_g$ against its own average rate. The collapsed decomposition first creates a changing within-block weighted rate $r_{Ht}$ and then treats that block as one segment. Variation in shares **inside** $H$ is no longer visible as mix; part of it is absorbed into the block's rate component. Thus the total remains exact, while the labels `mix` and `rate` are reallocated.

This is not a coding inconsistency. It is a lack of **aggregation invariance**. The decomposition answers a question conditional on the chosen partition. Report the segmentation rule and repeat the analysis at substantively plausible levels."""),
md(r"""## Stepwise Replacement: the bridge from Kitagawa to Das Gupta

Kitagawa has only two changing objects—shares and rates—and resolves their interaction by splitting it equally. When a function contains three or more changing factors, a simple alternative is to replace them **one at a time**.

Let

$$Y_t=F(x_{1t},x_{2t},\ldots,x_{Kt}).$$

Choose one order, for example $x_1\rightarrow x_2\rightarrow\cdots\rightarrow x_K$. Define hybrid states

$$z^{(0)}=(x_{10},x_{20},\ldots,x_{K0}),$$

$$z^{(1)}=(x_{11},x_{20},\ldots,x_{K0}),$$

$$z^{(2)}=(x_{11},x_{21},\ldots,x_{K0}),$$

and continue until $z^{(K)}=(x_{11},\ldots,x_{K1})$. The contribution assigned at step $j$ is

$$C_j^{(\pi)}=F\!\left(z^{(j)}\right)-F\!\left(z^{(j-1)}\right),$$

where $\pi$ denotes the selected order. Adding all steps produces a telescoping sum:

$$\sum_{j=1}^{K}C_j^{(\pi)}
=[F(z^{(1)})-F(z^{(0)})]
+[F(z^{(2)})-F(z^{(1)})]+\cdots
+[F(z^{(K)})-F(z^{(K-1)})].$$

Every intermediate value appears once with a positive sign and once with a negative sign, so the sum telescopes:

so

$$\boxed{\sum_j C_j^{(\pi)}=Y_1-Y_0}.$$

Thus every stepwise path is **exact**. But when factors interact, the individual $C_j^{(\pi)}$ depend on the order $\pi$, because a factor replaced later operates on a different hybrid state."""),
code(r"""# One exact stepwise decomposition for Revenue = Traffic × CVR × AOV.
stepwise_base = {'traffic': 1_000., 'cvr': .04, 'aov': 50.}
stepwise_final = {'traffic': 1_200., 'cvr': .05, 'aov': 48.}

def revenue_identity(values):
    return values['traffic'] * values['cvr'] * values['aov']

def stepwise_decomposition(base_values, final_values, order):
    # Replace factors in `order` and return their successive increments.
    state = base_values.copy()
    rows = []
    for factor in order:
        before = revenue_identity(state)
        state[factor] = final_values[factor]
        after = revenue_identity(state)
        rows.append({
            'factor replaced': factor,
            'value before': before,
            'value after': after,
            'contribution': after - before,
        })
    return pd.DataFrame(rows)

order_a = ['traffic', 'cvr', 'aov']
order_b = ['aov', 'cvr', 'traffic']
path_a = stepwise_decomposition(stepwise_base, stepwise_final, order_a)
path_b = stepwise_decomposition(stepwise_base, stepwise_final, order_b)

order_comparison = pd.DataFrame({
    'Traffic → CVR → AOV': path_a.set_index('factor replaced')['contribution'],
    'AOV → CVR → Traffic': path_b.set_index('factor replaced')['contribution'],
}).reindex(['traffic', 'cvr', 'aov'])

path_a, order_comparison, order_comparison.sum().rename('total change')"""),
md(r"""## Reading the Stepwise output

For the order Traffic → CVR → AOV, the contributions are 400, 600, and −120. For the reverse-style order AOV → CVR → Traffic, they are 480, 480, and −80. Both columns sum to the same observed revenue change, 880.

The difference is entirely due to interactions. For example, the traffic increase is worth 400 when evaluated at baseline CVR and AOV, but 480 when evaluated after CVR and AOV have already changed. Neither order is algebraically wrong; the problem is that an arbitrary order gives an arbitrary interaction allocation.

This motivates Das Gupta's symmetric construction: calculate the stepwise marginal contribution along every possible order and average them."""),
md(r"""## Das Gupta: from two factors to many factors

Kitagawa treats two changing objects: shares and rates. **Das Gupta (1978)** developed standardization and decomposition for a rate or function depending on several factors.

Let

$$F_t=F(x_{1t},x_{2t},\ldots,x_{Kt})$$

and consider a replacement order $\pi$. Start with all factors at period 0. Replace them one at a time with period-1 values. The marginal contribution of factor $j$ along order $\pi$ is

$$M_j^{(\pi)}=F\!\left(x^{(\pi,j,+)}\right)
-F\!\left(x^{(\pi,j,-)}\right),$$

where $x^{(\pi,j,-)}$ is the hybrid state immediately before replacing $j$ and $x^{(\pi,j,+)}$ is the state immediately after. Every path telescopes:

$$\sum_{j=1}^{K}M_j^{(\pi)}=F_1-F_0.$$

A symmetric contribution averages over all $K!$ orders:

$$C_j=\frac{1}{K!}\sum_{\pi}M_j^{(\pi)},\qquad
\sum_j C_j=F_1-F_0.$$

This averaging removes order dependence and distributes all higher-order interactions. In modern terminology it is closely connected to a Shapley–Shorrocks allocation. Das Gupta's demographic framework also uses standardized rates to isolate factors while preserving the marginal structures relevant to the application."""),
code(r"""# Das Gupta / all-orders replacement for a 3-factor growth funnel.
# Revenue per impression = CTR × post-click CVR × AOV.
from math import factorial

factor_names = ['ctr', 'post_click_cvr', 'aov']
period0 = {'ctr': .020, 'post_click_cvr': .040, 'aov': 55.}
period1 = {'ctr': .023, 'post_click_cvr': .037, 'aov': 58.}

def funnel_value(values):
    return values['ctr'] * values['post_click_cvr'] * values['aov']

contributions = {name: 0.0 for name in factor_names}
path_rows = []
orders = list(permutations(factor_names))

for order in orders:
    state = period0.copy()
    path_contributions = {}
    for name in order:
        before = funnel_value(state)
        state[name] = period1[name]
        after = funnel_value(state)
        marginal = after - before
        contributions[name] += marginal / len(orders)
        path_contributions[name] = marginal
    path_rows.append({'order': ' → '.join(order), **path_contributions})

das_gupta_summary = pd.Series({
    **contributions,
    'allocated': sum(contributions.values()),
    'observed': funnel_value(period1) - funnel_value(period0),
})
pd.DataFrame(path_rows), das_gupta_summary"""),
md(r"""## Reading the Das Gupta output

The first table shows that the marginal assigned to a factor changes with its replacement order; this is the interaction problem. The summary averages each factor's marginal across all six orders. `allocated` equals `observed`, so the symmetric multifactor decomposition is exact. The units are revenue per impression, because that is the function supplied to `funnel_value`.

Changing the function changes the scientific question. Multiplying by impressions would decompose total revenue; keeping revenue per impression deliberately removes traffic scale and focuses on funnel efficiency."""),
md(r"""## Chevan & Sutherland (2009): categorical refinement of Das Gupta

### What problem did they solve?

Das Gupta's cross-classification framework decomposes a difference in aggregate rates into effects associated with compositional variables and an overall rate effect. In many applications, a variable-level result is too coarse. Knowing that “channel composition” matters does not reveal whether Search, Social, Email, or Affiliate generated the contribution.

Chevan and Sutherland make explicit a refinement already latent in the cross-classified framework: decompose both composition and rate effects down to the **categories** of each compositional variable. The categorical effects are additive:

$$C_w=\sum_g C_{w,g},\qquad C_r=\sum_g C_{r,g},$$

and therefore

$$R_1-R_0=\sum_g(C_{w,g}+C_{r,g}).$$

With one composition variable, these are exactly the segment-level Kitagawa contributions already calculated in the worked example. The contribution of Chevan–Sutherland is most valuable with several cross-classified variables, where effects can be reported both by variable and by category.

### Cross-classified interpretation

Suppose conversion is cross-classified by channel $i$ and device $j$:

$$R_t=\sum_i\sum_j w_{ijt}r_{ijt}.$$

A complete report can show:

- total composition and rate effects;
- channel-level category effects;
- device-level category effects;
- cell-level diagnostics that reveal offsetting contributions hidden by variable totals.

Category effects may be large and opposite in sign even when their variable-level total is small. This is analogous to the aggregation issue demonstrated earlier: cancellation at a coarse level can hide substantively important movements.

### Polytomous response variables

Chevan–Sutherland also extend the orientation from a binary/rate outcome to a response with categories $k=1,\ldots,K$, such as subscription status {free, trial, paid, churned}. For each composition cell $(i,j)$, define

$$t_{ijkt}=\frac{N_{ijkt}}{\sum_kN_{ijkt}},\qquad \sum_kt_{ijkt}=1.$$

These cell percentages replace the scalar rate in separate decompositions for each response category. The results distinguish:

- changes in the composition of users across channel/device cells;
- changes in the propensity to occupy each response category inside a cell.

Because response-category percentages sum to one, gains in some categories are balanced by losses in others. The paper further demonstrates extensions to the standard deviation and the multivariate index of dissimilarity.

### What the refinement does not solve

It does not make category effects causal, choose theoretically relevant control variables, solve sparse cross-classified cells, or guarantee aggregation invariance. Chevan and Sutherland explicitly emphasize that any selected categorical variable will yield a result; scientific meaning depends on theoretically justified variable selection."""),
code(r"""# Chevan–Sutherland categorical reporting in the one-variable case.
# Segment contributions add to variable-level composition and rate effects.
category_report = d.set_index('segment')[['mix', 'rate']].copy()
category_report['total category contribution'] = category_report.sum(axis=1)
category_report.loc['variable-level total'] = category_report.sum(axis=0)
category_report.loc['observed aggregate change', 'total category contribution'] = R1-R0
category_report"""),
md(r"""### Reading the categorical report

The table exposes offsetting category effects. For example, the New segment has a negative mix effect but a positive rate effect. Reporting only the aggregate mix and rate totals would hide this internal structure.

The variable-level total is the sum of category rows, demonstrating Chevan–Sutherland additivity in this simplified case. With several composition variables, use cross-classified cells and aggregate category contributions carefully; do not compute separate one-way decompositions and add them, because that can double-count interactions."""),
md(r"""## Shorrocks (2013): a unified Shapley decomposition

### General indicator and activation/neutralization game

Shorrocks begins with an indicator completely determined by $m$ factors:

$$I=f(X_1,X_2,\ldots,X_m).$$

Define $K=\{1,\ldots,m\}$ and use one consistent convention for the set function $F(S)$: factors in $S$ are active at their target values, while factors outside $S$ are placed at declared neutral or baseline values. Thus $F(\varnothing)$ is fully neutralized and $F(K)$ is the complete target indicator. The scientific content lies partly in that rule: neutralizing growth might mean setting a growth factor to zero, while neutralizing an income source might mean replacing it by zero or by its mean.

For factor $k\notin S$, its marginal effect is

$$\Delta_kF(S)=F(S\cup\{k\})-F(S).$$

One elimination sequence yields an exact telescoping decomposition but is path-dependent. Shorrocks averages over all $m!$ sequences:

$$\boxed{C_k^{\mathrm{Sh}}=
\sum_{S\subseteq K\setminus\{k\}}
\frac{|S|!(m-|S|-1)!}{m!}\Delta_kF(S)}.$$

This is formally the Shapley value of the decomposition game.

### Key properties

- **Exactness/efficiency:** $\sum_kC_k^{\mathrm{Sh}}=F(K)-F(\varnothing)$.
- **Symmetry:** equivalent factors receive equal contributions.
- **Expected marginal interpretation:** $C_k^{\mathrm{Sh}}$ is the mean marginal impact of $k$ over random activation paths.
- **No residual:** interactions are absorbed symmetrically into factor contributions.
- **Generality:** the indicator need not be additively decomposable and factors can be categorical, continuous, distributions, or model components.

Shorrocks applies the rule to poverty growth/redistribution, subgroup poverty, inequality by subgroups, and inequality by income sources. In some classical cases it reproduces standard practice; in others it removes arbitrary residual terms.

### Hierarchies and the Owen value

Factors often have a hierarchy—for example Acquisition = {Search, Social} and Product = {Activation, Retention}. A two-stage Owen procedure first allocates across groups and then within each group.

Flat Shapley and hierarchical Owen allocations need not coincide. Shorrocks shows that hierarchical aggregation consistency generally fails; an important exception arises when the function is **separable** over the grouped factors, meaning their marginal contributions do not depend on factors outside the group in the relevant way.

This formalizes the segment-aggregation result seen earlier: grouping is a substantive modeling choice, not merely a display operation."""),
code(r"""# Reusable Shorrocks all-orders decomposition for a hybrid-state function.
def shorrocks_all_orders(base_values, final_values, value_function):
    factor_names = list(base_values)
    orders = list(permutations(factor_names))
    contributions = {name: 0.0 for name in factor_names}

    for order in orders:
        state = base_values.copy()
        for name in order:
            before = value_function(state)
            state[name] = final_values[name]
            after = value_function(state)
            contributions[name] += (after-before)/len(orders)

    observed = value_function(final_values)-value_function(base_values)
    return pd.Series({
        **contributions,
        'allocated': sum(contributions.values()),
        'observed': observed,
        'identity error': sum(contributions.values())-observed,
    })

shorrocks_all_orders(period0, period1, funnel_value)"""),
md(r"""### Das Gupta versus Shorrocks

The algorithms overlap but their scopes and motivations differ:

| Dimension | Das Gupta | Chevan–Sutherland | Shorrocks |
|---|---|---|---|
| Primary origin | demographic standardization | refinement of demographic cross-classification | distributional economics/cooperative games |
| Main object | differences in rates/means with composition | variable and category effects; polytomous distributions | any indicator $I=f(X_1,\ldots,X_m)$ |
| Core operation | standardize and replace factors | reveal additive category detail | average marginal elimination effects |
| Interaction handling | symmetric standardization/replacement | additive categorical refinement | Shapley averaging over every path |
| Hierarchies | not the central focus | variables and categories | explicit Owen/two-stage analysis |
| Causal by itself? | no | no | no |

Shorrocks supplies the broad axiomatic umbrella. Das Gupta and Chevan–Sutherland supply demographic structure and category-level interpretations that a generic Shapley formula does not choose automatically."""),
md(r"""## When a segment is absent in one period

Suppose segment $g$ is absent at baseline but appears later:

$$w_{g0}=0,\qquad w_{g1}>0.$$

Its baseline rate $r_{g0}$ is not observed because the denominator is zero. The segment's total contribution to the aggregate change is nevertheless identified algebraically:

$$w_{g1}r_{g1}-w_{g0}r_{g0}=w_{g1}r_{g1}.$$

But the Kitagawa split is not identified without choosing a reference $r_{g0}^*$:

$$C_{w,g}=w_{g1}\frac{r_{g1}+r_{g0}^*}{2},$$

$$C_{r,g}=w_{g1}\frac{r_{g1}-r_{g0}^*}{2}.$$

For any $r_{g0}^*$, the sum remains $w_{g1}r_{g1}$, but the mix/rate allocation changes. Setting the missing rate to zero is therefore not neutral; it splits entry equally between mix and rate.

### Recommended reporting choices

1. **Entry/exit component:** report $w_{g1}r_{g1}$ as a separate entrant contribution without claiming a mix/rate split. This is the most transparent default.
2. **Declared reference rate:** use a theoretically justified benchmark such as the portfolio rate, a comparable segment, or a model-based prediction; show sensitivity.
3. **Pool time or cells:** appropriate only when zero counts reflect sampling sparsity rather than a genuinely nonexistent segment.
4. **Smoothing/modeling:** shrink sparse rates using a binomial/hierarchical model and propagate uncertainty. Do not confuse an estimated counterfactual rate with an observed rate.

Distinguish a **structural zero** (the segment could not exist) from a **sampling zero** (it existed but no observations were recorded). The appropriate treatment differs."""),
code(r"""# Entrant segment: exact total, reference-sensitive mix/rate split.
w0_entry, w1_entry = 0.0, 0.08
r1_entry = 0.24
reference_rates = [0.0, 0.12, 0.18, 0.24]

entry_sensitivity = []
for r0_reference in reference_rates:
    mix_entry = (w1_entry-w0_entry)*(r1_entry+r0_reference)/2
    rate_entry = (r1_entry-r0_reference)*(w1_entry+w0_entry)/2
    entry_sensitivity.append({
        'assumed baseline rate': r0_reference,
        'mix': mix_entry,
        'rate': rate_entry,
        'allocated total': mix_entry+rate_entry,
        'identified entrant total': w1_entry*r1_entry,
    })
pd.DataFrame(entry_sensitivity)"""),
md(r"""### Reading the entrant sensitivity table

The allocated total is identical in every row, while the labels mix and rate move with the assumed missing baseline rate. If the reference equals the entrant's observed rate, all contribution is classified as mix. If the reference is zero, the contribution is split equally. Neither result is data-identified.

This is an important difference between **exactness** and **identification**: the identity closes perfectly even when the internal allocation depends on an unobserved quantity."""),
md(r"""## More than two periods: direct versus chained decomposition

For periods $t=0,1,\ldots,T$, there are two principal approaches.

### Direct endpoint comparison

Apply Kitagawa to $0$ and $T$:

$$R_T-R_0=C_w^{0,T}+C_r^{0,T}.$$

This provides a compact long-run comparison but ignores intermediate reversals, temporary segments, and the timing of change.

### Adjacent-period chaining

Decompose each transition:

$$R_t-R_{t-1}=C_w^{t-1,t}+C_r^{t-1,t}.$$

Summing across time is exactly additive:

$$R_T-R_0=\sum_{t=1}^{T}C_w^{t-1,t}
+\sum_{t=1}^{T}C_r^{t-1,t}.$$

The total is the same, but chained mix/rate contributions generally differ from the direct endpoint split. This is **path dependence across time**, not an arithmetic error. Chaining preserves the actual sequence and can record entry/exit at the link where it occurs—but only after applying an explicit entry/exit or reference-rate rule. Direct comparison answers a cleaner endpoint question.

### Recommendation

- Use adjacent-period decompositions for monitoring and operational narratives.
- Show the direct endpoint decomposition as a robustness comparison.
- Keep segment definitions stable or document taxonomy bridges.
- Report gross positive and negative contributions as well as the net; a factor can rise and later reverse.
- Do not average away chronological order when timing is substantively meaningful.
- For statistical rates, include uncertainty and multiple-comparison considerations across periods.

Multiperiod decomposition remains descriptive. A time series of contributions is not a causal event study."""),
code(r"""# Direct versus chained Kitagawa decomposition across four periods.
panel = pd.DataFrame({
    'period': np.repeat([0,1,2,3], 3),
    'segment': ['New','Returning','Enterprise']*4,
    'weight': [.50,.35,.15,  .44,.38,.18,  .39,.40,.21,  .42,.36,.22],
    'rate':   [.08,.18,.31,  .10,.17,.33,  .09,.20,.35,  .11,.19,.36],
})

def kitagawa_pair(left, right):
    joined = left.merge(right, on='segment', suffixes=('0','1'), how='outer')
    joined[['weight0','weight1']] = joined[['weight0','weight1']].fillna(0)
    # Rates must not be filled blindly when a segment is absent; this panel has none.
    if joined[['rate0','rate1']].isna().any().any():
        raise ValueError('Absent segment rate requires an entry/exit rule.')
    mix = ((joined.weight1-joined.weight0)*(joined.rate1+joined.rate0)/2).sum()
    rate = ((joined.rate1-joined.rate0)*(joined.weight1+joined.weight0)/2).sum()
    return pd.Series({'mix':mix, 'rate':rate, 'total':mix+rate})

period_data = {t:g[['segment','weight','rate']] for t,g in panel.groupby('period')}
direct = kitagawa_pair(period_data[0], period_data[3])
links = pd.DataFrame([
    kitagawa_pair(period_data[t-1],period_data[t]).rename(f'{t-1}→{t}')
    for t in range(1,4)
])
comparison_multiperiod = pd.DataFrame({
    'direct 0→3': direct,
    'sum of adjacent links': links.sum(),
})
links, comparison_multiperiod"""),
md(r"""### Reading the multiperiod output

Each adjacent row is an exact decomposition of that period's change. Their totals telescope to the direct $0\rightarrow3$ change. However, the accumulated mix and rate columns need not match the direct mix and rate components because the intermediate weights and rates affect the chained allocation.

Neither answer dominates universally. Direct decomposition is endpoint-oriented; chaining is history-oriented. Present both when the managerial story depends on when changes occurred."""),
md(r"""## Decision guide: which version should you use?

| Analytical need | Recommended method | Required disclosure |
|---|---|---|
| Two-period aggregate rate, stable segments | Kitagawa | segment definition and 50/50 interaction rule |
| More than two changing factors | Das Gupta/all-orders replacement | factorization, baseline, and permutations |
| Need contributions by category | Chevan–Sutherland refinement | cross-classification, sparse cells, category definitions |
| Arbitrary indicator or formal hierarchy | Shorrocks/Owen | neutralization game, hierarchy, separability assumptions |
| New or disappearing segment | separate entry/exit component | structural versus sampling zero; any reference rate |
| Operational monitoring over time | chained adjacent-period Kitagawa | link definitions, taxonomy changes, gross and net effects |
| Long-run endpoint narrative | direct endpoint Kitagawa | omitted intermediate reversals and entry/exit history |

Across every row, exactness means that allocated components reproduce the chosen total. It does not guarantee unique labels, statistical precision, aggregation invariance, or causal interpretation."""),
md(r"""## Limitations, robustness, and inference

- Shares must sum to one in each period and segments must be defined consistently.
- New/disappearing categories and small cells require explicit handling.
- Results depend on segmentation; always repeat at plausible aggregation levels.
- The total is aggregation-consistent under weighted aggregation, but the component split is generally not aggregation-invariant.
- Das Gupta/all-orders replacement costs $K!$ paths if implemented literally; sampling permutations is needed for many factors.
- Chevan–Sutherland category detail can become unstable or disclosure-sensitive in sparse cross-classified cells.
- Entrant and exit segments have unidentified missing-period rates; exact totals do not identify a mix/rate split.
- Chained multiperiod components are history-dependent and can differ from direct endpoint components.
- Observed rate changes combine treatment, seasonality, selection, composition within cells, and noise.
- For estimated rates, use a stratified bootstrap or delta method; exactness conditional on estimates is not zero sampling variance.
- If channel is affected by treatment, conditioning on it can create post-treatment bias in a causal analysis.

## What came next

**Das Gupta (1978)** extended the logic to several compositional factors and later systematized it in his 1993 monograph. **Chevan & Sutherland (2009)** refined cross-classified decompositions to reveal additive category-level composition and rate effects, polytomous response distributions, standard deviations, and dissimilarity. **Shorrocks (2013; first draft 1999)** supplied a unified Shapley framework for arbitrary indicators, including hierarchical/two-stage Owen decompositions and formal conditions related to separability and aggregation. Later computational work uses permutation sampling and model-specific shortcuts when exhaustive paths are infeasible.""")],

'02_index_numbers_lmdi_pvm_sda.ipynb': [
md(r"""## 1. Index numbers: Laspeyres, Paasche, and Fisher

### The problem

For several products, total expenditure or revenue is

$$V_t=\sum_{i=1}^n p_{it}q_{it}.$$

The value ratio $V_1/V_0$ combines price and quantity changes. An index-number system seeks price and quantity indexes satisfying approximately or exactly

$$\frac{V_1}{V_0}=P(0,1)Q(0,1).$$

### Fixed-weight indexes

The Laspeyres price index values current prices using baseline quantities:

$$P_L=\frac{\sum_i p_{i1}q_{i0}}{\sum_i p_{i0}q_{i0}}.$$

The Paasche price index uses current quantities:

$$P_P=\frac{\sum_i p_{i1}q_{i1}}{\sum_i p_{i0}q_{i1}}.$$

Their quantity counterparts are

$$Q_L=\frac{\sum_i p_{i0}q_{i1}}{\sum_i p_{i0}q_{i0}},\qquad
Q_P=\frac{\sum_i p_{i1}q_{i1}}{\sum_i p_{i1}q_{i0}}.$$

Laspeyres tends to overweight goods that were important in the base period; Paasche uses the comparison-period basket and may reflect substitution that already occurred. Neither is universally “correct”: they answer different basket questions.

### Fisher's symmetric index and factor reversal

Fisher takes geometric means:

$$P_F=\sqrt{P_LP_P},\qquad Q_F=\sqrt{Q_LQ_P}.$$

Because $P_LQ_P=V_1/V_0$ and $P_PQ_L=V_1/V_0$,

$$P_FQ_F=\sqrt{P_LP_PQ_LQ_P}=\frac{V_1}{V_0}.$$

Thus Fisher passes the **factor-reversal test** exactly. It also passes the time-reversal test: reversing periods gives the reciprocal index. Diewert (1976) called Fisher “superlative” because it is exact for a flexible quadratic aggregator and provides a second-order approximation to broad preference/technology structures.

Index numbers summarize relative change; they do not estimate demand elasticities or causal price effects."""),
md(r"""## Business and growth setting

This notebook uses a common hierarchy:

$$\text{Revenue}=\sum_i p_iq_i,$$

and, for a digital funnel,

$$\text{Revenue}=\text{Impressions}\times\text{CTR}
\times\text{CVR}\times\text{AOV}.$$

Index numbers summarize price and quantity change across products. PVM allocates an absolute revenue change. LMDI allocates multiplicative driver changes. Shift–share compares growth across channels or markets. SDA handles systems in which outputs depend on matrices of interdependencies.

A second identity,

$$\text{Profit}=\text{Revenue}-\text{Spend},\qquad
\text{Spend}=\text{Clicks}\times\text{CPC},$$

prevents a favorable gross-revenue attribution from being mistaken for incremental profit."""),
code(r"""# Compare bilateral price and quantity indexes on the same products.
index_data = pd.DataFrame({
    'product': ['A', 'B', 'C'],
    'p0': [10., 16., 25.],
    'p1': [11., 15., 27.],
    'q0': [100., 80., 50.],
    'q1': [115., 70., 66.],
})

p0_i, p1_i = index_data.p0, index_data.p1
q0_i, q1_i = index_data.q0, index_data.q1
V0 = (p0_i*q0_i).sum()
V1 = (p1_i*q1_i).sum()

P_L = (p1_i*q0_i).sum() / V0
P_P = V1 / (p0_i*q1_i).sum()
Q_L = (p0_i*q1_i).sum() / V0
Q_P = V1 / (p1_i*q0_i).sum()
P_F = np.sqrt(P_L*P_P)
Q_F = np.sqrt(Q_L*Q_P)

pd.Series({
    'value ratio V1/V0': V1/V0,
    'Laspeyres price': P_L,
    'Paasche price': P_P,
    'Fisher price': P_F,
    'Laspeyres quantity': Q_L,
    'Paasche quantity': Q_P,
    'Fisher quantity': Q_F,
    'Fisher factor-reversal error': P_F*Q_F - V1/V0,
})"""),
md(r"""### Reading the index-number output

Laspeyres and Paasche differ because they use different baskets. Fisher lies between them and its price index multiplied by its quantity index reproduces the observed value ratio, as shown by the near-zero factor-reversal error.

For pricing analytics, report which basket is used. A statement such as “prices increased 4%” is incomplete without the index formula, product coverage, treatment of new/disappearing products, and whether indexes are bilateral or chained."""),
md(r"""## 2. Divisia and LMDI

### From infinitesimal change to finite decomposition

For a differentiable aggregate $Y=F(x_1,\ldots,x_K)$, a Divisia decomposition follows the total differential

$$dY=\sum_{k=1}^K\frac{\partial F}{\partial x_k}dx_k,$$

or in logarithmic form

$$d\log Y=\sum_k\frac{\partial\log F}{\partial\log x_k}d\log x_k.$$

This identity is exact for infinitesimal changes. Discrete data require an integration or index approximation. For an additive aggregate

$$E_t=\sum_i E_{it},\qquad E_{it}=\prod_{k=1}^K x_{kit},$$

define the logarithmic mean

$$L(a,b)=\begin{cases}
\dfrac{a-b}{\log a-\log b},&a\ne b,\\[4pt]
a,&a=b.
\end{cases}$$

The additive LMDI contribution of factor $k$ is

$$\boxed{\Delta E_k=\sum_iL(E_{i1},E_{i0})
\log\left(\frac{x_{ki1}}{x_{ki0}}\right)}.$$

### Proof of perfect decomposition

For each component,

$$\sum_k\log\left(\frac{x_{ki1}}{x_{ki0}}\right)
=\log\left(\frac{E_{i1}}{E_{i0}}\right).$$

Therefore

$$\sum_k\Delta E_k
=\sum_iL(E_{i1},E_{i0})\log(E_{i1}/E_{i0})
=\sum_i(E_{i1}-E_{i0})=E_1-E_0.$$

The second equality follows directly from the definition of $L$. Hence LMDI has no unexplained residual for positive data.

The multiplicative form uses

$$D_k=\exp\left(\sum_i\frac{L(E_{i1},E_{i0})}{L(E_1,E_0)}
\log\frac{x_{ki1}}{x_{ki0}}\right),\qquad
\prod_kD_k=\frac{E_1}{E_0}.$$

Additive and multiplicative forms express the same comparison in absolute and ratio units, respectively."""),
code(r"""# Growth funnel: exact LMDI allocation of revenue change
f0 = pd.Series({'impressions': 1_000_000., 'ctr': .020, 'cvr': .040, 'aov': 55.})
f1 = pd.Series({'impressions': 1_150_000., 'ctr': .023, 'cvr': .037, 'aov': 58.})
y0, y1 = f0.prod(), f1.prod()
weight = L(np.array([y1]), np.array([y0]))[0]
funnel_contrib = weight * np.log(f1/f0)
funnel_result = pd.concat([
    funnel_contrib.rename('absolute contribution'),
    (100*funnel_contrib/(y1-y0)).rename('share of change (%)'),
], axis=1)
funnel_result.loc['sum / observed change'] = [funnel_contrib.sum(), 100.]
funnel_result
"""),
md(r"""### Reading the growth-funnel LMDI

The contributions sum to the observed revenue change. Impressions, CTR, and AOV contribute positively; CVR contributes negatively because it falls. A negative CVR contribution does not prove that a campaign harmed conversion—it describes the arithmetic role of the observed CVR movement.

Growth applications include revenue waterfalls, CAC identities, energy-like decompositions of marketing spend, and cohort LTV factorizations. Always state units and factorization. `Impressions × CTR × CVR × AOV` and `Clicks × orders-per-click × AOV` are algebraically compatible but create different reporting levels."""),
md(r"""## 3. Price–Volume–Mix (PVM)

### One product

For revenue $R=pq$,

$$\Delta R=q_0\Delta p+p_0\Delta q+\Delta p\Delta q.$$

The three terms are baseline-weighted price, baseline-weighted volume, and interaction. Three common exact conventions are:

1. **Explicit interaction:** report all three terms.
2. **Price-first:** $C_p=q_0\Delta p$ and $C_q=p_1\Delta q$.
3. **Symmetric:** split interaction equally:

$$C_p=\Delta p\frac{q_0+q_1}{2},\qquad
C_q=\Delta q\frac{p_0+p_1}{2}.$$

All sum to $\Delta R$, but individual contributions differ.

### Multiple products and genuine mix

Write $q_i=Qs_i$, where $Q=\sum_iq_i$ is total units and $s_i=q_i/Q$ is product share. Then

$$R=Q\sum_i p_is_i=Q\bar p.$$

A change in product shares is distinct from a change in total units. A useful decomposition must separate total volume from mix rather than labeling every product-level quantity change “mix.” Because $Q$, $s$, and $p$ interact, the analyst must state the interaction rule—base weights, symmetric polar average, LMDI, or all-orders/Shapley."""),
code(r"""# Compare three exact PVM conventions for one product.
p0_one, p1_one = 50., 48.
q0_one, q1_one = 40., 60.
dp, dq = p1_one-p0_one, q1_one-q0_one
observed_change = p1_one*q1_one - p0_one*q0_one

pvm_conventions = pd.DataFrame({
    'explicit interaction': [q0_one*dp, p0_one*dq, dp*dq],
    'price first': [q0_one*dp, p1_one*dq, 0.],
    'symmetric split': [dp*(q0_one+q1_one)/2,
                        dq*(p0_one+p1_one)/2, 0.],
}, index=['price','volume','interaction'])
pvm_conventions.loc['allocated total'] = pvm_conventions.sum()
pvm_conventions.loc['observed change'] = observed_change
pvm_conventions"""),
md(r"""### PVM applications and econometric interpretation

PVM is useful for financial planning, SKU revenue bridges, subscription plan migration, and separating AOV from order growth. In paid growth, an analogous bridge can separate CPC, clicks, conversion, and order value.

Limitations:

- Results depend on interaction convention and base period.
- New/discontinued products require explicit entry/exit treatment.
- Discounts, returns, taxes, and currency effects can make “price” ambiguous.
- Observed price and quantity are jointly determined; the price contribution is not a demand curve or elasticity.
- Product-mix shifts can be endogenous to promotions, availability, and customer selection.

For a causal pricing question, estimate what quantity and mix would have occurred under the alternative price policy; do not freeze observed quantity mechanically."""),
md(r"""## 4. Shift–share decomposition

Shift–share was developed for regional employment analysis but transfers naturally to channels, geographies, products, and customer segments. Let $E_{ir0}$ be baseline activity in industry/channel $i$ and region/business unit $r$. Define national/portfolio growth $g$, category growth $g_i$, and local category growth $g_{ir}$.

The classical change is decomposed as

$$\Delta E_{ir}=
\underbrace{E_{ir0}g}_{\text{overall growth}}
+\underbrace{E_{ir0}(g_i-g)}_{\text{category mix}}
+\underbrace{E_{ir0}(g_{ir}-g_i)}_{\text{local competitive shift}}.$$

The proof is immediate:

$$g+(g_i-g)+(g_{ir}-g_i)=g_{ir}.$$

Multiplying by $E_{ir0}$ recovers $E_{ir0}g_{ir}=E_{ir1}-E_{ir0}$.

The “competitive” residual is a benchmark-relative descriptive term. It is not evidence that local management or a campaign caused superior growth."""),
code(r"""# Marketing shift–share: conversions by channel in one region.
shift_share = pd.DataFrame({
    'channel': ['Search', 'Social', 'Email'],
    'local_0': [500., 300., 200.],
    'local_1': [570., 390., 218.],
    'portfolio_0': [5_000., 4_000., 2_000.],
    'portfolio_1': [5_400., 4_800., 2_100.],
})
portfolio_total_growth = (
    shift_share.portfolio_1.sum()/shift_share.portfolio_0.sum()-1
)
shift_share['category_growth'] = (
    shift_share.portfolio_1/shift_share.portfolio_0-1
)
shift_share['local_growth'] = shift_share.local_1/shift_share.local_0-1
shift_share['overall'] = shift_share.local_0*portfolio_total_growth
shift_share['category_mix'] = shift_share.local_0*(
    shift_share.category_growth-portfolio_total_growth
)
shift_share['local_shift'] = shift_share.local_0*(
    shift_share.local_growth-shift_share.category_growth
)
shift_share['observed_change'] = shift_share.local_1-shift_share.local_0
shift_share['identity_error'] = (
    shift_share[['overall','category_mix','local_shift']].sum(axis=1)
    - shift_share.observed_change
)
shift_share[['channel','overall','category_mix','local_shift',
             'observed_change','identity_error']]"""),
md(r"""### Shift–share applications and limitations

In growth marketing, overall growth measures the expansion expected if the local unit followed the portfolio; category mix measures exposure to faster/slower-growing channels; local shift measures growth beyond the category benchmark.

- Results are sensitive to benchmark, base year, geographic/channel classification, and analysis window.
- The competitive term is a residual containing measurement error, omitted composition, shocks, and regression-to-the-mean.
- Classical shift–share has no sampling model or causal identification by itself.
- Dynamic shift–share updates weights through time but can change interpretation and introduce path dependence.
- In econometrics, a **Bartik/shift–share instrument** is a different object: causal validity requires exposure shares, shocks, exclusion restrictions, and appropriate inference. The descriptive decomposition does not establish instrument validity."""),
md(r"""## 5. Structural Decomposition Analysis (SDA)

SDA decomposes changes in systems represented by matrix identities, especially input–output economics. In the Leontief quantity model,

$$x=Ax+y,$$

where $x$ is gross output, $A$ is the matrix of intermediate-input coefficients, and $y$ is final demand. If $I-A$ is invertible,

$$x=(I-A)^{-1}y=Ly,$$

where $L$ is the Leontief inverse. Between periods,

$$\Delta x=L_1y_1-L_0y_0.$$

Two exact polar decompositions are

$$\Delta x=(L_1-L_0)y_0+L_1(y_1-y_0),$$

and

$$\Delta x=(L_1-L_0)y_1+L_0(y_1-y_0).$$

Their average gives a symmetric two-factor allocation:

$$\Delta x_L=\frac12(L_1-L_0)(y_0+y_1),$$

$$\Delta x_y=\frac12(L_0+L_1)(y_1-y_0).$$

Adding them recovers $\Delta x$. With more determinants—technology, demand level, demand composition, trade, emissions intensity—the number of polar paths increases, motivating average-polar or all-permutation rules."""),
code(r"""# Two-sector SDA: technology versus final-demand effects.
A0 = np.array([[.20, .10], [.05, .25]])
A1 = np.array([[.18, .12], [.07, .22]])
y0_sda = np.array([100., 80.])
y1_sda = np.array([118., 86.])
I = np.eye(2)
L0 = np.linalg.inv(I-A0)
L1 = np.linalg.inv(I-A1)
x0_sda = L0 @ y0_sda
x1_sda = L1 @ y1_sda

technology_effect = .5*(L1-L0) @ (y0_sda+y1_sda)
demand_effect = .5*(L0+L1) @ (y1_sda-y0_sda)

pd.DataFrame({
    'technology effect': technology_effect,
    'final-demand effect': demand_effect,
    'allocated change': technology_effect+demand_effect,
    'observed change': x1_sda-x0_sda,
    'identity error': technology_effect+demand_effect-(x1_sda-x0_sda),
}, index=['sector 1','sector 2'])"""),
md(r"""### SDA applications and limitations

SDA is used for emissions footprints, supply-chain requirements, productivity, trade, and demand propagation. A business analogue is a networked operating model in which acquisition, fulfillment, support, and retention depend on one another through a coefficient matrix.

- Results inherit input–output measurement error and sector aggregation choices.
- The Leontief model assumes fixed linear coefficients within each period and requires $(I-A)^{-1}$ to exist.
- Symmetric polar averaging resolves two-factor order dependence but remains an allocation convention.
- With many factors, exact all-order averaging can become expensive.
- Technology coefficients describe requirements, not necessarily causal production responses under policy intervention.
- Structural change may include prices, capacity constraints, substitution, and behavioral adjustment absent from the fixed-coefficient identity."""),
md(r"""## Comparison of the five method families

| Method | Object | Typical output | Exact? | Main choice | Causal by itself? |
|---|---|---|---|---|---|
| Laspeyres/Paasche/Fisher | price–quantity value ratio | price and quantity indexes | Fisher passes factor reversal | weights/base/chaining | no |
| LMDI | positive multiplicative identity | additive or multiplicative factor effects | yes | factorization and zero handling | no |
| PVM | revenue identity | price, volume, mix, interaction | yes under declared rule | interaction convention | no |
| shift–share | benchmarked segment growth | overall, mix, local shift | yes | benchmark and base period | no |
| SDA | matrix structural identity | technology/demand/etc. effects | yes under polar/path rule | determinant order and aggregation | no |

## Econometric point of view

These methods primarily transform observed aggregates or estimated accounting objects. They do not estimate behavioral parameters merely because a component is called price, intensity, competitiveness, or technology.

- A price contribution is not an elasticity.
- A local shift is not a treatment effect.
- An LMDI factor is not a structural coefficient.
- An SDA technology effect is not automatically policy-invariant.

Sampling uncertainty matters when inputs are survey estimates, noisy rates, forecasts, or estimated matrices. Use bootstrap, delta-method, or simulation procedures appropriate to how those inputs were generated. Identification uncertainty cannot be repaired by a zero algebraic residual."""),
md(r"""## Limitations and robustness

- Run bilateral and chained index variants and disclose revisions, entry/exit rules, and quality adjustment.
- For LMDI, document zero handling; logarithms require positive values and negative totals can be inadmissible.
- For PVM, compare explicit-interaction, base-weighted, and symmetric allocations.
- For shift–share, repeat with credible benchmarks, base periods, and classifications.
- For SDA, compare the two polar decompositions, their average, and all-order results when feasible.
- Stress-test factorization and aggregation level across every identity method.
- None corrects endogeneity, anticipation, auction feedback, substitution, or interference without a separate econometric/causal design.

## What came next

**Diewert (1976)** formalized exact and superlative indexes. **Ang & Choi (1997)** refined the logarithmic-mean Divisia method; **Ang (2005, 2015)** consolidated LMDI theory and implementation. **Esteban-Marquillas (1972)** introduced a homothetic refinement of shift–share, while later dynamic and spatial variants addressed changing weights and dependence. In SDA, **Rose & Casler (1996)** reviewed the field and **Dietzenbacher & Los (1998)** demonstrated sensitivity to polar forms and advocated averaging paths. Modern work connects these allocation problems to Shapley and structural path methods, but exact attribution remains distinct from causal identification.""")],

'03_oaxaca_reweighting_rif.ipynb': [
md(r"""## Threefold decomposition and identification

Adding and subtracting $\bar X_B'\hat\beta_A$ yields a threefold form:

$$\bar Y_A-\bar Y_B=
(\bar X_A-\bar X_B)'\hat\beta_B
+\bar X_B'(\hat\beta_A-\hat\beta_B)
+(\bar X_A-\bar X_B)'(\hat\beta_A-\hat\beta_B).$$

These are endowment, coefficient, and interaction terms. The twofold form chooses a nondiscriminatory/reference coefficient vector $\beta^*$:

$$\Delta=(\bar X_A-\bar X_B)'\beta^*
+\bar X_A'(\hat\beta_A-\beta^*)
+\bar X_B'(\beta^*-\hat\beta_B).$$

The reference choice is not innocuous. A causal counterfactual additionally needs overlap, consistency, and a conditional exchangeability or structural assumption; OB algebra alone supplies none of these."""),
md(r"""## Growth-marketing case: cohort LTV and conversion gaps

Compare organic ($A$) and paid-social ($B$) users. Let $Y$ be 90-day LTV or activation, and $X$ include device, country, signup week, acquisition creative, and pre-acquisition intent proxies.

- The composition term asks how much of the mean gap is associated with different observed user mixes under a reference response surface.
- The structure term captures different fitted mappings from $X$ to $Y$ plus omitted variables, misspecification, selection, and potentially treatment effects.
- Do not call structure “campaign quality” or “incrementality.” Paid acquisition changes who appears in the sample and may change downstream experiences.

For binary conversion, linear probability OB is transparent but may predict outside $[0,1]$; nonlinear decompositions require an averaging/path rule."""),
code(r"""# Bootstrap uncertainty for the twofold OB components
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
ci"""),
md(r"""## Limitations and robustness

- **Index-number problem:** results change with the reference coefficient vector.
- **Omitted variables/endogeneity:** the coefficient component absorbs much more than a structural effect.
- **Support:** extrapolation occurs when covariate distributions do not overlap; trim/report overlap diagnostics.
- **Path dependence:** nonlinear and detailed decompositions depend on ordering or normalization.
- **Selection:** acquisition and outcome observation may both be selected.
- **Inference:** bootstrap the entire workflow, including reweighting/model fitting; cluster when observations share campaigns, geographies, or time shocks.
- Run specification curves over reference group, covariate set, link function, trimming threshold, and cohort window.

## What came next

**Juhn, Murphy & Pierce (1993)** decomposed distributional changes using residual ranks. **DiNardo, Fortin & Lemieux (1996)** constructed counterfactual densities via reweighting. **Machado & Mata (2005)** used quantile regression for counterfactual distributions. **Firpo, Fortin & Lemieux (2009)** introduced RIF regression for unconditional distributional statistics, and **Fortin, Lemieux & Firpo (2011)** unified the modern decomposition toolkit and clarified identification.""")],

'04_shapley_anova_ml.ipynb': [
md(r"""## 1. Shapley value: derivation, axioms, and proof of efficiency

### The allocation problem

Let $N=\{1,\ldots,p\}$ be a set of players and let $v:2^N\rightarrow\mathbb R$ be a value function. Usually $v(\varnothing)=0$; if it is not, allocate the surplus $v(N)-v(\varnothing)$. For player $j$, the marginal contribution after coalition $S$ is

$$\Delta_j(S)=v(S\cup\{j\})-v(S),\qquad j\notin S.$$

Shapley's idea is to imagine that players enter in a random order. If $S$ is the set appearing before $j$, then $j$ receives $\Delta_j(S)$. Averaging over all $p!$ orders gives

$$\phi_j(v)=\frac{1}{p!}\sum_{\pi}
\left[v(P_j^\pi\cup\{j\})-v(P_j^\pi)\right],$$

where $P_j^\pi$ is the predecessor set of $j$ in permutation $\pi$.

The same predecessor set $S$ occurs in $|S|!(p-|S|-1)!$ permutations: members of $S$ can be ordered before $j$ in $|S|!$ ways and the remaining players after $j$ in $(p-|S|-1)!$ ways. Grouping permutations by predecessor set yields the familiar coalition formula

$$\boxed{\phi_j(v)=\sum_{S\subseteq N\setminus\{j\}}
\frac{|S|!(p-|S|-1)!}{p!}\,[v(S\cup\{j\})-v(S)]}.$$

### Why the allocations sum to the total

For any fixed order $\pi=(\pi_1,\ldots,\pi_p)$, successive marginal contributions telescope:

$$\sum_{k=1}^p\left[v(\{\pi_1,\ldots,\pi_k\})
-v(\{\pi_1,\ldots,\pi_{k-1}\})\right]=v(N)-v(\varnothing).$$

Averaging an equality over all permutations preserves it, so

$$\boxed{\sum_{j=1}^p\phi_j(v)=v(N)-v(\varnothing)}.$$

This is **efficiency**, not causality.

### Characterizing axioms

Shapley (1953) showed that the allocation is uniquely characterized by:

1. **Efficiency:** all surplus is allocated.
2. **Symmetry:** players with identical marginal contributions receive the same value.
3. **Dummy/null player:** if $j$ never changes $v(S)$, then $\phi_j=0$.
4. **Additivity:** for games $v$ and $u$, $\phi(v+u)=\phi(v)+\phi(u)$.

A proof sketch uses unanimity games $u_T(S)=\mathbf 1\{T\subseteq S\}$ as a basis for all set functions. Symmetry, efficiency, and the dummy axiom force each member of $T$ to receive $1/|T|$ in $u_T$; additivity then determines the value uniquely for any linear combination of unanimity games."""),
md(r"""## Shapley application: interacting growth drivers

For

$$\text{Revenue}=\text{Traffic}\times\text{CVR}\times\text{Price},$$

the factors interact: the value of an extra conversion-rate point depends on traffic and price. A one-order replacement gives the interaction to whichever factor arrives later. Shapley averages every order, creating a symmetric exact allocation.

The business value function must still be declared. Here,

$$v(S)=R(x_S^1,x_{-S}^0)-R(x^0),$$

where drivers in $S$ use current values and the rest use baseline values. This is a descriptive hybrid game. It does not represent $E[R\mid do(X_S=x_S^1)]$."""),
code(r"""# Baseline sensitivity of the revenue Shapley allocation
def shapley_between(baseline, target):
    def v(S):
        z={k:(target[k] if k in S else baseline[k]) for k in features}
        return revenue(z)-revenue(baseline)
    out={j:0. for j in features}
    orders=list(permutations(features))
    for order in orders:
        S=set()
        for j in order:
            out[j]+=(v(S|{j})-v(S))/len(orders); S.add(j)
    return out
alt={'traffic':900.,'conversion':.045,'price':52.}
pd.DataFrame({'original baseline':shapley_between(base,current),
              'alternative baseline':shapley_between(alt,current)})"""),
md(r"""### Interpreting the baseline-sensitivity result

Both columns satisfy efficiency relative to their own baseline, yet the driver allocations differ materially. Shapley removes arbitrary **order** dependence after a game is defined; it does not remove dependence on the baseline, value function, unit of analysis, or data-generating model.

For growth reporting, disclose the baseline (previous month, budget, forecast, control group, or long-run mean) and show sensitivity to plausible alternatives."""),
md(r"""## 2. Aumann–Shapley: continuous-path attribution

Shapley treats players as discrete entrants. Aumann–Shapley extends the allocation idea to continuously divisible factors. Let $F:\mathbb R^p\rightarrow\mathbb R$ be differentiable, with baseline $x^0$ and target $x^1$. Along the straight-line path

$$x(t)=x^0+t(x^1-x^0),\qquad 0\le t\le1,$$

the Aumann–Shapley contribution of coordinate $j$ is

$$\boxed{AS_j=(x_j^1-x_j^0)\int_0^1
\frac{\partial F(x(t))}{\partial x_j}\,dt}.$$

The chain rule proves efficiency:

$$\frac{dF(x(t))}{dt}=
\sum_j\frac{\partial F(x(t))}{\partial x_j}(x_j^1-x_j^0).$$

Integrating from 0 to 1 gives

$$F(x^1)-F(x^0)=\sum_j AS_j.$$

Unlike the discrete Shapley value, Aumann–Shapley depends on a continuous path. The straight line is conventional; a different path can produce a different allocation when interactions are present.

For a two-factor product $F(x_1,x_2)=x_1x_2$, integration gives

$$AS_1=\Delta x_1\left(x_2^0+\frac{\Delta x_2}{2}\right),\qquad
AS_2=\Delta x_2\left(x_1^0+\frac{\Delta x_1}{2}\right),$$

which is exactly the equal-interaction split seen in Kitagawa for one product term."""),
code(r"""# Numerical Aumann–Shapley allocation along a straight line.
def revenue_vector(x):
    return x[0] * x[1] * x[2]

def revenue_gradient(x):
    traffic, cvr, price = x
    return np.array([cvr*price, traffic*price, traffic*cvr])

x0 = np.array([base['traffic'], base['conversion'], base['price']])
x1 = np.array([current['traffic'], current['conversion'], current['price']])
grid = np.linspace(0, 1, 20_001)
path_points = x0[None, :] + grid[:, None] * (x1-x0)[None, :]
gradients = np.array([revenue_gradient(point) for point in path_points])
average_gradient = np.trapezoid(gradients, grid, axis=0)
as_contributions = (x1-x0) * average_gradient

pd.Series({
    'traffic': as_contributions[0],
    'conversion': as_contributions[1],
    'price': as_contributions[2],
    'allocated': as_contributions.sum(),
    'observed': revenue_vector(x1)-revenue_vector(x0),
})"""),
md(r"""### Aumann–Shapley applications and limitations

Applications include allocating continuously varying costs, emissions, revenue, and risk. In growth analytics it is useful when drivers such as traffic, CVR, and AOV are naturally continuous and gradients are available.

- It requires a differentiable function or a numerical gradient.
- The path may pass through unrealistic business states.
- Results change with path choice unless the differential allocation is path-independent in the relevant sense.
- It attributes a finite change; it does not estimate a behavioral derivative or causal elasticity merely because derivatives appear in the formula.
- Numerical integration adds approximation error, which should be checked by refining the grid.

The displayed `allocated` and `observed` values coincide up to numerical error, verifying the line-integral identity."""),
md(r"""## 3. Functional ANOVA: decomposing a function and its variance

Functional ANOVA answers a different question. Instead of allocating one finite change, it decomposes a square-integrable function $f(X)$ into main effects and interactions relative to an input distribution.

For independent inputs $X=(X_1,\ldots,X_p)$,

$$f(X)=f_\varnothing+\sum_j f_j(X_j)
+\sum_{j<k}f_{jk}(X_j,X_k)+\cdots+f_{1\cdots p}(X).$$

The components are defined recursively:

$$f_\varnothing=E[f(X)],$$

$$f_j(x_j)=E[f(X)\mid X_j=x_j]-f_\varnothing,$$

$$f_{jk}(x_j,x_k)=E[f(X)\mid X_j=x_j,X_k=x_k]
-f_j(x_j)-f_k(x_k)-f_\varnothing,$$

and analogously for higher orders by subtracting all lower-order terms.

Under the product measure induced by independent inputs, each nonempty component integrates to zero over any of its arguments. Consequently distinct components are orthogonal:

$$E[f_S(X_S)f_T(X_T)]=0\quad(S\ne T).$$

Orthogonality yields the variance decomposition

$$\boxed{\operatorname{Var}(f(X))=
\sum_{\varnothing\ne S\subseteq N}V_S},\qquad
V_S=\operatorname{Var}(f_S(X_S)).$$

Sobol indices normalize these components:

$$S_j=\frac{V_j}{\operatorname{Var}(f(X))},\qquad
S_j^{\text{total}}=
\frac{\sum_{S\ni j}V_S}{\operatorname{Var}(f(X))}.$$

Functional ANOVA is therefore global and distribution-dependent. It is not the same object as a local Shapley explanation."""),
code(r"""# Exact functional ANOVA for f(x1,x2)=x1+x2+2*x1*x2,
# with independent, centered Uniform(-1,1) inputs.
n_anova = 300_000
x_anova = rng.uniform(-1, 1, size=(n_anova, 2))
x_1, x_2 = x_anova[:, 0], x_anova[:, 1]

f_empty = 0.0
f_1 = x_1
f_2 = x_2
f_12 = 2*x_1*x_2
f_total = f_empty + f_1 + f_2 + f_12

variance_components = pd.Series({
    'V1 main effect': np.var(f_1),
    'V2 main effect': np.var(f_2),
    'V12 interaction': np.var(f_12),
})
anova_summary = pd.DataFrame({
    'variance': variance_components,
    'share': variance_components / np.var(f_total),
})
anova_summary.loc['sum of components'] = anova_summary.sum()
anova_summary.loc['observed Var(f)'] = [np.var(f_total), 1.0]
anova_summary"""),
md(r"""### Reading and applying functional ANOVA

For independent centered inputs, the main effects $x_1$ and $x_2$ and interaction $2x_1x_2$ are orthogonal. The Monte Carlo sum of component variances is therefore close to the observed variance of $f$; the small discrepancy is simulation error.

Growth applications include determining whether variation in predicted conversion is dominated by acquisition channel, user intent, pricing, or interactions; screening simulator inputs; and diagnosing whether a KPI model is mostly additive.

Limitations:

- Classical uniqueness and orthogonality rely on a chosen product measure and usually independent inputs.
- With dependent features, several generalized decompositions exist and answer different questions.
- A large variance share means the feature explains model-output variability under the chosen distribution, not that intervening on it produces a large effect.
- Estimates require integration or Monte Carlo sampling; rare but important regions can be missed.
- Results change when the reference population or input distribution changes."""),
md(r"""## 4. SHAP: Shapley values for model predictions

SHAP does not introduce a new cooperative-game solution; it specifies games for explaining a fitted prediction $f(x)$. An additive explanation has the form

$$g(z')=\phi_0+\sum_{j=1}^p\phi_jz'_j,$$

where $z'_j=1$ means feature $j$ is present. **Local accuracy** requires

$$f(x)=\phi_0+\sum_j\phi_j.$$

Usually $\phi_0=E[f(X)]$, so the feature values allocate

$$f(x)-E[f(X)].$$

The crucial scientific choice is the coalition value. Two common definitions are

$$v_{\text{cond}}(S)=E[f(X)\mid X_S=x_S]$$

and

$$v_{\text{marg}}(S)=E_{X_{-S}}[f(x_S,X_{-S})].$$

The conditional game respects observed dependence but can give attribution to a feature unused by the model because it carries information about another feature. The marginal game evaluates the model after breaking dependence and may create implausible feature combinations.

A causal game is different:

$$v_{\text{causal}}(S)=E[Y\mid do(X_S=x_S)].$$

It requires a structural causal model or identified intervention distribution. Calling marginal SHAP “interventional” in software does not by itself make it causal.

### Linear-model special case

If $f(x)=\beta_0+\sum_j\beta_jx_j$ and the marginal game uses background means $\mu_j=E[X_j]$, then

$$\boxed{\phi_j=\beta_j(x_j-\mu_j)},\qquad
\phi_0=\beta_0+\sum_j\beta_j\mu_j.$$

This follows because the marginal contribution of feature $j$ is the same for every coalition, so averaging does not change it."""),
code(r"""# Verify local accuracy for marginal SHAP in a linear conversion model.
feature_names = ['intent_score', 'sessions', 'discount']
beta = np.array([0.30, 0.08, 0.45])
intercept = -1.20
background_mean = np.array([0.0, 3.0, 0.10])
customer = np.array([1.2, 5.0, 0.25])

baseline_prediction = intercept + background_mean @ beta
customer_prediction = intercept + customer @ beta
linear_shap = beta * (customer-background_mean)

pd.Series({
    'base value E[f(X)]': baseline_prediction,
    **dict(zip(feature_names, linear_shap)),
    'base + SHAP': baseline_prediction + linear_shap.sum(),
    'model prediction f(x)': customer_prediction,
})"""),
md(r"""## Growth-marketing uses of SHAP—and defensible language

Appropriate uses include:

- explaining why a churn or conversion model scored a user above its background expectation;
- detecting leakage, such as `discount_seen` measured after treatment;
- comparing model behavior across cohorts or acquisition channels;
- identifying nonlinearities and interactions that deserve model or product investigation;
- monitoring explanation drift when the background population changes.

Defensible statement:

> Relative to the declared background dataset, the fitted model assigns +0.067 probability points to recent sessions for this user.

Unsupported statement without causal identification:

> Increasing sessions will cause conversion to rise by 0.067.

Additional traps include correlated channel and geography features, aggregation of $|\phi_j|$ that hides direction, poorly calibrated predictions, and post-treatment features that are predictive but not valid intervention levers."""),
md(r"""## Comparison of the four methods

| Method | Object decomposed | Output | Exactness | Main reference choice | Causal by itself? |
|---|---|---|---|---|---|
| Shapley | discrete coalition value $v(S)$ | allocation of $v(N)-v(\varnothing)$ | exact | value function/baseline | no |
| Aumann–Shapley | differentiable finite change $F(x^1)-F(x^0)$ | path-integrated contributions | exact up to numerical integration | baseline, target, path | no |
| Functional ANOVA | function $f(X)$ under a distribution | main/interaction functions and variance shares | exact in population | input distribution/dependence structure | no |
| SHAP | fitted prediction relative to background | local feature attributions | locally exact for chosen game | model, background, missing-feature semantics | no |

The common theme is allocation. The mathematical objects, reference choices, and permissible interpretations are different."""),
md(r"""## Limitations and robustness

- **Shapley:** exponential exact computation, value-function dependence, and no canonical baseline.
- **Aumann–Shapley:** differentiability and path dependence; numerical integration error.
- **Functional ANOVA:** distribution dependence and complications under correlated inputs.
- **SHAP:** model, background, and missing-feature dependence; predictive explanation can be mistaken for intervention effect.
- For approximate allocations, report Monte Carlo or numerical error in addition to sampling uncertainty.
- Repeat results across plausible baselines/background samples, group strongly collinear variables, and examine stability across model folds and seeds.
- Validate model discrimination and calibration before explaining predictions.
- Never infer causal leverage from attribution magnitude alone.

## What came next

**Aumann & Shapley (1974)** extended discrete cooperative-game allocation to nonatomic/continuous settings. **Hoeffding (1948)** supplied an early orthogonal decomposition underlying functional ANOVA; **Sobol' (1993)** developed variance-based global sensitivity indices. **Lundberg & Lee (2017)** connected additive feature attribution to Shapley values, and **Lundberg et al. (2020)** developed efficient TreeSHAP explanations. **Aas, Jullum & Løland (2021)** addressed dependent features. **Sundararajan & Najmi (2020)** clarified that different coalition games produce different Shapley explanations, while **Heskes et al. (2020)** incorporated explicit causal structure.""")],

'05_econometrics_causality_driver_trees.ipynb': [
md(r"""## Econometric estimands behind common methods

For treatment $D$, outcome $Y$, and potential outcomes $Y(1),Y(0)$,

$$ATE=E[Y(1)-Y(0)],\qquad ATT=E[Y(1)-Y(0)\mid D=1].$$

Under conditional exchangeability, $(Y(1),Y(0))\perp D\mid X$, positivity, and consistency,

$$ATE=E_X\{E[Y\mid D=1,X]-E[Y\mid D=0,X]\}.$$

DiD instead uses parallel trends; IV identifies a local effect under relevance, independence, exclusion, and monotonicity; RD identifies a local cutoff effect under continuity/no precise manipulation. Fixed effects remove time-invariant additive confounding, not time-varying confounding. DML reduces regularization bias through orthogonal scores and cross-fitting, but still requires causal identification."""),
md(r"""## Growth-marketing designs

| Question | Preferred design | Key threat |
|---|---|---|
| incremental conversions from ads | geo/user randomized holdout | interference, noncompliance |
| lifecycle email effect | randomized send/holdout | triggered eligibility, spillovers |
| bid-policy rollout | staggered experiment or credible DiD | heterogeneous timing, anticipation |
| threshold-based offer | RD | manipulation around cutoff |
| channel incrementality with auction instrument | IV only if exclusion is credible | direct effects of instrument |
| heterogeneous treatment effects | causal forest after identification | overlap, multiple testing |

A driver tree helps define measurement identities and candidate interventions. A DAG documents confounders, mediators, colliders, and selection. An experiment or identification strategy estimates causal edges. Keep these artifacts connected but conceptually separate."""),
code(r"""# Omitted-variable-bias sensitivity in the simulation
rows=[]
for strength in [0.0,.5,1.0,1.4,2.0]:
    q=rng.normal(size=n)
    d_=(rng.random(n)<1/(1+np.exp(-strength*q))).astype(int)
    y_=10+3*q+1.5*d_+rng.normal(size=n)
    naive_=sm.OLS(y_,sm.add_constant(d_)).fit().params[1]
    adjusted_=sm.OLS(y_,sm.add_constant(np.c_[d_,q])).fit().params[1]
    rows.append((strength,naive_,adjusted_))
pd.DataFrame(rows,columns=['selection_strength','naive','adjusted']).set_index('selection_strength')"""),
md(r"""## Limitations and robustness by design

- **OLS/GLM:** functional form, omitted variables, measurement error, simultaneity; use residual diagnostics and sensitivity analysis, not causal language by default.
- **Panel fixed effects:** remaining time-varying confounding, dynamic bias, clustered dependence; cluster at the assignment/shock level.
- **DiD:** inspect pre-trends, treatment timing, anticipation, spillovers, and negative weighting under heterogeneous effects.
- **Matching/propensity scores:** cannot fix unmeasured confounding; check overlap and covariate balance, not propensity-model fit alone.
- **IV:** weak instruments and exclusion violations; report first stage and weak-IV-robust inference.
- **RD:** bandwidth/specification sensitivity, manipulation, and local external validity.
- **DML/causal forests:** overlap and identification still dominate; cross-fitting is not a cure for bad controls.
- **BSTS/synthetic control:** donor contamination, post-selection, unstable pre-period fit, and limited placebo units.

## What came next

**Rubin (1974)** formalized potential-outcome reasoning; **Rosenbaum & Rubin (1983)** developed propensity-score design. **Pearl (1995, 2009)** formalized causal graphs and intervention calculus. Modern robust estimation includes **Imbens & Lemieux (2008)** for RD practice, **Abadie, Diamond & Hainmueller (2010)** for synthetic control, **Chernozhukov et al. (2018)** for DML, and **Athey, Tibshirani & Wager (2019)** for generalized random forests. For staggered DiD with heterogeneous effects, **Callaway & Sant'Anna (2021)** and **Sun & Abraham (2021)** repair failures of naive two-way fixed-effects summaries.""")],
}

ADDITIONAL_REFS = {
'00_field_map.ipynb': r"""
- Juhn, C., Murphy, K. M., & Pierce, B. (1993). Wage inequality and the rise in returns to skill. *Journal of Political Economy*, 101, 410–442. https://doi.org/10.1086/261881
- Fortin, N., Lemieux, T., & Firpo, S. (2011). Decomposition methods in economics. In *Handbook of Labor Economics*, Vol. 4A, 1–102. https://doi.org/10.1016/S0169-7218(11)00407-2
- Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *NeurIPS 30*, 4765–4774. https://papers.nips.cc/paper/7062
- Holland, P. W. (1986). Statistics and causal inference. *JASA*, 81, 945–960. https://doi.org/10.1080/01621459.1986.10478354
- Imbens, G. W., & Rubin, D. B. (2015). *Causal Inference for Statistics, Social, and Biomedical Sciences*. Cambridge University Press. https://doi.org/10.1017/CBO9781139025751
- Hernán, M. A., & Robins, J. M. (2020). *Causal Inference: What If*. Chapman & Hall/CRC. https://www.hsph.harvard.edu/miguel-hernan/causal-inference-book/
""",
'01_rates_kitagawa_dasgupta.ipynb': r"""
- Das Gupta, P. (1993). *Standardization and Decomposition of Rates: A User's Manual*. U.S. Census Bureau.
- Shorrocks, A. F. (2013). Decomposition procedures for distributional analysis. *Journal of Economic Inequality*, 11, 99–126. https://doi.org/10.1007/s10888-011-9214-z
- Owen, G. (1977). Values of games with a priori unions. In *Mathematical Economics and Game Theory*, 76–88. Springer. https://doi.org/10.1007/978-3-642-45494-3_7
- Young, H. P. (1985). Monotonic solutions of cooperative games. *International Journal of Game Theory*, 14, 65–72. https://doi.org/10.1007/BF01769885
""",
'02_index_numbers_lmdi_pvm_sda.ipynb': r"""
- Balk, B. M. (2008). *Price and Quantity Index Numbers: Models for Measuring Aggregate Change and Difference*. Cambridge University Press. https://doi.org/10.1017/CBO9780511720758
- Ang, B. W., & Choi, K.-H. (1997). Decomposition of aggregate energy and gas emission intensities for industry: A refined Divisia index method. *The Energy Journal*, 18(3), 59–73.
- Ang, B. W. (2015). LMDI decomposition approach: A guide for implementation. *Energy Policy*, 86, 233–238. https://doi.org/10.1016/j.enpol.2015.07.007
- Esteban-Marquillas, J. M. (1972). A reinterpretation of shift-share analysis. *Regional and Urban Economics*, 2, 249–255. https://doi.org/10.1016/0034-3331(72)90033-4
- Dietzenbacher, E., & Los, B. (1998). Structural decomposition techniques: Sense and sensitivity. *Economic Systems Research*, 10, 307–324. https://doi.org/10.1080/09535319800000023
- Miller, R. E., & Blair, P. D. (2009). *Input–Output Analysis: Foundations and Extensions* (2nd ed.). Cambridge University Press. https://doi.org/10.1017/CBO9780511626982
""",
'03_oaxaca_reweighting_rif.ipynb': r"""
- Juhn, C., Murphy, K. M., & Pierce, B. (1993). Wage inequality and the rise in returns to skill. *Journal of Political Economy*, 101, 410–442. https://doi.org/10.1086/261881
- Machado, J. A. F., & Mata, J. (2005). Counterfactual decomposition of changes in wage distributions using quantile regression. *Journal of Applied Econometrics*, 20, 445–465. https://doi.org/10.1002/jae.788
- Fortin, N., Lemieux, T., & Firpo, S. (2011). Decomposition methods in economics. In *Handbook of Labor Economics*, Vol. 4A, 1–102. https://doi.org/10.1016/S0169-7218(11)00407-2
""",
'04_shapley_anova_ml.ipynb': r"""
- Sobol', I. M. (1993). Sensitivity estimates for nonlinear mathematical models. *Mathematical Modelling and Computational Experiments*, 1, 407–414.
- Owen, A. B. (2014). Sobol' indices and Shapley value. *SIAM/ASA Journal on Uncertainty Quantification*, 2, 245–251. https://doi.org/10.1137/130936233
- Lundberg, S. M. et al. (2020). From local explanations to global understanding with explainable AI for trees. *Nature Machine Intelligence*, 2, 56–67. https://doi.org/10.1038/s42256-019-0138-9
- Sundararajan, M., & Najmi, A. (2020). The many Shapley values for model explanation. *ICML 2020*, PMLR 119, 9269–9278. https://proceedings.mlr.press/v119/sundararajan20b.html
- Aas, K., Jullum, M., & Løland, A. (2021). Explaining individual predictions when features are dependent. *Artificial Intelligence*, 298, 103502. https://doi.org/10.1016/j.artint.2021.103502
- Heskes, T., Sijben, E., Bucur, I. G., & Claassen, T. (2020). Causal Shapley values: Exploiting causal knowledge to explain individual predictions of complex models. *NeurIPS 33*, 4778–4789.
""",
'05_econometrics_causality_driver_trees.ipynb': r"""
- Rosenbaum, P. R., & Rubin, D. B. (1983). The central role of the propensity score in observational studies for causal effects. *Biometrika*, 70, 41–55. https://doi.org/10.1093/biomet/70.1.41
- Imbens, G. W., & Lemieux, T. (2008). Regression discontinuity designs: A guide to practice. *Journal of Econometrics*, 142, 615–635. https://doi.org/10.1016/j.jeconom.2007.05.001
- Abadie, A., Diamond, A., & Hainmueller, J. (2010). Synthetic control methods for comparative case studies. *JASA*, 105, 493–505. https://doi.org/10.1198/jasa.2009.ap08746
- Athey, S., Tibshirani, J., & Wager, S. (2019). Generalized random forests. *Annals of Statistics*, 47, 1148–1178. https://doi.org/10.1214/18-AOS1709
- Callaway, B., & Sant'Anna, P. H. C. (2021). Difference-in-differences with multiple time periods. *Journal of Econometrics*, 225, 200–230. https://doi.org/10.1016/j.jeconom.2020.12.001
- Sun, L., & Abraham, S. (2021). Estimating dynamic treatment effects in event studies with heterogeneous treatment effects. *Journal of Econometrics*, 225, 175–199. https://doi.org/10.1016/j.jeconom.2020.09.006
""",
}

ORIENTATION = {
'00_field_map.ipynb': r"""## How to use this notebook

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

The core lesson is simple: **a decomposition is defined by its target and allocation rule, not by the chart used to display it.**""",
'01_rates_kitagawa_dasgupta.ipynb': r"""## Roadmap and notation

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

The sequence moves from Kitagawa to Stepwise and Das Gupta, then to Chevan–Sutherland's categorical refinement and Shorrocks's general axiomatic rule. It closes with two practical complications: missing segment rates and multiperiod chaining.""",
'02_index_numbers_lmdi_pvm_sda.ipynb': r"""## Roadmap and notation

### Guiding question

How can a change in a total generated by several multiplicative factors be allocated without leaving an unexplained residual?

### Prerequisites

Products, logarithms, matrix multiplication, and before/after comparisons. Differential and matrix arguments are derived explicitly; the Python examples require only basic NumPy and pandas usage.

### Symbols

- $i$: product, channel, sector, or other component.
- $Q_{it}$: activity or quantity; $I_{it}$: intensity or rate.
- $E_{it}=Q_{it}I_{it}$: component total.
- $L(a,b)$: logarithmic mean, a symmetric weight between positive $a$ and $b$.
- $p_{it},q_{it}$: price and quantity in PVM/index-number notation.

The sequence moves from scalar indexes to exact multiplicative decomposition, business revenue bridges, benchmark-relative growth, and finally matrix systems. Each method receives a separate derivation because their outputs are not interchangeable.""",
'03_oaxaca_reweighting_rif.ipynb': r"""## Roadmap and notation

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

The decomposition is first an algebraic identity built from fitted regressions. A causal reading requires additional identification assumptions that are examined later.""",
'04_shapley_anova_ml.ipynb': r"""## Roadmap and notation

### Guiding question

When several factors interact, what principled rule can allocate a total change or model prediction among them?

### Prerequisites

Sets, averages, conditional expectations, partial derivatives, and basic variance. The line-integral and orthogonality arguments are derived explicitly, so prior game theory is not required.

### Symbols

- $N=\{1,\ldots,p\}$: all players or features.
- $S\subseteq N$: coalition already present.
- $v(S)$: value produced by coalition $S$.
- $v(S\cup\{j\})-v(S)$: marginal contribution of player $j$ after $S$.
- $\phi_j$: Shapley allocation to player $j$.

The sequence is deliberate: discrete allocation (Shapley), continuous allocation (Aumann–Shapley), global function/variance decomposition (functional ANOVA), and prediction explanation (SHAP). Their shared vocabulary should not obscure that they decompose different mathematical objects.""",
'05_econometrics_causality_driver_trees.ipynb': r"""## Roadmap and notation

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

The simulated example makes confounding visible. The method table then links practical growth questions to the assumptions needed for causal identification.""",
}

RESULT_GUIDES = {
'00_field_map.ipynb': r"""## How to read the taxonomy output

Each row is defined by the **object decomposed**, not by industry. The `causal_by_itself` column is false even for Difference-in-Differences because an estimator name alone is insufficient: DiD becomes causally interpretable only when parallel trends and the rest of its design assumptions are credible. The final column identifies the choice or assumption that an analyst must make visible.""",
'01_rates_kitagawa_dasgupta.ipynb': r"""## Reading the worked example

The aggregate rate rises from 0.1495 to 0.1746, a change of **0.0251**, or **2.51 percentage points**. The method allocates 1.43 pp to mix and 1.08 pp to within-segment rates. The near-zero `error` verifies the identity numerically.

At segment level, New users contribute negatively through mix because their traffic share falls, but positively through rate because their CVR improves. Enterprise contributes positively through both channels. Positive does not mean “good intervention”; it means positive contribution to the observed aggregate change under this rule.""",
'02_index_numbers_lmdi_pvm_sda.ipynb': r"""## Reading the worked examples

The first table shows an observed total increase of 58.6. Activity contributes +73.13 while intensity contributes −14.53; these sum exactly to 58.6. Intensity therefore offsets part of the activity-driven increase. The zero residual is an algebraic property of LMDI for these positive data.

The PVM chart answers a different question. Its bars use base-period quantities and prices, while the interaction is shown separately. A large price bar is not a demand elasticity and does not estimate what quantity would have been under a different price.""",
'03_oaxaca_reweighting_rif.ipynb': r"""## Reading the worked example

The first output verifies that the mean gap equals composition plus structure under group 0's coefficients. The second output changes the reference to group 1. Both decompositions reproduce the same total gap, but the allocation changes. This is the index-number problem in observable form.

The correct conclusion is therefore conditional: “using group 0 (or group 1) as the reference response structure.” The structure component should not be renamed discrimination, campaign quality, or treatment effect without a separate identification argument.""",
'04_shapley_anova_ml.ipynb': r"""## Reading the worked example

Revenue increases by 880. Shapley allocates +440.67 to traffic, +538.67 to conversion, and −99.33 to price. `allocated = observed` verifies efficiency. The negative price contribution reflects the fall from 50 to 48; it does not estimate price elasticity.

Why is this preferable to one arbitrary replacement order? Every possible order is averaged, so interaction credit is shared symmetrically. The later baseline-sensitivity table shows that symmetry does not make the answer baseline-free.""",
'05_econometrics_causality_driver_trees.ipynb': r"""## Reading the simulation

The data-generating process sets the treatment effect to 1.5. The naive regression estimates about 4.56 because high-quality units are more likely to receive the campaign and quality also raises sales. After controlling for the simulated confounder, the coefficient is about 1.43; the remaining difference from 1.5 is ordinary sampling noise.

This is a teaching example in which the confounder is observed and correctly modeled. In real growth data, adjustment is credible only if the required confounders are measured, pre-treatment, and modeled with adequate overlap. A close adjusted estimate in this simulation is not a general endorsement of regression adjustment.""",
}

SUMMARIES = {
'00_field_map.ipynb': r"""## Takeaways and bridge to Notebook 01

1. Always name the target, contrast, and allocation rule.
2. Exact contributions need not be unique or causal.
3. A descriptive hybrid holds factors fixed by convention; a causal counterfactual compares well-defined interventions and requires identification.
4. Choose a method by the mathematical object, not by dashboard terminology.

Notebook 01 now specializes this framework to the simplest important object: an aggregate rate formed as a weighted average of segment rates.""",
'01_rates_kitagawa_dasgupta.ipynb': r"""## Takeaways and bridge to Notebook 02

1. Kitagawa separates mix from within-segment rate changes exactly.
2. The equal interaction split is symmetric but conventional.
3. Chevan–Sutherland exposes additive category effects and extends the framework to polytomous distributions.
4. Shorrocks generalizes all-orders marginal allocation to arbitrary indicators and explicit hierarchies.
5. An absent-period segment rate is not observed: the entrant total is identified, but its mix/rate split is not.
6. Chained and direct multiperiod totals agree, while their component allocations can differ.
7. Segmentation, hierarchy, time path, and sampling uncertainty can materially change the story.
8. Use causal language only with a separate design.

Notebook 02 moves from weighted rates to totals generated by multiplicative business identities.""",
'02_index_numbers_lmdi_pvm_sda.ipynb': r"""## Takeaways and bridge to Notebook 03

1. Index numbers summarize relative price and quantity change under declared weighting systems.
2. LMDI gives an exact additive or multiplicative allocation for positive factorized data.
3. PVM requires an explicit definition of genuine mix and an interaction convention.
4. Shift–share is an exact benchmark decomposition; its competitive term is descriptive.
5. SDA extends identity decomposition to interdependent matrix systems and remains path-sensitive.
6. Factorization, aggregation, baseline, and benchmark are modeling choices even when the residual is zero.
7. None of these contributions is an elasticity or incremental causal effect by itself.

Notebook 03 changes the target from an accounting total to a gap estimated from individual-level regressions.""",
'03_oaxaca_reweighting_rif.ipynb': r"""## Takeaways and bridge to Notebook 04

1. Oaxaca–Blinder splits a fitted mean gap into composition and structure.
2. Reference coefficients, support, specification, and selection matter.
3. Bootstrap uncertainty and report sensitivity across credible specifications.
4. Distributional extensions answer questions beyond the mean.

Notebook 04 replaces reference-coefficient choices with an axiomatic rule that averages marginal contributions over coalitions.""",
'04_shapley_anova_ml.ipynb': r"""## Takeaways and bridge to Notebook 05

1. Shapley uniquely allocates a discrete coalition surplus under its four axioms once $v(S)$ is fixed.
2. Aumann–Shapley allocates a continuous finite change by integrating gradients along a declared path.
3. Functional ANOVA decomposes a function and, under orthogonality, its variance relative to an input distribution.
4. SHAP applies a Shapley game to fitted predictions under declared background and missing-feature semantics.
5. Efficiency does not remove baseline, path, distribution, or model dependence.
6. None of the four methods identifies an intervention effect without additional causal structure.

Notebook 05 completes the course by defining the additional assumptions and research designs needed for causal claims.""",
'05_econometrics_causality_driver_trees.ipynb': r"""## Final takeaways

1. Contribution, association, prediction, and causation are different claims.
2. A driver tree organizes identities and hypotheses; it does not identify causal edges.
3. Every causal estimator depends on design-specific assumptions.
4. Growth decisions should combine descriptive monitoring, predictive validation, and causal experimentation without conflating their outputs.

The recommended final deliverable is three coordinated artifacts: a decomposition for monitoring, a predictive model for forecasting or targeting, and a causal design for intervention decisions.""",
}

for filename, cells in notebooks.items():
    # Rebuild every lesson in one consistent pedagogical order.
    title, theory, setup = cells[:3]
    checklist, references = cells[-2:]
    examples = cells[3:-2]
    exercise = next(c for c in examples if c.cell_type == 'markdown' and
                    ('### Exercise' in c.source or '### Capstone' in c.source))
    examples = [c for c in examples if c is not exercise]
    deep = ENRICHMENTS[filename]
    if filename == '00_field_map.ipynb':
        formal, *middle_sections, limits_next = deep
        cells[:] = [title, md(ORIENTATION[filename]), theory, formal, setup,
                    *examples, md(RESULT_GUIDES[filename]), *middle_sections,
                    limits_next, md(SUMMARIES[filename]), exercise,
                    checklist, references]
    else:
        derivation, application, *robustness_cells, limits_next = deep
        cells[:] = [title, md(ORIENTATION[filename]), theory, derivation,
                    setup, *examples, md(RESULT_GUIDES[filename]), application,
                    *robustness_cells, limits_next, md(SUMMARIES[filename]),
                    exercise, checklist, references]
    references.source += "\n" + ADDITIONAL_REFS[filename].strip()
    nb=nbf.v4.new_notebook(cells=cells, metadata={'kernelspec':{'display_name':'Python 3 (uv)','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12'}})
    nbf.write(nb, OUT/filename)
    print(f"built {filename}")
