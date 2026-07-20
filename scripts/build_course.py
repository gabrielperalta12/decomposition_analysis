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

This averaging removes order dependence and distributes all higher-order interactions. In modern terminology it is closely connected to a Shapley–Shorrocks allocation. Das Gupta's demographic framework also uses standardized rates to isolate factors while preserving the marginal structures relevant to the application.

### What “multidimensional” means for an aggregate rate

Suppose conversion is cross-classified by channel $i$ and device $j$. A useful sequential factorization is

$$R_t=\sum_i p_{it}\sum_j q_{j\mid i,t}r_{ijt},$$

where $p_{it}=P_t(\text{channel}=i)$ and $q_{j\mid i,t}=P_t(\text{device}=j\mid\text{channel}=i)$. There are now three changing blocks:

1. $P$: channel composition $p$;
2. $Q$: device-within-channel composition $q$;
3. $R$: cell-specific conversion rates $r$.

Define the hybrid standardized rate

$$H(a,b,c)=\sum_i p_{i,a}\sum_j q_{j\mid i,b}r_{ij,c},\qquad a,b,c\in\{0,1\}.$$

Every $H(a,b,c)$ is a coherent synthetic population: $p_{\cdot,a}$ sums to one and every conditional distribution $q_{\cdot\mid i,b}$ sums to one. The channel contribution, averaged over the six paths, can equivalently be written with subset weights:

$$\begin{aligned}
C_P={}&\frac13[H(1,0,0)-H(0,0,0)]\\
&+\frac16[H(1,1,0)-H(0,1,0)]\\
&+\frac16[H(1,0,1)-H(0,0,1)]\\
&+\frac13[H(1,1,1)-H(0,1,1)].
\end{aligned}$$

$C_Q$ and $C_R$ follow by rotating the factor being replaced. The weights $1/3,1/6,1/6,1/3$ are the probabilities that zero, either one, or both of the other factors precede the target factor in a random order. Consequently,

$$C_P+C_Q+C_R=H(1,1,1)-H(0,0,0).$$

### The factorization is part of the estimand

Writing $P(\text{channel})P(\text{device}\mid\text{channel})$ asks a different descriptive question from writing $P(\text{device})P(\text{channel}\mid\text{device})$. Both reproduce the observed joint distributions at the endpoints, but their intermediate standardized populations differ. Therefore the separate channel and device allocations need not agree. This is not an algebraic error: it is sensitivity to the declared standardization structure.

If the joint cell shares $w_{ijt}$ are treated as one indivisible block, the analysis has only two factors—joint composition and cell rates—and cannot separately identify a channel-composition and device-composition component. Conversely, independently combining channel and device marginals implicitly assumes away their association and can create unrealistic hybrid cells.

### Closed form for a multiplicative business identity

For $F(x)=\prod_{k=1}^K x_k$, write $x_{k1}=x_{k0}+\Delta x_k$. Expansion gives one term for every nonempty interaction set $S$:

$$F_1-F_0=\sum_{\varnothing\neq S\subseteq K}
\left(\prod_{j\in S}\Delta x_j\right)
\left(\prod_{\ell\notin S}x_{\ell0}\right).$$

All-orders averaging allocates each $|S|$-way interaction equally among its participating factors. Hence

$$C_j=\sum_{S\ni j}\frac{1}{|S|}
\left(\prod_{k\in S}\Delta x_k\right)
\left(\prod_{\ell\notin S}x_{\ell0}\right).$$

For a growth identity such as revenue $=\text{traffic}\times\text{CVR}\times\text{AOV}$, this formula makes explicit where the pairwise and three-way interactions go. For a nonlinear or cross-classified rate, enumeration through $H$ is safer than trying to invent a product shortcut."""),
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

It does not make category effects causal, choose theoretically relevant control variables, solve sparse cross-classified cells, or guarantee aggregation invariance. Chevan and Sutherland explicitly emphasize that any selected categorical variable will yield a result; scientific meaning depends on theoretically justified variable selection.

### Applied templates for Chevan–Sutherland

| Decision problem | Composition dimensions | Response | What category detail can reveal |
|---|---|---|---|
| Paid-media CVR | channel × device | purchase/no purchase | Paid Search gained mix but mobile Search lost within-cell CVR |
| Lifecycle migration | acquisition cohort × tenure | free/trial/paid/churned | a larger young-cohort share versus worse paid retention inside mature cohorts |
| Subscription churn | country × plan | churn/no churn | churn pressure concentrated in a plan within one market rather than a global plan effect |
| Lead quality | source × firm size | MQL/SQL/won/lost | a source brings more volume while its distribution shifts toward low-quality states |
| Health or labor rates | age × sex or education × region | event rate or response categories | composition shifts separated from changes within comparable cells |

The operational sequence is: define the cross-classification before looking at results; construct coherent hybrid distributions; calculate variable-level Das Gupta effects; refine each replacement marginal into its categories; verify that categories sum to their parent factor and parent factors sum to $R_1-R_0$; then repeat under defensible alternative orderings, segment definitions, and sparse-cell rules.

### A two-dimensional growth-marketing refinement

For the factorization $p_iq_{j\mid i}r_{ij}$, a replacement of $P$ has category-$i$ marginal

$$m_{P,i}=\Delta p_i\sum_j q_{j\mid i}^{*}r_{ij}^{*},$$

a replacement of $Q$ has device-category-$j$ marginal

$$m_{Q,j}=\sum_i p_i^{*}\Delta q_{j\mid i}r_{ij}^{*},$$

and a replacement of the rate block has cell marginal

$$m_{R,ij}=p_i^{*}q_{j\mid i}^{*}\Delta r_{ij}.$$

The asterisk means “use the state reached at that point in the path.” Average these marginals over every factor order. This preserves two nested adding-up identities:

$$\sum_i C_{P,i}=C_P,\qquad \sum_j C_{Q,j}=C_Q,
\qquad \sum_{ij}C_{R,ij}=C_R,$$

and $C_P+C_Q+C_R=R_1-R_0$. Crucially, channel categories are summed only to the channel parent and device categories only to the device parent. Adding a separate one-way channel decomposition to a separate one-way device decomposition would count the same aggregate change twice."""),
code(r"""# Chevan–Sutherland-style refinement for channel × device.
# q is device conditional on channel; every row therefore sums to one.
channels = ['Paid Search', 'Organic']
devices = ['Mobile', 'Desktop']
p = [np.array([.55, .45]), np.array([.48, .52])]
q = [np.array([[.70, .30], [.60, .40]]),
     np.array([[.76, .24], [.64, .36]])]
r = [np.array([[.045, .080], [.035, .065]]),
     np.array([[.041, .086], [.040, .070]])]

factor_order = ['channel mix', 'device|channel mix', 'cell rate']
category_totals = {
    'channel mix': np.zeros(len(channels)),
    'device|channel mix': np.zeros(len(devices)),
    'cell rate': np.zeros((len(channels), len(devices))),
}

def aggregate_rate(state):
    return np.sum(p[state[0]][:, None] * q[state[1]] * r[state[2]])

for order in permutations(range(3)):
    state = [0, 0, 0]
    for factor in order:
        if factor == 0:
            marginal = (p[1] - p[0])[:, None] * q[state[1]] * r[state[2]]
            category_totals['channel mix'] += marginal.sum(axis=1) / 6
        elif factor == 1:
            marginal = p[state[0]][:, None] * (q[1] - q[0]) * r[state[2]]
            category_totals['device|channel mix'] += marginal.sum(axis=0) / 6
        else:
            marginal = p[state[0]][:, None] * q[state[1]] * (r[1] - r[0])
            category_totals['cell rate'] += marginal / 6
        state[factor] = 1

rows = []
rows += [('channel mix', name, value)
         for name, value in zip(channels, category_totals['channel mix'])]
rows += [('device|channel mix', name, value)
         for name, value in zip(devices, category_totals['device|channel mix'])]
rows += [('cell rate', f'{channels[i]} × {devices[j]}',
          category_totals['cell rate'][i, j])
         for i in range(2) for j in range(2)]

refined = pd.DataFrame(rows, columns=['parent factor', 'category', 'contribution'])
parent_check = refined.groupby('parent factor')['contribution'].sum()
identity_check = pd.Series({
    'baseline rate': aggregate_rate([0, 0, 0]),
    'comparison rate': aggregate_rate([1, 1, 1]),
    'observed change': aggregate_rate([1, 1, 1]) - aggregate_rate([0, 0, 0]),
    'allocated change': refined['contribution'].sum(),
})
refined, parent_check, identity_check"""),
md(r"""### How to read the two-dimensional result

Read the first output inside each parent factor. The Paid Search and Organic rows partition only the **channel-mix** effect. Mobile and Desktop partition only the **device-within-channel** effect. The four channel × device rows partition the **within-cell-rate** effect. The parent check then sums those category rows, and the identity check verifies that the three parents reproduce the observed aggregate change.

A negative Paid Search mix contribution does not say Paid Search caused conversion to fall. It says its share changed in a direction that lowers the standardized aggregate rate under the averaged replacement rule. Likewise, a positive Mobile rate contribution can coexist with a negative Mobile composition contribution. This distinction is exactly why category-level reporting is useful.

For production use, report contributions in percentage points, include cell counts or effective sample sizes, flag cells created or removed between periods, and bootstrap the complete decomposition if sampling uncertainty matters. If a rate is undefined because a cell has zero exposure, use the explicit entrant/exit or reference-rate conventions discussed later; never silently replace the missing rate with zero."""),
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

# Notebook 06 is a standalone bridge from descriptive decomposition to
# interpretable pattern discovery. It intentionally follows Notebook 05 so the
# causal boundary is already established before adaptive subgroup searches.
subgroup_cells = [
md(r"""# 06 — Subgroup Discovery and Pattern Mining: where does the change occur?

**Graduate course: Decomposition Analysis in Python**

## Learning objectives

- Define conditional change as an estimand rather than an informal “driver.”
- Distinguish generic decision trees, CART, CHAID, subgroup discovery, and RuleFit.
- Discover candidate segments on training data and evaluate them honestly on holdout data.
- Quantify support, effect size, uncertainty, multiplicity, and stability.
- Separate descriptive change localization from heterogeneous causal effects."""),
md(r"""## Guiding question and notation

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

Thus a regression tree or rule ensemble fitted to $Z$ searches directly for heterogeneous **descriptive changes**."""),
code(r"""import numpy as np
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
plt.style.use('seaborn-v0_8-whitegrid')"""),
md(r"""## Synthetic growth-marketing case

We observe two independent monthly samples. Conversion improves mainly for high-intent Paid Search visitors on mobile, deteriorates for low-intent Social traffic, and changes slightly with tenure. Because the data-generating process is known, we can judge whether each method recovers the planted patterns.

In real work, discovery never reveals “truth” this cleanly. The simulation is a unit test for reasoning, not evidence that a method controls business confounding."""),
code(r"""n = 10_000
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
overall"""),
md(r"""## Honest discovery/evaluation split

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

This interval is valid for a prespecified subgroup under ordinary sampling assumptions. Selection makes discovery-sample intervals optimistic; holdout evaluation limits that problem."""),
code(r"""train, holdout = train_test_split(
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

train.shape, holdout.shape"""),
md(r"""## 1. Decision trees: the broad family

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

Large leaves improve precision but can hide narrow patterns; small leaves increase discovery power at the cost of variance and false findings."""),
code(r"""features = ['channel', 'device', 'market', 'tenure_months', 'sessions_30d']
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
cart_holdout"""),
code(r"""# Verify the CART gain identity for a candidate Paid Search split.
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
})"""),
code(r"""# Overfitting diagnostic: compare a deep tree with the regularized tree.
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
})"""),
md(r"""### Cost-complexity pruning in practice

CART's weakest-link pruning first grows a large tree and produces a nested sequence of subtrees indexed by $\alpha$. Cross-validation should choose among those subtrees. The smallest validation error is not the only defensible choice: the **one-standard-error rule** selects the simplest tree whose validation loss is statistically indistinguishable from the minimum.

The next diagnostic evaluates a compact grid of pruning values on the untouched holdout pseudo-outcome only for teaching. In a real honest workflow, tune $\alpha$ inside the discovery sample and reserve holdout solely for final evaluation. Otherwise the holdout quietly becomes another training set."""),
code(r"""holdout_p = holdout.period.mean()
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
pruning_diagnostic.sort_values('holdout proxy MSE').head(8)"""),
md(r"""## 2. CHAID: categorical, multiway, significance-driven splitting

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

Below is a **CHAID-style first-node diagnostic**, not a full CHAID implementation. For each categorical feature it compares a logistic model containing period and category main effects with one also containing their interaction. The likelihood-ratio test asks whether descriptive period change varies across categories."""),
code(r"""def chaid_style_screen(data, variables):
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
chaid_screen"""),
code(r"""# Make the winning candidate substantively interpretable.
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
chaid_category_changes.sort_values('change', ascending=False)"""),
md(r"""### Reading the CHAID-style output

The interaction screen answers whether change differs somewhere across the categories of a variable. It does not say every category differs, nor does its $p$-value measure effect size. Read it jointly with category rates, denominators, absolute changes, and a commercially relevant threshold.

The demonstration stops after the first node and does not implement CHAID's iterative category merging, multiway recursion, ordinal restrictions, or full multiplicity bookkeeping. Calling it “CHAID-style interaction screening” is deliberate. A production CHAID analysis should use a tested implementation and disclose its merge/split thresholds."""),
md(r"""## 3. Subgroup Discovery: explicit search for interesting rules

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

The code below uses a compact beam search over one- and two-condition conjunctions. It is deliberately transparent: production systems should also canonicalize duplicate rules, control the search budget, correct multiplicity, and evaluate frozen rules on fresh data."""),
code(r"""conditions = {
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
discovered"""),
code(r"""# Calibrate the maximum search score under a shuffled-period null.
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
permutation_diagnostic"""),
code(r"""# Freeze the top rules and estimate them on holdout observations.
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
honest_rules"""),
code(r"""# Quantify discovery optimism and redundancy among the selected rules.
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
honest_rules[['rule', 'change_train', 'change_holdout', 'absolute_shrinkage']], jaccard"""),
md(r"""### Reading the Subgroup Discovery diagnostics

`absolute_shrinkage > 0` indicates that the discovered magnitude became smaller on holdout—a direct view of winner's curse. Negative values can occur by chance and are not proof of anti-overfitting. The Jaccard matrix reveals when several impressive rows describe almost the same people.

For confirmatory use, freeze the rule list before holdout, report all frozen rules rather than only survivors, and use simultaneous or multiplicity-adjusted inference when making a family of claims. For exploratory use, label the table as hypothesis generation."""),
md(r"""## 4. RuleFit: a sparse model built from tree rules

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

The implementation below is pedagogical and RuleFit-inspired: gradient-boosted leaf rules plus scaled encoded linear terms followed by `LassoCV`. A production implementation should use repeated validation, explicit winsorization, rule deduplication, stability selection, and a supported RuleFit library."""),
code(r"""gb = GradientBoostingRegressor(
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
}), top_rules"""),
code(r"""# Verify that support-adjusted importance is the rule contribution's SD.
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
})"""),
md(r"""### Reading the RuleFit output

The table ranks selected leaf rules by support-adjusted importance, not merely by coefficient size. A positive coefficient means the rule raises the fitted change target conditional on all other selected terms; it is not the rule's standalone period change. Inspect the raw baseline/comparison rates for any rule before giving it business meaning.

This demonstration differs from canonical RuleFit in four ways: it extracts terminal leaves only, uses a fixed shallow boosting depth, scales encoded linear columns without explicit winsorization, and does not deduplicate logically equivalent rules. Those simplifications keep the mechanism visible but should be removed or documented in production research.

### Why RuleFit and Subgroup Discovery can disagree

Subgroup Discovery scores each rule largely on its own. RuleFit estimates coefficients jointly, so one rule can absorb variation that another overlapping rule would explain marginally. CART, meanwhile, forces a single exhaustive partition. Disagreement is therefore expected and useful: it exposes dependence on the search space and estimand rather than revealing which algorithm found the one “true” segment."""),
md(r"""## Method comparison

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
- Wager, S., & Athey, S. (2018). Estimation and inference of heterogeneous treatment effects using random forests. *JASA*, 113, 1228–1242. https://doi.org/10.1080/01621459.2017.1319839"""),
]

subgroup_nb = nbf.v4.new_notebook(
    cells=subgroup_cells,
    metadata={
        'kernelspec': {
            'display_name': 'Python 3 (uv)', 'language': 'python', 'name': 'python3'
        },
        'language_info': {'name': 'python', 'version': '3.12'},
    },
)
nbf.write(subgroup_nb, OUT / '06_subgroup_discovery_pattern_mining.ipynb')
print('built 06_subgroup_discovery_pattern_mining.ipynb')

# Notebook 07 operationalizes the full workflow developed across the course.
growth_cells = [
md(r"""# 07 — Complete Growth workflow: from KPI change to an experiment decision

**Graduate course: Decomposition Analysis in Python**

## Learning objectives

- Translate a KPI alert into an auditable analytical contract.
- Separate known-segment decomposition from adaptive subgroup discovery.
- Build an honest HCD-style tree whose leaf contributions conserve the KPI change.
- Convert a replicated descriptive pattern into a testable mechanism.
- Estimate experimental lift and connect it to an economic launch decision.
- Produce a final Growth readout without confusing contribution, prediction, and causation."""),
md(r"""## The workflow

```text
Metric contract → data QA → aggregate change
      ↓
Known segmentation → exact Kitagawa mix/rate decomposition
      ↓
Discovery sample → regularized CART/HCD candidate hierarchy
      ↓
Holdout sample → exact leaf ledger + uncertainty + stability
      ↓
Mechanism hypothesis → randomized experiment
      ↓
Causal lift → incremental conversions → economic decision
```

The stages answer different questions:

| Stage | Question | Permitted claim |
|---|---|---|
| Monitoring | Did the KPI move? | observed change |
| Kitagawa | How is the change allocated across known cells? | descriptive contribution |
| HCD discovery | Where is the change concentrated? | replicated descriptive localization |
| Experiment | Does the intervention change the KPI? | causal effect under the design |
| Economics | Is rollout valuable at relevant scale? | decision conditional on costs and transport |

No arrow automatically upgrades the previous result into a causal statement."""),
code(r"""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from pathlib import Path
from itertools import permutations
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor, export_text

PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from scripts.rate_decomposition import kitagawa_two_period

pd.options.display.float_format = '{:,.4f}'.format
rng = np.random.default_rng(2027)
plt.style.use('seaborn-v0_8-whitegrid')"""),
md(r"""## 1. Write the metric and decision contract first

**Scenario.** Monthly website conversion rate fell. The team wants to know where the movement occurred and whether a new mobile landing experience should be tested.

The contract is:

- **population:** eligible web sessions, excluding bots and internal traffic;
- **unit:** session;
- **numerator:** sessions with a purchase inside the attribution window;
- **denominator:** eligible sessions;
- **contrast:** comparison month minus baseline month;
- **known reporting dimensions:** channel and device;
- **discovery dimensions:** channel, device, market, intent, and tenure;
- **decision:** launch a landing-page intervention only if replicated incremental value exceeds cost;
- **causal status:** the monthly contrast is descriptive; only the later randomized treatment identifies lift.

Changing any of these definitions changes the estimand."""),
md(r"""## 2. Simulate two monthly populations

The simulation contains both composition drift and within-segment performance changes. Paid Search receives more traffic, while high-intent Paid Search mobile sessions deteriorate. Social desktop improves slightly. This structure lets us test whether the workflow distinguishes aggregate allocation from local pattern discovery."""),
code(r"""def simulate_month(period, n):
    channel_p = ([.32, .43, .25] if period == 0 else [.40, .36, .24])
    data = pd.DataFrame({
        'period': period,
        'channel': rng.choice(['Paid Search', 'Organic', 'Social'], n,
                              p=channel_p),
        'device': rng.choice(['Mobile', 'Desktop'], n, p=[.70, .30]),
        'market': rng.choice(['Lima', 'Mexico City', 'Bogota'], n,
                             p=[.35, .38, .27]),
        'high_intent': rng.binomial(1, .38, n),
        'tenure_months': rng.gamma(2.1, 5.5, n).clip(0, 36),
    })
    logit = (
        -2.75
        + .28 * data.high_intent
        + .30 * data.device.eq('Desktop')
        + .16 * data.channel.eq('Paid Search')
        + .012 * data.tenure_months
    )
    period_shift = (
        -.05
        - .72 * (data.channel.eq('Paid Search')
                 & data.device.eq('Mobile') & data.high_intent.eq(1))
        + .32 * (data.channel.eq('Social') & data.device.eq('Desktop'))
        + .16 * (data.market.eq('Lima') & data.tenure_months.ge(12))
    )
    probability = 1 / (1 + np.exp(-(logit + period * period_shift)))
    data['converted'] = rng.binomial(1, probability)
    return data

events = pd.concat([
    simulate_month(0, 14_000),
    simulate_month(1, 14_000),
], ignore_index=True)

metric = events.groupby('period').converted.agg(['sum', 'count', 'mean'])
metric.loc['change', 'mean'] = metric.loc[1, 'mean'] - metric.loc[0, 'mean']
metric"""),
md(r"""## 3. Data-quality gate

Before explaining a movement, verify that it is not a broken metric. At minimum check uniqueness of the analysis unit, missing dimensions, invalid outcome values, denominator volume, period coverage, and category drift. A production workflow should also compare event-schema versions, bot filters, consent logic, attribution windows, and late-arriving events."""),
code(r"""qa = pd.Series({
    'rows': len(events),
    'duplicate index': events.index.duplicated().sum(),
    'missing key dimensions': events[
        ['period', 'channel', 'device', 'market', 'converted']
    ].isna().any(axis=1).sum(),
    'invalid outcomes': (~events.converted.isin([0, 1])).sum(),
    'baseline denominator': events.period.eq(0).sum(),
    'comparison denominator': events.period.eq(1).sum(),
})
channel_volume = pd.crosstab(events.channel, events.period, normalize='columns')
qa, channel_volume"""),
md(r"""## 4. Known-segment decomposition with Kitagawa

First use the business hierarchy already understood by stakeholders. For joint cell $g=\text{channel}\times\text{device}$,

$$R_t=\sum_g w_{gt}r_{gt}.$$

Kitagawa allocates the observed change exactly:

$$\Delta R=\sum_g
(w_{g1}-w_{g0})\frac{r_{g1}+r_{g0}}{2}
+\sum_g(r_{g1}-r_{g0})\frac{w_{g1}+w_{g0}}{2}.$$

This stage answers whether the aggregate movement is mainly associated with **joint channel × device composition** or performance inside those joint cells. It does not separately attribute composition to channel and device, because the joint share vector $w_{gt}$ is treated as one changing block."""),
code(r"""cell = (
    events.groupby(['period', 'channel', 'device'])
    .converted.agg(['mean', 'size']).reset_index()
)
cell['weight'] = cell['size'] / cell.groupby('period')['size'].transform('sum')
wide = cell.pivot(index=['channel', 'device'], columns='period',
                  values=['weight', 'mean']).reset_index()
wide.columns = ['channel', 'device', 'w0', 'w1', 'r0', 'r1']
wide['segment'] = wide.channel + ' × ' + wide.device

kitagawa = kitagawa_two_period(
    wide[['segment', 'w0', 'w1', 'r0', 'r1']]
)
known_ledger = kitagawa.detail.set_index('segment')[
    ['w0', 'w1', 'r0', 'r1', 'mix', 'rate', 'total_contribution']
].sort_values('total_contribution')
kitagawa.summary, known_ledger"""),
code(r"""known_ledger[['mix', 'rate']].plot.barh(figsize=(9, 4.8))
plt.axvline(0, color='black', linewidth=.8)
plt.xlabel('Contribution to aggregate CVR change')
plt.title('Known channel × device decomposition')
plt.tight_layout()
plt.show()"""),
md(r"""## 4.1 Should Growth use Kitagawa, Das Gupta, or Shapley?

The answer depends on the question—not on which method sounds more advanced.

### Kitagawa is preferable when

- the KPI is a weighted rate;
- segments form one declared partition;
- the desired distinction is **joint composition versus within-cell rate**;
- stakeholders can act on the joint cells;
- a transparent two-component ledger is more valuable than finer attribution.

For `channel × device`, Kitagawa treats the six joint shares as one composition vector. It is exact and robust, but cannot say how much of composition is specifically “channel” versus “device.”

### Das Gupta is preferable when

- there are several scientifically distinct changing blocks;
- the analyst can define coherent standardized populations;
- the question requires a separate allocation to channel, device-within-channel, and cell rate;
- interaction allocation and factorization sensitivity are reported.

Use the sequential factorization

$$R_t=\sum_i p_{it}\sum_j q_{j\mid i,t}r_{ijt},$$

where $p_{it}=P_t(\text{channel}=i)$ and $q_{j\mid i,t}=P_t(\text{device}=j\mid\text{channel}=i)$. Define

$$H(a,b,c)=\sum_i p_{i,a}\sum_j q_{j\mid i,b}r_{ij,c}.$$

Replacing the three blocks along all $3!=6$ orders and averaging produces exact symmetric contributions.

### Where Shapley enters

For the same hybrid value function $H$, all-orders Das Gupta replacement and a Shapley allocation are mathematically the same averaging principle. “Use Shapley” is incomplete until the coalition value—here, the rule for constructing each hybrid distribution—is defined. Shapley resolves order dependence conditional on that game; it does not resolve ambiguity about factorization, causal interpretation, or impossible hybrid populations.

### Recommendation for this workflow

Report both levels:

1. **primary operational ledger:** Kitagawa on joint channel × device cells;
2. **secondary diagnostic:** Das Gupta/Shapley over channel composition, device conditional composition, and cell rates;
3. **sensitivity analysis:** reverse the factorization and disclose how component labels move;
4. never sum the Kitagawa and Das Gupta tables—they are alternative views of the same total change."""),
code(r"""# Multidimensional Das Gupta/Shapley allocation for a sequential factorization.
def sequential_components(data, first_col, second_col):
    first_levels = sorted(data[first_col].unique())
    second_levels = sorted(data[second_col].unique())
    first_share, conditional_share, cell_rate = [], [], []
    for period in [0, 1]:
        period_data = data.loc[data.period.eq(period)]
        counts = pd.crosstab(period_data[first_col], period_data[second_col]).reindex(
            index=first_levels, columns=second_levels, fill_value=0
        )
        outcomes = period_data.pivot_table(
            index=first_col, columns=second_col, values='converted', aggfunc='mean'
        ).reindex(index=first_levels, columns=second_levels)
        if outcomes.isna().any().any():
            raise ValueError('Every cross-classified cell needs support in both periods')
        first_share.append(counts.sum(axis=1).to_numpy()/counts.to_numpy().sum())
        conditional_share.append(counts.div(counts.sum(axis=1), axis=0).to_numpy())
        cell_rate.append(outcomes.to_numpy())
    return first_levels, second_levels, first_share, conditional_share, cell_rate

def all_orders_standardization(first_share, conditional_share, cell_rate, labels):
    def H(state):
        return float(np.sum(
            first_share[state[0]][:, None]
            * conditional_share[state[1]]
            * cell_rate[state[2]]
        ))
    allocation = np.zeros(3)
    path_rows = []
    for order in permutations(range(3)):
        state = [0, 0, 0]
        row = {'order': ' → '.join(labels[index] for index in order)}
        for factor in order:
            before = H(state)
            state[factor] = 1
            marginal = H(state)-before
            allocation[factor] += marginal/6
            row[labels[factor]] = marginal
        path_rows.append(row)
    summary = pd.Series(allocation, index=labels)
    summary['allocated'] = allocation.sum()
    summary['observed'] = H([1, 1, 1])-H([0, 0, 0])
    summary['error'] = summary.allocated-summary.observed
    return summary, pd.DataFrame(path_rows)

_, _, p_cd, q_cd, r_cd = sequential_components(events, 'channel', 'device')
channel_first, channel_paths = all_orders_standardization(
    p_cd, q_cd, r_cd,
    ['channel composition', 'device | channel composition', 'cell rate'],
)

_, _, p_dc, q_dc, r_dc = sequential_components(events, 'device', 'channel')
device_first, device_paths = all_orders_standardization(
    p_dc, q_dc, r_dc,
    ['device composition', 'channel | device composition', 'cell rate'],
)

multidimensional_comparison = pd.concat(
    [channel_first.rename('channel first'), device_first.rename('device first')],
    axis=1,
)
multidimensional_comparison, channel_paths"""),
md(r"""### Reading the multidimensional comparison

Both columns reproduce the same observed CVR change and should have near-zero error. Their intermediate composition labels differ because they answer different standardization questions:

- `channel first` preserves channel marginals and changes device conditional on channel;
- `device first` preserves device marginals and changes channel conditional on device.

If the channel contribution changes materially across these representations, the data do not support a factorization-invariant statement such as “channel explains exactly X.” The defensible statement is conditional: “under the channel-first standardization, X is allocated to channel composition.”

For routine Growth reporting, joint-cell Kitagawa is usually the safest primary table. Use multidimensional Das Gupta/Shapley when the separate factor labels matter enough to justify the additional modeling choices."""),
md(r"""## 5. HCD discovery: search without sacrificing conservation

The known table may hide an interaction involving intent, market, or tenure. We now split the data **before** searching. If period membership depends on observed composition, the constant-$p$ transformed outcome is inappropriate. Let

$$
e(x)=P(T=1\mid X=x),
\qquad
m_t(x)=E[Y\mid T=t,X=x].
$$

Use the cross-fitted augmented inverse-probability pseudo-outcome

$$
\phi_i=\widehat m_1(X_i)-\widehat m_0(X_i)
+\frac{T_i\{Y_i-\widehat m_1(X_i)\}}{\widehat e(X_i)}
-\frac{(1-T_i)\{Y_i-\widehat m_0(X_i)\}}{1-\widehat e(X_i)}.
$$

Under correct nuisance estimation and overlap, $E[\phi\mid X=x]$ targets the conditional descriptive period contrast. Cross-fitting prevents each observation's outcome from training its own nuisance prediction. Propensity clipping controls variance but changes the practical target in regions without overlap; report the clipping rate.

A regularized CART searches for heterogeneity in $\phi$. The tree proposes a partition; it does not define the final contributions. This orthogonalized target is usually less noisy and more robust to composition drift than the simple constant-propensity transformation.

On holdout data, each frozen leaf receives exact KPI-mass contribution

$$
c(A)=w_1(A)r_1(A)-w_0(A)r_0(A).
$$

Because leaves partition the population, their contributions sum to the holdout aggregate change."""),
code(r"""discovery, evaluation = train_test_split(
    events, test_size=.45, random_state=42, stratify=events.period
)
discovery = discovery.copy()
evaluation = evaluation.copy()

features = ['channel', 'device', 'market', 'high_intent', 'tenure_months']
categorical = ['channel', 'device', 'market']
transformer = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical),
    ('num', 'passthrough', ['high_intent', 'tenure_months']),
])
X_discovery = transformer.fit_transform(discovery[features])
X_evaluation = transformer.transform(evaluation[features])

# Cross-fitted AIPW pseudo-outcome for descriptive period heterogeneity.
t = discovery.period.to_numpy()
y = discovery.converted.to_numpy()
e_hat = np.zeros(len(discovery))
m0_hat = np.zeros(len(discovery))
m1_hat = np.zeros(len(discovery))
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for fit_index, predict_index in folds.split(X_discovery, t):
    propensity = LogisticRegression(max_iter=2_000).fit(
        X_discovery[fit_index], t[fit_index]
    )
    e_hat[predict_index] = propensity.predict_proba(
        X_discovery[predict_index]
    )[:, 1]
    for period, destination in [(0, m0_hat), (1, m1_hat)]:
        period_fit = fit_index[t[fit_index] == period]
        outcome_model = GradientBoostingClassifier(
            n_estimators=80, max_depth=2, min_samples_leaf=150,
            learning_rate=.04, random_state=42+period,
        ).fit(X_discovery[period_fit], y[period_fit])
        destination[predict_index] = outcome_model.predict_proba(
            X_discovery[predict_index]
        )[:, 1]

e_raw = e_hat.copy()
e_hat = np.clip(e_hat, .05, .95)
discovery['change_target'] = (
    m1_hat-m0_hat
    + t*(y-m1_hat)/e_hat
    - (1-t)*(y-m0_hat)/(1-e_hat)
)
overlap_diagnostic = pd.Series({
    'minimum estimated propensity': e_raw.min(),
    'maximum estimated propensity': e_raw.max(),
    'share clipped': np.mean((e_raw < .05) | (e_raw > .95)),
    'pseudo-outcome mean': discovery.change_target.mean(),
})
display(overlap_diagnostic.rename('overlap diagnostic'))

hcd_tree = DecisionTreeRegressor(
    max_depth=4, min_samples_leaf=500, ccp_alpha=0.00001,
    random_state=42,
).fit(X_discovery, discovery.change_target)

tree_feature_names = list(transformer.get_feature_names_out())

def describe_leaf(tree, target_leaf, names):
    def walk(node, path):
        if node == target_leaf:
            return path
        if tree.children_left[node] == tree.children_right[node]:
            return None
        feature = names[tree.feature[node]]
        threshold = tree.threshold[node]
        left = walk(
            tree.children_left[node],
            path + [f'{feature} <= {threshold:.3g}'],
        )
        if left is not None:
            return left
        return walk(
            tree.children_right[node],
            path + [f'{feature} > {threshold:.3g}'],
        )
    return ' AND '.join(walk(0, []) or ['root'])

print(export_text(
    hcd_tree, feature_names=tree_feature_names,
    decimals=3,
))
evaluation['hcd_leaf'] = hcd_tree.apply(X_evaluation)"""),
md(r"""## 6. Build the exact holdout leaf ledger

The ledger keeps raw accounting separate from search. For each leaf it reports both period denominators, shares, rates, total contribution, Kitagawa mix/rate refinement, and uncertainty in its within-leaf rate change.

Two identities must pass:

$$
\sum_A c(A)=R_1-R_0,
\qquad
\sum_A[c_w(A)+c_r(A)]=R_1-R_0.
$$

The uncertainty interval for $r_1(A)-r_0(A)$ is not an interval for the total contribution and is not selection-adjusted. It is a transparent holdout diagnostic after the hierarchy has been frozen."""),
code(r"""def exact_leaf_ledger(data, leaf_col):
    counts = data.groupby(['period', leaf_col]).converted.agg(['mean', 'size'])
    rows = []
    totals = data.groupby('period').size()
    for leaf in sorted(data[leaf_col].unique()):
        r0, n0 = counts.loc[(0, leaf), ['mean', 'size']]
        r1, n1 = counts.loc[(1, leaf), ['mean', 'size']]
        w0, w1 = n0/totals.loc[0], n1/totals.loc[1]
        mix = (w1-w0)*(r1+r0)/2
        rate = (r1-r0)*(w1+w0)/2
        se_rate_change = np.sqrt(r1*(1-r1)/n1 + r0*(1-r0)/n0)
        rows.append({
            'leaf': leaf, 'n0': n0, 'n1': n1,
            'w0': w0, 'w1': w1, 'r0': r0, 'r1': r1,
            'rate_change': r1-r0,
            'rate_ci_low': r1-r0-1.96*se_rate_change,
            'rate_ci_high': r1-r0+1.96*se_rate_change,
            'mix': mix, 'rate': rate,
            'total_contribution': w1*r1-w0*r0,
        })
    return pd.DataFrame(rows).set_index('leaf')

hcd_ledger = exact_leaf_ledger(evaluation, 'hcd_leaf')
eval_rates = evaluation.groupby('period').converted.mean()
eval_change = eval_rates.loc[1] - eval_rates.loc[0]
checks = pd.Series({
    'observed evaluation change': eval_change,
    'sum total contributions': hcd_ledger.total_contribution.sum(),
    'sum mix + rate': hcd_ledger[['mix', 'rate']].to_numpy().sum(),
    'conservation error': hcd_ledger.total_contribution.sum()-eval_change,
})
hcd_ledger.sort_values('total_contribution'), checks"""),
code(r"""hcd_ledger.sort_values('total_contribution')[['mix', 'rate']].plot.barh(
    figsize=(9, 4.8)
)
plt.axvline(0, color='black', linewidth=.8)
plt.xlabel('Contribution to evaluation-sample CVR change')
plt.title('HCD leaf ledger: exact contribution frontier')
plt.tight_layout()
plt.show()"""),
md(r"""### Fixed-hierarchy bootstrap

Exactness is algebraic; the estimated contributions still have sampling uncertainty. Because the hierarchy was frozen before looking at `evaluation`, bootstrap observations separately within each period and recompute the complete ledger. Every bootstrap draw should conserve its own resampled aggregate change.

This bootstrap conditions on the discovered tree. It captures uncertainty in shares and rates on evaluation data, but not variability from relearning the hierarchy. To assess full pipeline stability, repeat discovery inside every resample or across temporal folds and compare both leaf membership and contributions."""),
code(r"""bootstrap_rng = np.random.default_rng(99)
bootstrap_rows = []
leaf_order = hcd_ledger.index.tolist()
for draw in range(300):
    sampled_parts = []
    for period in [0, 1]:
        period_data = evaluation.loc[evaluation.period.eq(period)]
        sampled_parts.append(period_data.sample(
            n=len(period_data), replace=True,
            random_state=int(bootstrap_rng.integers(0, 2**31-1)),
        ))
    sampled = pd.concat(sampled_parts, ignore_index=True)
    sampled_ledger = exact_leaf_ledger(sampled, 'hcd_leaf').reindex(leaf_order)
    sampled_rates = sampled.groupby('period').converted.mean()
    sampled_change = sampled_rates.loc[1]-sampled_rates.loc[0]
    conservation_error = (
        sampled_ledger.total_contribution.sum()-sampled_change
    )
    for leaf, contribution in sampled_ledger.total_contribution.items():
        bootstrap_rows.append({
            'draw': draw, 'leaf': leaf,
            'total_contribution': contribution,
            'conservation_error': conservation_error,
        })

bootstrap_results = pd.DataFrame(bootstrap_rows)
bootstrap_intervals = bootstrap_results.groupby('leaf').total_contribution.quantile(
    [.025, .5, .975]
).unstack().rename(columns={.025: 'q025', .5: 'median', .975: 'q975'})
bootstrap_diagnostic = pd.Series({
    'draws': bootstrap_results.draw.nunique(),
    'maximum absolute conservation error': (
        bootstrap_results.groupby('draw').conservation_error.first().abs().max()
    ),
})
bootstrap_intervals.join(hcd_ledger[['total_contribution']]), bootstrap_diagnostic"""),
md(r"""## 7. Convert a pattern into a mechanism hypothesis

Select a leaf because it is material, replicated, sufficiently large, and actionable—not merely because it ranks first. The tree and ledger support a statement such as:

> “The largest negative holdout contribution is concentrated in the population defined by this frozen tree path; most of its contribution is associated with within-leaf rate change rather than share movement.”

Next inspect funnel diagnostics, instrumentation, page speed, creative, inventory, and policy history. Suppose the product review suggests that the mobile landing page mismatches high-intent Paid Search queries. That is a mechanism hypothesis, still not a causal conclusion.

The intervention is now explicit: a faster, query-matched landing experience. The discovered leaf determines eligibility for a future experiment; experiment assignment remains randomized inside that population."""),
code(r"""# Choose the most negative replicated leaf and quantify its footprint.
candidate_leaf = hcd_ledger.total_contribution.idxmin()
candidate = hcd_ledger.loc[candidate_leaf]
candidate_rule = describe_leaf(
    hcd_tree.tree_, candidate_leaf, tree_feature_names
)
candidate_summary = pd.Series({
    'candidate leaf': candidate_leaf,
    'frozen rule': candidate_rule,
    'evaluation baseline sessions': candidate.n0,
    'evaluation comparison sessions': candidate.n1,
    'rate change': candidate.rate_change,
    'total KPI contribution': candidate.total_contribution,
    'share of signed aggregate change': (
        candidate.total_contribution / eval_change if eval_change != 0 else np.nan
    ),
})
print('Frozen rule:', candidate_rule)
print(candidate_summary.drop('frozen rule').to_string())"""),
md(r"""## 8. Randomized causal validation

Generate a future eligible population using the same feature schema, route it through the frozen tree, and retain the candidate leaf. Randomly assign the new landing experience:

\[
D_i\sim\operatorname{Bernoulli}(0.5).
\]

Under random assignment, consistency, and no interference, the difference in conversion rates estimates the sample average treatment effect for eligible sessions:

\[
\widehat\tau=ar Y_{D=1}-\bar Y_{D=0}.
\]

This effect is causal for the experimental population. Transporting it to future traffic additionally assumes that the eligible population and implementation remain comparable."""),
code(r"""future = simulate_month(1, 30_000).drop(columns='converted')
future_X = transformer.transform(future[features])
future['hcd_leaf'] = hcd_tree.apply(future_X)
eligible = future.loc[future.hcd_leaf.eq(candidate_leaf)].copy()
eligible['treatment'] = rng.binomial(1, .5, len(eligible))

# A transparent experimental DGP: the new experience adds 2.5 pp.
baseline_probability = np.clip(
    candidate.r1 + .012*(eligible.high_intent-eligible.high_intent.mean()),
    .005, .95,
)
true_experiment_lift = .025
eligible['converted'] = rng.binomial(
    1, np.clip(baseline_probability
               + true_experiment_lift*eligible.treatment, 0, 1)
)

experiment = eligible.groupby('treatment').converted.agg(['mean', 'size'])
lift = experiment.loc[1, 'mean'] - experiment.loc[0, 'mean']
lift_se = np.sqrt(
    experiment.loc[1, 'mean']*(1-experiment.loc[1, 'mean'])/experiment.loc[1, 'size']
    + experiment.loc[0, 'mean']*(1-experiment.loc[0, 'mean'])/experiment.loc[0, 'size']
)
experiment_readout = pd.Series({
    'eligible sessions': len(eligible),
    'control CVR': experiment.loc[0, 'mean'],
    'treatment CVR': experiment.loc[1, 'mean'],
    'estimated lift': lift,
    '95% CI low': lift-1.96*lift_se,
    '95% CI high': lift+1.96*lift_se,
    'true simulated lift': true_experiment_lift,
})
experiment, experiment_readout"""),
md(r"""## 9. Translate lift into an economic decision

Statistical significance is not the launch criterion. Let $N_E$ be expected eligible sessions, $v$ contribution margin per incremental conversion, $C_F$ fixed implementation cost, and $c$ variable cost per treated session. Expected incremental value is

\[
V=N_E\widehat\tau v-C_F-N_Ec.
\]

Use the confidence interval to construct downside and upside scenarios. This calculation assumes the experimental lift transports to the planned rollout volume and that no general-equilibrium, novelty, or interference effects appear."""),
code(r"""monthly_eligible_sessions = 80_000
margin_per_conversion = 42.0
fixed_monthly_cost = 35_000.0
variable_cost_per_session = .08

def net_value(effect):
    return (
        monthly_eligible_sessions * effect * margin_per_conversion
        - fixed_monthly_cost
        - monthly_eligible_sessions * variable_cost_per_session
    )

economics = pd.Series({
    'incremental conversions at point estimate': monthly_eligible_sessions*lift,
    'net monthly value — point estimate': net_value(lift),
    'net monthly value — CI low': net_value(lift-1.96*lift_se),
    'net monthly value — CI high': net_value(lift+1.96*lift_se),
    'break-even lift': (
        fixed_monthly_cost + monthly_eligible_sessions*variable_cost_per_session
    ) / (monthly_eligible_sessions*margin_per_conversion),
})
economics"""),
md(r"""## 10. Final Growth readout

A defensible one-page conclusion contains four separate panels.

### A. Observed KPI

Report baseline, comparison, absolute percentage-point change, denominators, metric definition, and QA status.

### B. Exact descriptive decomposition

Report Kitagawa mix/rate totals for the joint operational hierarchy. When separate multidimensional labels are decision-relevant, add a Das Gupta/Shapley table with the factorization and reversal sensitivity stated explicitly. Report conservation errors for both views and never add their components together. Do not call either table incremental effects.

### C. Replicated localization

Report the frozen HCD leaf rule, both-period support, rates, total contribution, mix/rate refinement, and holdout interval. Note that ranking was learned on separate data.

### D. Causal and economic validation

Report randomization unit, eligibility, treatment/control sizes, lift and interval, guardrails, expected eligible volume, break-even lift, and value scenarios.

The recommended action can be:

- **launch** when replicated lift and downside economics clear thresholds;
- **iterate and retest** when the point estimate is promising but uncertainty crosses break-even;
- **do not launch** when the intervention fails even if the descriptive leaf was real.

A failed experiment does not invalidate the decomposition. It rejects this intervention or mechanism as the remedy for the localized descriptive change.

## Production checklist

- [ ] Metric contract versioned and reviewed.
- [ ] Numerator, denominator, population, and attribution window stable.
- [ ] Known-segment decomposition conserves the observed change.
- [ ] Adaptive discovery uses only approved pre-outcome descriptors.
- [ ] Final hierarchy is frozen before evaluation.
- [ ] Every leaf has support in both periods or an explicit entry/exit policy.
- [ ] One partition frontier—and only one—is summed.
- [ ] Descriptive and causal tables use different labels.
- [ ] Experiment assignment, exclusions, and guardrails are preregistered.
- [ ] Economics include uncertainty and transport assumptions.

## Limitations

- The simulation is cleaner than real event data and contains no delayed conversions.
- A single discovery/evaluation split does not measure full hierarchy instability.
- CART is used as a candidate partition generator, not a complete optimized HCD estimator.
- The rate-change interval ignores uncertainty from selecting the hierarchy, mitigated but not eliminated by holdout evaluation.
- The fixed-hierarchy bootstrap does not include variability from relearning the tree.
- Multidimensional allocations are exact conditional on their hybrid-population factorization, not invariant scientific truths.
- The experiment targets one discovered leaf; operational eligibility errors can dilute lift.
- Exact conservation makes the ledger auditable, not causal or uniquely invariant to segmentation.

## Exercises

1. Add a new channel that appears only in period 1 and implement the entry/exit policy.
2. Repeat discovery over 30 random splits and calculate leaf-membership stability.
3. Replace the greedy tree score with contribution-at-risk.
4. Compare direct channel × device decomposition with the HCD leaf frontier.
5. Add a negative guardrail effect and formulate a constrained launch decision.
6. Run the experiment on all traffic and estimate treatment-effect heterogeneity honestly.
7. Replace conversion with revenue per session and determine which accounting identity changes.

## References

- Kitagawa, E. M. (1955). Components of a difference between two rates. *JASA*, 50, 1168–1194.
- Das Gupta, P. (1978). A general method of decomposing a difference between two rates into several components. *Demography*, 15, 99–112.
- Breiman, L., Friedman, J. H., Olshen, R. A., & Stone, C. J. (1984). *Classification and Regression Trees*.
- Athey, S., & Imbens, G. (2016). Recursive partitioning for heterogeneous causal effects. *PNAS*, 113, 7353–7360.
- Kohavi, R., Tang, D., & Xu, Y. (2020). *Trustworthy Online Controlled Experiments*. Cambridge University Press.
- Working-paper companion: `working_paper/hierarchical_counterfactual_decomposition.md`."""),
]

# Normalize an accidental control character that can be introduced when this
# generator is patched through JSON-aware clients (backspace + "ar" -> LaTeX bar).
for growth_cell in growth_cells:
    growth_cell.source = (
        growth_cell.source
        .replace(chr(8) + 'ar', r'\overline')
        .replace(chr(9) + 'ext', r'\text')
        .replace('\n\\[\n', '\n$$\n')
        .replace('\n\\]\n', '\n$$\n')
        .replace(r'\bar Y_{D=0}', r'\overline Y_{D=0}')
    )

growth_nb = nbf.v4.new_notebook(
    cells=growth_cells,
    metadata={
        'kernelspec': {
            'display_name': 'Python 3 (uv)', 'language': 'python', 'name': 'python3'
        },
        'language_info': {'name': 'python', 'version': '3.12'},
    },
)
nbf.write(growth_nb, OUT / '07_complete_growth_workflow.ipynb')
print('built 07_complete_growth_workflow.ipynb')
