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
"- Derive the two-factor rate decomposition.\n- Recognize path dependence with three or more factors.\n- Implement symmetric and stepwise allocations.",
r"""## Setup

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
md("### Exercise\n\nBootstrap observations within segments, recompute the decomposition, and form percentile intervals. Why is algebraic exactness not the same as statistical certainty?")],
"""- Kitagawa, E. M. (1955). *JASA*, 50, 1168–1194. https://doi.org/10.1080/01621459.1955.10501299
- Das Gupta, P. (1978). A general method of decomposing a difference between two rates into several components. *Demography*, 15, 99–112. https://doi.org/10.2307/2060493
- Chevan, A., & Sutherland, M. (2009). Revisiting Das Gupta. *Demography*, 46, 429–449. https://doi.org/10.1353/dem.0.0051""")

notebooks['02_index_numbers_lmdi_pvm_sda.ipynb'] = lesson(
"02 — Index numbers, LMDI, PVM, shift–share, and SDA",
"- Compare Laspeyres, Fisher, and logarithmic-mean allocations.\n- Implement exact additive LMDI.\n- Understand PVM and SDA as decompositions of identities.",
r"""## Multiplicative identities

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
md("### Exercise\n\nReallocate the PVM interaction 50/50 and compare rankings. Explain why both answers can be exact yet non-unique.")],
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
"- Implement exact Shapley values.\n- Relate efficiency and symmetry to attribution.\n- Separate predictive attribution from causal attribution.",
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
md("### Exercise\n\nChange the baseline and recompute. Which axioms remain true? Why does baseline sensitivity weaken any claim that a contribution is intrinsic?")],
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
md(r"""## Derivation and interaction allocation

For one segment, write $w_1=w_0+\Delta w$ and $r_1=r_0+\Delta r$:

$$\Delta(wr)=r_0\Delta w+w_0\Delta r+\Delta w\Delta r.$$

Kitagawa splits the interaction equally:

$$C_w=\Delta w\left(r_0+\frac{\Delta r}{2}\right),\qquad
C_r=\Delta r\left(w_0+\frac{\Delta w}{2}\right).$$

Summing over $g$ yields exactness. The half-interaction rule is symmetric under exchanging periods, but it is an allocation convention—not an empirical discovery. With $K$ changing factors, stepwise replacement produces $K!$ paths; averaging their marginal increments is order-invariant and closely related to the Shapley value."""),
md(r"""## Growth-marketing case: conversion rate

Let $w_{gt}$ be the traffic share of channel $g$ and $r_{gt}$ its conversion rate. Then total CVR is $R_t=\sum_g w_{gt}r_{gt}$.

- **Mix contribution:** traffic moved toward channels with higher/lower CVR.
- **Rate contribution:** within-channel CVR changed.
- **Actionable diagnostic:** split further by device, geography, landing page, or cohort, but check sparse cells and post-treatment segmentation.

Example claim: “Total CVR rose 1.18 pp; under the Kitagawa two-period rule, +0.34 pp is allocated to channel mix and +0.84 pp to within-channel rates.” This is precise and descriptive. “The new campaign caused +0.84 pp” is not supported without an experiment or credible quasi-experiment."""),
code(r"""# Sensitivity to aggregation: collapse Returning and Enterprise
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
             index=['mix','rate'])"""),
md(r"""## Limitations, robustness, and inference

- Shares must sum to one in each period and segments must be defined consistently.
- New/disappearing categories and small cells require explicit handling.
- Results depend on segmentation; always repeat at plausible aggregation levels.
- Observed rate changes combine treatment, seasonality, selection, composition within cells, and noise.
- For estimated rates, use a stratified bootstrap or delta method; exactness conditional on estimates is not zero sampling variance.
- If channel is affected by treatment, conditioning on it can create post-treatment bias in a causal analysis.

## What came next

**Das Gupta (1978)** extended the logic to several compositional factors. **Chevan & Sutherland (2009)** revisited Das Gupta and supplied a general algorithmic treatment. In applications with many interacting factors, averaging stepwise-replacement orders leads naturally to the axiomatic decomposition presented by **Shorrocks (2013)**.""")],

'02_index_numbers_lmdi_pvm_sda.ipynb': [
md(r"""## Index-number formulas and exactness

For prices $p_t$ and quantities $q_t$, the Laspeyres and Paasche price indexes are

$$P_L=\frac{\sum_i p_{i1}q_{i0}}{\sum_i p_{i0}q_{i0}},\qquad
P_P=\frac{\sum_i p_{i1}q_{i1}}{\sum_i p_{i0}q_{i1}},$$

and Fisher's superlative index is $P_F=(P_LP_P)^{1/2}$. For a multiplicative identity $Y=\prod_k x_k$, the Divisia differential is

$$d\log Y=\sum_k \frac{\partial\log Y}{\partial\log x_k}d\log x_k.$$

LMDI replaces the continuous integral by logarithmic-mean weights and produces a perfect finite-change decomposition for positive data. PVM instead expands

$$\Delta(pq)=q_0\Delta p+p_0\Delta q+\Delta p\Delta q,$$

so the price–volume interaction must be reported or allocated by a declared rule."""),
md(r"""## Growth-marketing case: revenue and paid acquisition

A useful driver identity is

$$\text{Revenue}=\text{Impressions}\times\text{CTR}\times\text{CVR}\times\text{AOV}.$$

LMDI can allocate the observed change across scale, click efficiency, conversion, and order value. A second identity,

$$\text{Profit}=\text{Revenue}-\text{Spend},\qquad
\text{Spend}=\text{Clicks}\times\text{CPC},$$

prevents a favorable revenue attribution from being mistaken for incremental profit. Use channel-level components, report zeros explicitly, and distinguish gross from incremental conversions."""),
code(r"""# Growth funnel: exact LMDI allocation of revenue change
f0 = pd.Series({'impressions':1_000_000., 'ctr':.020, 'cvr':.040, 'aov':55.})
f1 = pd.Series({'impressions':1_150_000., 'ctr':.023, 'cvr':.037, 'aov':58.})
y0, y1 = f0.prod(), f1.prod()
weight = L(np.array([y1]), np.array([y0]))[0]
funnel_contrib = weight * np.log(f1/f0)
pd.concat([funnel_contrib.rename('contribution'),
           (100*funnel_contrib/(y1-y0)).rename('share_pct')], axis=1).assign(
               total_change=y1-y0 if False else np.nan)
"""),
md(r"""## Limitations and robustness

- Standard LMDI requires positive values; zeros need documented limiting conventions and negative values can invalidate logs.
- Identity decompositions are sensitive to the chosen factorization: CTR×CVR and clicks×orders-per-click are algebraically related but tell different stories.
- Chained indexes reduce base-period staleness but introduce chain drift and revisions.
- PVM “mix” is often mislabeled: changing product shares, pure volume, and interaction should be separated.
- SDA is computationally expensive as determinants grow and inherits input–output measurement error.
- None of these methods corrects endogeneity, anticipatory behavior, ad auctions, or cross-channel substitution.

## What came next

**Diewert (1976)** formalized exact and superlative indexes, including Fisher's desirable approximation properties. For energy/environment identities, **Ang & Choi (1997)** introduced a refined logarithmic-mean Divisia method; **Ang (2005)** consolidated the preferred LMDI formulation, and **Ang (2015)** provided a practical guide. In structural decomposition, **Dietzenbacher & Los (1998)** showed why averaging polar forms or all paths matters when multiple determinants change.""")],

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
md(r"""## Conditional, marginal, and causal games

For a prediction model $f$ and explained point $x$, common coalition games include

$$v_{\text{cond}}(S)=E[f(X)\mid X_S=x_S],$$

and

$$v_{\text{marg}}(S)=E_{X_{-S}}[f(x_S,X_{-S})].$$

The conditional game respects observed dependence but may assign credit to a feature unused by $f$ through correlation. The marginal/interventional game breaks dependence and can evaluate unrealistic combinations. A causal game would instead be defined from an SCM, for example $v_{\text{causal}}(S)=E[Y\mid do(X_S=x_S)]$; it is a different estimand, not a switch in plotting software.

Computing exact Shapley values costs $O(2^p)$ coalition evaluations (or $p!$ paths); practical SHAP methods exploit model structure or Monte Carlo approximation."""),
md(r"""## Growth-marketing case: churn and conversion models

Use SHAP to debug a fitted churn or conversion model, detect leakage, compare segments, and explain why the model scored a user highly. Do not use it to rank budget interventions unless the model and coalition game encode the intervention process.

Examples of traps:

- `discount_seen` is post-treatment and can dominate prediction without being an actionable cause.
- channel and geography are correlated; conditional and marginal SHAP answer different questions.
- aggregating $|\phi_j|$ hides direction and can favor high-cardinality/noisy features.
- background data from last quarter makes attribution drift when the user mix changes."""),
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
md(r"""## Limitations and robustness

- Attribution is model-specific; a misspecified or poorly calibrated model can be explained perfectly.
- Results depend on baseline/background distribution and missing-feature semantics.
- Correlated features make credit allocation scientifically ambiguous.
- Exactness/efficiency does not imply stability; report Monte Carlo error and variation across folds/seeds/background samples.
- Global mean absolute SHAP is not a causal elasticity and not necessarily useful for intervention.
- Functional ANOVA orthogonality depends on the input measure; dependent inputs complicate uniqueness.
- Validate predictive performance out of sample, test leakage, group collinear features, and compare conditional versus marginal games.

## What came next

**Lundberg & Lee (2017)** connected additive feature attribution to Shapley values. **Lundberg et al. (2020)** developed TreeSHAP-based explanations for tree ensembles. **Aas, Jullum & Løland (2021)** addressed dependent features with conditional distributions. **Sundararajan & Najmi (2020)** clarified the many distinct “Shapley” games, while **Heskes et al. (2020)** and related work developed causal Shapley values using explicit causal structure.""")],

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
""",
'01_rates_kitagawa_dasgupta.ipynb': r"""
- Das Gupta, P. (1993). *Standardization and Decomposition of Rates: A User's Manual*. U.S. Census Bureau.
- Shorrocks, A. F. (2013). Decomposition procedures for distributional analysis. *Journal of Economic Inequality*, 11, 99–126. https://doi.org/10.1007/s10888-011-9214-z
""",
'02_index_numbers_lmdi_pvm_sda.ipynb': r"""
- Ang, B. W., & Choi, K.-H. (1997). Decomposition of aggregate energy and gas emission intensities for industry: A refined Divisia index method. *The Energy Journal*, 18(3), 59–73.
- Ang, B. W. (2015). LMDI decomposition approach: A guide for implementation. *Energy Policy*, 86, 233–238. https://doi.org/10.1016/j.enpol.2015.07.007
- Dietzenbacher, E., & Los, B. (1998). Structural decomposition techniques: Sense and sensitivity. *Economic Systems Research*, 10, 307–324. https://doi.org/10.1080/09535319800000023
""",
'03_oaxaca_reweighting_rif.ipynb': r"""
- Juhn, C., Murphy, K. M., & Pierce, B. (1993). Wage inequality and the rise in returns to skill. *Journal of Political Economy*, 101, 410–442. https://doi.org/10.1086/261881
- Machado, J. A. F., & Mata, J. (2005). Counterfactual decomposition of changes in wage distributions using quantile regression. *Journal of Applied Econometrics*, 20, 445–465. https://doi.org/10.1002/jae.788
- Fortin, N., Lemieux, T., & Firpo, S. (2011). Decomposition methods in economics. In *Handbook of Labor Economics*, Vol. 4A, 1–102. https://doi.org/10.1016/S0169-7218(11)00407-2
""",
'04_shapley_anova_ml.ipynb': r"""
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

We first establish the identity, then calculate it, interpret the output, and finally test whether the answer changes when segments are aggregated.""",
'02_index_numbers_lmdi_pvm_sda.ipynb': r"""## Roadmap and notation

### Guiding question

How can a change in a total generated by several multiplicative factors be allocated without leaving an unexplained residual?

### Prerequisites

Products, logarithms, and before/after comparisons. No calculus is required to run the examples; the Divisia differential is included to explain the origin of LMDI.

### Symbols

- $i$: product, channel, sector, or other component.
- $Q_{it}$: activity or quantity; $I_{it}$: intensity or rate.
- $E_{it}=Q_{it}I_{it}$: component total.
- $L(a,b)$: logarithmic mean, a symmetric weight between positive $a$ and $b$.
- $p_{it},q_{it}$: price and quantity in PVM/index-number notation.

The notebook treats LMDI and PVM separately because both decompose identities but allocate interactions differently.""",
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

Sets and averages. Knowledge of machine learning is useful only for the SHAP section.

### Symbols

- $N=\{1,\ldots,p\}$: all players or features.
- $S\subseteq N$: coalition already present.
- $v(S)$: value produced by coalition $S$.
- $v(S\cup\{j\})-v(S)$: marginal contribution of player $j$ after $S$.
- $\phi_j$: Shapley allocation to player $j$.

We begin with an exact business identity. Only afterward do we map the same allocation logic to model explanations, where the definition of $v(S)$ becomes a substantive modeling choice.""",
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
3. Choose a method by the mathematical object, not by dashboard terminology.

Notebook 01 now specializes this framework to the simplest important object: an aggregate rate formed as a weighted average of segment rates.""",
'01_rates_kitagawa_dasgupta.ipynb': r"""## Takeaways and bridge to Notebook 02

1. Kitagawa separates mix from within-segment rate changes exactly.
2. The equal interaction split is symmetric but conventional.
3. Segmentation and sampling uncertainty can materially change the story.
4. Use causal language only with a separate design.

Notebook 02 moves from weighted rates to totals generated by multiplicative business identities.""",
'02_index_numbers_lmdi_pvm_sda.ipynb': r"""## Takeaways and bridge to Notebook 03

1. LMDI gives an exact decomposition for positive multiplicative data.
2. PVM and index-number methods differ mainly in weighting and interaction allocation.
3. Factorization is a modeling choice, even when the identity is exact.
4. These are descriptive contributions, not elasticities or incrementality estimates.

Notebook 03 changes the target from an accounting total to a gap estimated from individual-level regressions.""",
'03_oaxaca_reweighting_rif.ipynb': r"""## Takeaways and bridge to Notebook 04

1. Oaxaca–Blinder splits a fitted mean gap into composition and structure.
2. Reference coefficients, support, specification, and selection matter.
3. Bootstrap uncertainty and report sensitivity across credible specifications.
4. Distributional extensions answer questions beyond the mean.

Notebook 04 replaces reference-coefficient choices with an axiomatic rule that averages marginal contributions over coalitions.""",
'04_shapley_anova_ml.ipynb': r"""## Takeaways and bridge to Notebook 05

1. Shapley values solve an allocation problem once the game $v(S)$ is fixed.
2. Efficiency and symmetry do not remove baseline or model dependence.
3. SHAP explains predictions under a chosen missing-feature semantics.
4. Predictive attribution is not intervention effect.

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
        formal, application, limits_next = deep
        cells[:] = [title, md(ORIENTATION[filename]), theory, formal, setup,
                    *examples, md(RESULT_GUIDES[filename]), application,
                    limits_next, md(SUMMARIES[filename]), exercise,
                    checklist, references]
    else:
        derivation, application, robustness_code, limits_next = deep
        cells[:] = [title, md(ORIENTATION[filename]), theory, derivation,
                    setup, *examples, md(RESULT_GUIDES[filename]), application,
                    robustness_code, limits_next, md(SUMMARIES[filename]),
                    exercise, checklist, references]
    references.source += "\n" + ADDITIONAL_REFS[filename].strip()
    nb=nbf.v4.new_notebook(cells=cells, metadata={'kernelspec':{'display_name':'Python 3 (uv)','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12'}})
    nbf.write(nb, OUT/filename)
    print(f"built {filename}")
