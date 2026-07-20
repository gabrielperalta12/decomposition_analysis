# Hierarchical Counterfactual Decomposition (HCD)

## Exact, regularized discovery of where aggregate KPI changes occur

**Working-paper artifact — research design, version 0.1**  
**Status:** conceptual proposal with formal estimand, candidate algorithms, falsifiable claims, and validation plan.  
**Application domain:** Product Analytics, Growth, experimentation, metric monitoring, and descriptive root-cause localization.

---

## Abstract

Product and Growth teams routinely observe a change in an aggregate KPI and then inspect hundreds or thousands of intersections—channel, device, country, cohort, plan, tenure, and behavior—to determine where it occurred. Classical decomposition methods guarantee that labelled contributions reproduce the aggregate change, but generally require analysts to specify the relevant factors or cells in advance. Decision trees, CHAID, Subgroup Discovery, and rule ensembles can search automatically for heterogeneous patterns, but their selected segments do not generally form an additive decomposition of the observed KPI change; overlapping rules can double count it, pruning can discard it, and predictive importance need not have the units of the KPI.

This paper proposes **Hierarchical Counterfactual Decomposition (HCD)**: a framework that separates an exact accounting layer from a regularized discovery layer. For any tree partition of the population, leaf contributions are defined as changes in additive KPI mass and therefore sum exactly to the observed aggregate change. A search algorithm chooses a compact hierarchy whose leaves concentrate large, stable, and interpretable contributions. Honest sample splitting, hierarchical shrinkage, and post-selection validation reduce spurious segments. The output distinguishes composition, within-segment performance, entry/exit, and residual uncertainty while preserving the accounting identity. Descriptive discoveries become hypotheses for later experiments or quasi-experiments; they are not relabelled causal effects.

The proposed novelty is not “Shapley plus trees” in the abstract. Closely related work already studies distribution-change attribution, Shapley explanations of model-performance shifts, tree-defined subgroup explanations, causal trees, and subgroup discovery. The narrower research contribution is a unified method for **observed business KPIs** that jointly provides: exact conservation at every hierarchy level, adaptive but constrained segment discovery, regularized and honest reporting, explicit handling of unstable or absent cells, and a formal bridge from descriptive localization to causal validation.

---

## 1. Research question

Let an aggregate KPI change between periods (t=0) and (t=1):

\[
\Delta K = K_1-K_0.
\]

The operational question is:

> Can we discover a small, interpretable hierarchy of subgroups that localizes the observed KPI change while preserving the exact identity and controlling adaptive-search error?

The desired output is not merely a ranked list of correlates. It is a hierarchy such as:

```text
All traffic: ΔCVR = −0.80 pp
├── Paid Search: +0.25 pp contribution
│   ├── Mobile, high intent: +0.42 pp
│   └── Remainder of Paid Search: −0.17 pp
└── Other channels: −1.05 pp contribution
    ├── Social, new users: −0.72 pp
    └── Remainder: −0.33 pp
```

Only a **partition frontier**—normally the leaves—may be summed. Parents are roll-ups of their descendants, not additional contributions. This prevents the double counting common in dashboards that display both a parent segment and its intersections as if all rows were additive.

---

## 2. Terminology: is “counterfactual” the right word?

HCD initially constructs **descriptive hybrid scenarios**: for example, the period-1 segment mix evaluated at period-0 within-segment rates. These are counterfactual in the broad mathematical sense of “what would the aggregate be if one observed component were held fixed,” but they are not potential outcomes under interventions.

This distinction should appear in the title or subtitle of any paper. Three naming options are defensible:

1. **Hierarchical Counterfactual Decomposition**, retaining HCD but explicitly saying “descriptive counterfactual”;
2. **Hierarchical Conserving Decomposition**, emphasizing the exact adding-up property and avoiding causal overtones;
3. **Hierarchical Change Decomposition**, the clearest product-analytics name but the least theoretically distinctive.

The working recommendation is to retain **HCD** provisionally while defining two layers:

- **HCD-D:** descriptive decomposition of an observed contrast;
- **HCD-C:** a later causal extension in which the primitive contrast is identified from potential outcomes or a randomized experiment.

No HCD-D output should use “incrementality,” “impact,” or “caused” without an additional identification design.

---

## 3. Why existing families do not fully solve the workflow

### 3.1 Exact decomposition

Kitagawa, Das Gupta, LMDI, Shapley, and Shorrocks provide exact allocations under declared factorization, reference, or coalition rules. They answer “how should a known change be divided?” They do not generally search a massive segment language and choose a compact hierarchy.

### 3.2 Trees and rule learners

CART and CHAID find partitions; Subgroup Discovery and RuleFit find interpretable patterns. They can answer “where is the target unusual?” But:

- CART impurity reduction is not an additive KPI contribution;
- CHAID significance does not guarantee business materiality or conservation;
- overlapping subgroup rules cannot be summed without a reconciliation rule;
- RuleFit coefficients are conditional predictive parameters, not contributions to an observed aggregate change;
- ordinary feature importance has neither the units nor the conservation identity of the KPI.

### 3.3 Heterogeneous causal-effect methods

Causal trees and causal forests search for treatment-effect heterogeneity under experimental or observational identification assumptions. They solve a different problem from a before/after KPI diagnosis. They become relevant after an intervention and estimand have been defined.

### 3.4 Distribution-shift and metric-attribution methods

This is the closest neighboring literature. Budhathoki et al. define Shapley attribution for distribution changes; Zhang et al. attribute model-performance changes to changing data-generating distributions; ShapShift explains shifts in average model predictions using tree-defined subgroups and conditional Shapley values. Recent work also proposes additive attribution for arbitrary measures and recursive metric trees.

Therefore, a credible HCD paper must not claim that nobody has combined trees with Shapley attribution. Its differentiating target should be the joint package of:

1. an **observed KPI**, rather than only a fitted-model prediction or model-performance metric;
2. exact conservation for a **data-adaptive segment hierarchy**;
3. explicit rate/mix and entrant/exit semantics;
4. regularized selection with honest statistical evaluation;
5. hierarchy-coherent reporting and causal follow-up.

---

## 4. Formal accounting layer

### 4.1 Additive KPI mass

Let (X_i) be pre-period segment descriptors, (T_i\in\{0,1\}) the period, and (Y_i) an individual outcome. For a rate KPI, define for any measurable subgroup (A\subseteq\mathcal X):

\[
w_t(A)=P_t(X\in A),
\qquad
r_t(A)=E_t[Y\mid X\in A].
\]

The subgroup's KPI mass is

\[
m_t(A)=w_t(A)r_t(A)=E_t[Y\mathbf 1\{X\in A\}].
\]

Define its raw contribution to change as the signed measure

\[
c(A)=m_1(A)-m_0(A).
\]

For a finite partition \(\mathcal P=\{A_1,\ldots,A_L\}\),

\[
\boxed{
\Delta K
=K_1-K_0
=\sum_{\ell=1}^{L}c(A_\ell).
}
\]

This identity is true for **every** partition. Exact conservation does not need to be optimized or approximately learned; it is built into the definition.

### 4.2 Proposition 1: partition conservation

**Proposition.** If the leaves of a tree are mutually exclusive and exhaustive, their HCD contributions sum exactly to the observed KPI change.

**Proof sketch.** The indicator functions of the leaves satisfy

\[
\sum_{\ell=1}^{L}\mathbf 1\{X\in A_\ell\}=1.
\]

Therefore,

\[
\sum_\ell m_t(A_\ell)
=E_t\left[Y\sum_\ell\mathbf 1\{X\in A_\ell\}\right]
=E_t[Y]=K_t.
\]

Subtracting the two period identities proves the result.

### 4.3 Proposition 2: hierarchical coherence

For a node (A) split into disjoint children (A_L,A_R),

\[
c(A)=c(A_L)+c(A_R).
\]

Thus every internal-node contribution is an exact roll-up. A dashboard can be expanded or collapsed without changing the total, provided it sums one frontier only.

### 4.4 Rate and composition refinement

The raw leaf contribution can be decomposed with Kitagawa:

\[
c(A)=c_w(A)+c_r(A),
\]

where

\[
c_w(A)=\big[w_1(A)-w_0(A)\big]
\frac{r_1(A)+r_0(A)}{2},
\]

\[
c_r(A)=\big[r_1(A)-r_0(A)\big]
\frac{w_1(A)+w_0(A)}{2}.
\]

For a stable partition,

\[
\sum_A c_w(A)+\sum_A c_r(A)=\Delta K.
\]

This gives every leaf three interpretable outputs:

- total contribution (c(A));
- mix contribution (c_w(A));
- within-segment rate contribution (c_r(A)).

### 4.5 Entrants and exits

If (w_0(A)=0), then (m_0(A)=0) and the total entrant contribution

\[
c(A)=w_1(A)r_1(A)
\]

is observed. But (r_0(A)) is not observed, so its mix/rate split is not identified. HCD must either:

1. report a separate entry/exit contribution;
2. use an explicit reference rate with sensitivity analysis;
3. merge the leaf into a parent with support in both periods.

Silently setting an absent-period rate to zero confuses zero exposure with zero performance.

### 4.6 General KPIs

For totals of the form (K_t=\sum_i y_{it}), leaf additivity is immediate. For ratios (K_t=N_t/D_t), a leaf's numerator and denominator must be carried separately. A ratio cannot generally be decomposed by summing leaf-level ratios. Candidate solutions include:

- share-weighted rate decomposition when (D_t>0);
- exact Shapley/Das Gupta allocation over numerator and denominator factors;
- influence-function linearization for smooth nonlinear metrics;
- an exact arbitrary-measure attribution engine when available.

The first paper should restrict the main theorem to totals and weighted rates, then treat arbitrary differentiable metrics as an extension.

---

## 5. Discovery layer

Conservation alone is trivial once a partition is fixed. The research problem is choosing a useful partition without manufacturing noise.

### 5.1 Candidate tree

Let \(\mathbb T_d\) be the set of trees of maximum depth (d), constructed from an allowed description language:

- equality tests for categorical features;
- threshold tests for ordered features;
- optional domain-approved grouped categories;
- no post-period or post-treatment variables;
- minimum exposure in both periods.

Each tree (\mathcal T\) induces leaves \(\mathcal L(\mathcal T)\) and exact raw contributions \(c(A)\).

### 5.2 What should the tree maximize?

“Maximize contribution” is incomplete because the total signed contribution is constant:

\[
\sum_{A\in\mathcal L(\mathcal T)}c(A)=\Delta K
\]

for every tree. A useful objective must reward **concentration, stability, and interpretability**, not the total itself.

A candidate objective is

\[
J(\mathcal T)=
\sum_{A\in\mathcal L(\mathcal T)}
\omega(A)\rho\!\left(\frac{c(A)}{s(A)}\right)
-\lambda_L|\mathcal L(\mathcal T)|
-\lambda_D\operatorname{depth}(\mathcal T)
-\lambda_U U(\mathcal T),
\]

where:

- (s(A)) is an uncertainty scale;
- \(\rho\) is a robust utility, such as Huber loss or a capped square;
- \(\omega(A)) penalizes low support;
- (U(\mathcal T)) measures instability across bootstrap samples or time folds;
- the lambdas control complexity.

An operational alternative ranks leaves by **contribution at risk**:

\[
\operatorname{CaR}_\alpha(A)
=|\hat c(A)|-z_{1-\alpha/2}\widehat{SE}\{\hat c(A)\}.
\]

Positive CaR means the contribution remains material after an uncertainty penalty. It should be described as a selection score, not a formal simultaneous confidence guarantee unless multiplicity is handled.

### 5.3 Split gain

For parent (A) and candidate children (A_L,A_R), define

\[
G(A\to A_L,A_R)
=u(A_L)+u(A_R)-u(A)-\lambda,
\]

with (u(A)=\omega(A)\rho(c(A)/s(A))). The split is allowed only when:

- both children satisfy minimum baseline and comparison denominators;
- the gain replicates across internal folds;
- no prohibited or leakage-prone feature is used;
- the resulting definition remains operationally interpretable.

Greedy splitting is scalable but not globally optimal. Beam search, optimal trees, mixed-integer optimization, or Bayesian tree priors are plausible alternatives.

### 5.4 Signed change versus gross movement

If positive and negative contributions cancel, optimizing only (|\Delta K|) can miss large internal movement. Report:

\[
G^+=\sum_A\max(c(A),0),
\qquad
G^-=\sum_A\min(c(A),0),
\]

\[
\Delta K=G^++G^-,
\qquad
M=G^+-G^-=\sum_A|c(A)|.
\]

(M) is gross movement and is segmentation-sensitive. It should be regularized and compared only across declared candidate hierarchies.

---

## 6. Statistical layer: preventing spurious segments

### 6.1 Honest sample splitting

Use at least three roles:

1. **discovery sample:** choose splits and depth;
2. **estimation sample:** calculate frozen-leaf contributions and intervals;
3. **temporal replication sample:** verify stability in a later window.

Cross-fitting can rotate these roles and average estimates, but the final reported hierarchy must remain well defined.

### 6.2 Hierarchical shrinkage

For a binary outcome, model leaf-period conversions as

\[
Y_{Alt}\sim\operatorname{Binomial}(n_{Alt},\theta_{Alt}),
\]

\[
\operatorname{logit}(\theta_{Alt})
=\mu_t+\sum_{v\in\operatorname{path}(A)}b_{vt},
\qquad
b_{vt}\sim N(0,\sigma^2_{\operatorname{depth}(v)}).
\]

Depth-dependent priors shrink narrow leaves more strongly toward parents. Partial pooling stabilizes ranking and prediction.

However, posterior mean contributions may no longer reproduce the **observed** change exactly. HCD should not hide this tension. It should provide two coordinated layers:

- **raw accounting contributions**, exactly conserving the observed KPI;
- **shrunk signal estimates**, used for selection and uncertainty.

If a single reconciled vector is required, solve the constrained projection

\[
\tilde c
=\arg\min_z (z-\hat c)^{\top}V^{-1}(z-\hat c)
\quad\text{s.t.}\quad \mathbf 1^{\top}z=\Delta K.
\]

The solution is

\[
\tilde c
=\hat c+
V\mathbf 1
\frac{\Delta K-\mathbf 1^{\top}\hat c}
{\mathbf 1^{\top}V\mathbf 1}.
\]

This is a minimum-distance reconciliation, not evidence that the adjusted leaf effects are true.

### 6.3 Multiplicity

Searching (10^5) rules and reporting the largest ordinary (t)-statistic is invalid. Candidate safeguards include:

- holdout confirmation;
- permutation distribution of the **maximum** tree or rule score;
- selective-inference methods when their assumptions match the search;
- false-discovery control for a frozen candidate family;
- stability selection across bootstraps;
- empirical-Bayes shrinkage of leaf contrasts.

The primary paper should make one inferential claim only: honest holdout intervals for a hierarchy frozen before evaluation. More ambitious selective inference can follow.

### 6.4 Stability

Define split stability or leaf stability across resamples. For leaf rule (A), a Jaccard-style membership stability is

\[
\operatorname{Stab}(A)
=E_{b\neq b'}
\frac{|A^{(b)}\cap A^{(b')}|}{|A^{(b)}\cup A^{(b')}|}.
\]

Because different rules can describe nearly identical populations, prediction-based or membership-based stability is preferable to exact string matching.

---

## 7. Candidate HCD algorithm

### 7.1 HCD-D v1

```text
Input:
    two periods of event or user-level data
    KPI numerator/outcome and denominator
    approved pre-period descriptors
    maximum depth, minimum support, penalties

1. Validate metric definitions and population comparability.
2. Split data into discovery and honest evaluation samples.
3. Start with a root containing the whole population.
4. For every admissible split:
       compute child raw contributions exactly;
       estimate uncertainty or shrinkage score;
       calculate penalized split gain;
       reject insufficient-support or unstable splits.
5. Apply the best positive-gain split.
6. Repeat until no admissible gain remains or budget is exhausted.
7. Freeze the tree.
8. On evaluation data, calculate for every leaf:
       baseline share and rate;
       comparison share and rate;
       total, mix, rate, and entry/exit contribution;
       uncertainty interval and stability diagnostics.
9. Verify:
       sum(leaves) = observed KPI change;
       parent = sum(children) at every internal node.
10. Rank leaves without changing their additive values.
11. Send replicated, actionable leaves to causal validation.
```

### 7.2 Important architectural decision

The split criterion and contribution definition must be separated:

- the **contribution** is fixed by the accounting estimand;
- the **search score** chooses which partition to display;
- the **regularizer** discourages fragile complexity;
- the **causal module** evaluates interventions later.

Allowing the model to redefine contribution to improve predictive fit would sacrifice the main scientific advantage of HCD.

### 7.3 Overlapping rules

Subgroup Discovery and RuleFit naturally return overlapping rules. Exact conservation then requires one of four strategies:

1. convert selected rules into disjoint atoms;
2. impose a priority order and assign each observation to its first matching rule;
3. compute a Shapley allocation over rule coverage;
4. restrict the primary method to tree partitions and use overlapping rules only as candidate generators.

Version 1 should use option 4. It is easiest to audit and guarantees a readable hierarchy. A later “HCD-Rules” extension can study overlap explicitly.

---

## 8. Desired axioms and properties

| Property | Requirement | Status under proposed HCD-D |
|---|---|---|
| Conservation | leaf contributions sum to (Delta K) | exact by construction |
| Hierarchical coherence | parent equals sum of children | exact by construction |
| Unit consistency | contributions use KPI units | yes |
| Null segment | unchanged KPI mass gets zero raw contribution | yes |
| Row-order invariance | reordering observations changes nothing | yes |
| Symmetry of mix/rate interaction | reverse periods changes signs | Kitagawa layer |
| Honest evaluation | search and final estimation separated | procedural guarantee |
| Stability | nearby samples give similar hierarchy | objective/diagnostic, not automatic |
| Aggregation invariance | refinements cannot alter component labels | generally impossible |
| Causal validity | contribution equals intervention effect | no; requires HCD-C assumptions |

The impossibility of full aggregation invariance should be explicit. Refining a segment can change the split between mix and within-segment rate even when the total change is preserved. HCD guarantees conservation, not uniqueness across all possible segment languages.

---

## 9. Transition to causal validation

### 9.1 Descriptive output

HCD-D supports the statement:

> “Under this population definition and hierarchy, Mobile Paid Search users with at least four recent sessions account for (x) percentage points of the observed KPI change.”

It does not support:

> “Mobile Paid Search caused the KPI change.”

### 9.2 HCD-C for a randomized intervention

Suppose (D\in\{0,1\}) is randomized treatment and (Y(1),Y(0)) are potential outcomes. For disjoint leaves (A), define

\[
\tau(A)=E[Y(1)-Y(0)\mid X\in A].
\]

A population-scale incremental contribution could be

\[
c_{\mathrm{causal}}(A)=P(X\in A)\tau(A),
\]

which satisfies

\[
\sum_A c_{\mathrm{causal}}(A)=E[Y(1)-Y(0)]
\]

for a fixed partition. Honest causal trees or forests can discover heterogeneity, while a final partition preserves aggregation of the estimated average treatment effect.

This is a natural extension but not the same estimand as the before/after HCD-D change. The paper should keep the two analyses in separate panels.

### 9.3 Growth workflow

1. HCD-D detects and localizes a KPI change.
2. Analysts inspect logging, operational, and policy changes affecting the leaf.
3. A mechanism and intervention are specified.
4. The leaf definition is preregistered or learned with honest causal methods.
5. An A/B test or credible quasi-experiment estimates incrementality.
6. HCD-C aggregates heterogeneous causal effects across a disjoint target hierarchy.

The transition is a workflow, not an algebraic conversion of descriptive contributions into causal effects.

---

## 10. Research hypotheses

The paper should test falsifiable hypotheses rather than assume HCD is superior.

### H1: accounting fidelity

HCD has machine-precision conservation and hierarchical coherence; CART importance, overlapping subgroup scores, and RuleFit coefficients do not generally reproduce the KPI change.

### H2: localization recovery

When the true change is concentrated in sparse interactions, regularized HCD recovers the relevant population with higher precision at a fixed leaf budget than manual one-way slicing and ordinary CART.

### H3: false-discovery control

Honest HCD produces lower discovery-to-holdout exaggeration than an adaptive tree evaluated on its training data.

### H4: stability

Hierarchical shrinkage and temporal stability penalties improve leaf-membership and contribution stability relative to unpruned CART and unconstrained beam search.

### H5: decision usefulness

At a fixed analyst-review budget, HCD surfaces more replicated, actionable segments than exhaustive dashboard slicing.

H5 requires a user study or retrospective decision audit; it cannot be established from predictive metrics alone.

---

## 11. Simulation program

### 11.1 Data-generating processes

Vary:

- number of descriptors and category cardinality;
- correlation among channel, device, market, cohort, and behavior;
- balanced versus drifting segment composition;
- homogeneous versus sparse interaction changes;
- positive/negative cancellation;
- entrant and exiting cells;
- small denominators and rare outcomes;
- metric types: totals, binary rates, continuous averages, and ratios;
- stationary versus temporally unstable patterns.

### 11.2 Baselines

Compare HCD with:

- manual one-way and two-way slicing;
- regularized CART on a period-contrast pseudo-outcome;
- CHAID;
- Subgroup Discovery with weighted relative accuracy or unusualness;
- RuleFit;
- causal tree only in randomized-treatment scenarios;
- ShapShift or other metric-shift attribution where the target is compatible;
- a no-segmentation root-only model.

### 11.3 Metrics

Measure:

- conservation error;
- parent/child coherence error;
- recovery of true subgroup membership;
- signed contribution error;
- top-(k) precision and recall;
- discovery/holdout shrinkage ratio;
- coverage of holdout intervals;
- hierarchy stability;
- number and depth of leaves;
- computation time;
- analyst-rated interpretability.

An exact total with incorrect leaves is not success. Conservation is necessary but insufficient.

---

## 12. Empirical study design

A compelling application needs event-level data with:

- a stable KPI specification;
- known logging changes;
- enough volume in both periods;
- descriptors available before the outcome;
- at least one later experiment or operational incident for validation.

Recommended studies:

1. **Retrospective product incident:** can HCD recover the population affected by a known bug?
2. **Marketing mix shift:** can it distinguish channel-share movement from within-channel CVR deterioration?
3. **Experiment audit:** ignoring treatment labels, can descriptive HCD localize where the observed outcome moved; and how does that compare with honest treatment-effect heterogeneity?

The experiment audit is especially informative because it reveals when descriptive change localization and causal heterogeneity agree—and when random composition noise makes them diverge.

---

## 13. Threats to validity

### 13.1 Scientific

- The selected hierarchy depends on the allowed feature and rule language.
- Conservation can create false confidence: a perfectly additive explanation can still be substantively wrong.
- Pre/post contrasts mix secular trends, selection, seasonality, instrumentation, and intervention effects.
- Segment definitions may be post-outcome or post-treatment.
- Fine partitions change the mix/rate allocation even when total conservation survives.

### 13.2 Statistical

- Adaptive search produces winner's curse and invalid naive (p)-values.
- Rare leaves generate unstable rates and extreme contributions.
- Correlated descriptors permit several near-equivalent trees.
- Sample splitting reduces effective sample size.
- Standard bootstrap procedures must repeat the discovery process if they aim to capture model-selection uncertainty.

### 13.3 Operational

- Protected attributes and proxies can create unfair or legally restricted targeting.
- A statistically stable leaf may not map to an available intervention.
- Logging changes can appear as business changes.
- Users may incorrectly interpret ranked descriptive contributions as causes.

---

## 14. Novelty assessment as of July 2026

The broad claim that “the pieces exist separately but no work combines exact attribution and subgroup trees” is no longer safe. The most relevant nearby work includes:

- distribution-change attribution using Shapley values;
- attribution of model-performance shifts to changed distributions;
- ShapShift, which uses tree-defined subgroups and conditional Shapley values to explain average prediction shifts;
- additive attribution frameworks for arbitrary measures;
- recursive metric-tree root-cause analysis;
- classical Subgroup Discovery and Exceptional Model Mining;
- causal trees and forests for treatment-effect heterogeneity.

The defensible gap is narrower:

| Dimension | Typical neighboring focus | Proposed HCD focus |
|---|---|---|
| Target | model prediction/performance or unusual subgroup target | observed product KPI change |
| Structure | features, distributions, or discovered rules | disjoint hierarchical population partition |
| Adding up | model-specific Shapley efficiency or approximate coverage | exact leaf and parent/child conservation |
| Rate semantics | often generic | explicit mix, within-rate, entry/exit |
| Search control | predictive regularization or approximation | honest discovery plus hierarchy stability |
| Output | attribution or ranked patterns | auditable KPI ledger and hypothesis queue |
| Causal link | separate causal RCA or HTE literature | explicit staged transition HCD-D → HCD-C |

This gap may still be publishable, but it must be demonstrated through formal properties and comparisons with the closest 2023–2026 methods.

---

## 15. Minimum viable paper

The first paper should avoid solving every extension. A focused contribution would contain:

1. weighted-rate estimand and exact hierarchical-conservation theorem;
2. binary partition trees with minimum support in both periods;
3. a penalized contribution-at-risk split objective;
4. honest discovery/evaluation protocol;
5. raw and Kitagawa-refined leaf contributions;
6. explicit entrant/exit handling;
7. simulations against CART, Subgroup Discovery, and manual slicing;
8. one real Growth or Product incident;
9. a separate discussion—not implementation—of HCD-C.

Later papers could address:

- overlapping HCD rules and Shapley reconciliation;
- arbitrary nonlinear metrics;
- Bayesian tree priors and constrained posterior reconciliation;
- streaming and sequential false-alarm control;
- causal HCD under randomized and observational designs;
- multi-KPI constrained discovery;
- privacy-preserving and fairness-constrained hierarchies.

---

## 16. Proposed paper outline

1. Introduction and Product Analytics motivation
2. Related work and novelty boundary
3. KPI-mass estimand and exact hierarchy
4. Penalized HCD tree search
5. Honest inference and hierarchical shrinkage
6. Simulations
7. Product/Growth case study
8. Causal-validation protocol
9. Limitations and ethics
10. Discussion

---

## 17. Immediate implementation roadmap

### Phase A: reference implementation

- accept event-level or aggregated two-period data;
- validate shares, denominators, missing rates, and segment support;
- implement exact leaf contribution ledger;
- implement greedy binary splits;
- expose mix, rate, entry/exit, uncertainty, and hierarchy checks;
- produce a machine-readable tree plus analyst report.

### Phase B: statistical validation

- honest discovery/evaluation split;
- temporal cross-validation;
- permutation maximum-score calibration;
- hierarchical empirical-Bayes shrinkage;
- stability selection.

### Phase C: research benchmark

- public simulation suite;
- CART, CHAID, Subgroup Discovery, RuleFit, and ShapShift-compatible baselines;
- predeclared recovery and false-discovery metrics;
- ablation of conservation, regularization, and stability penalties.

### Phase D: causal bridge

- randomized experiment module;
- honest causal partitioning;
- population-weighted leaf CATE ledger;
- comparison between descriptive and causal hierarchies.

---

## 18. Concluding position

The central idea is strong, but the exact contribution requires discipline. Conservation should not be treated as a property that a tree “learns”; it follows from defining subgroup contributions as additive KPI mass on a partition. The methodological work lies in choosing a hierarchy that is compact, stable, statistically honest, and useful without destroying that identity.

The most promising formulation is therefore:

> **HCD is a constrained pattern-discovery framework over an exact signed measure of KPI change.**

This formulation connects demographic decomposition, cooperative-game allocation, pattern mining, tree regularization, hierarchical modeling, and causal experimentation while keeping their claims separate. It matches the actual Product Analytics workflow: detect a change, conserve the metric, localize it, regularize the story, replicate it, and only then test an intervention.

---

## References and closest prior art

- Kitagawa, E. M. (1955). Components of a difference between two rates. *Journal of the American Statistical Association*, 50, 1168–1194. https://doi.org/10.1080/01621459.1955.10501299
- Das Gupta, P. (1978). A general method of decomposing a difference between two rates into several components. *Demography*, 15, 99–112. https://doi.org/10.2307/2060493
- Kass, G. V. (1980). An exploratory technique for investigating large quantities of categorical data. *Applied Statistics*, 29, 119–127. https://doi.org/10.2307/2986296
- Breiman, L., Friedman, J. H., Olshen, R. A., & Stone, C. J. (1984). *Classification and Regression Trees*. Wadsworth.
- Shorrocks, A. F. (2013). Decomposition procedures for distributional analysis: a unified framework based on the Shapley value. *Journal of Economic Inequality*, 11, 99–126. https://doi.org/10.1007/s10888-011-9214-z
- Friedman, J. H., & Popescu, B. E. (2008). Predictive learning via rule ensembles. *Annals of Applied Statistics*, 2, 916–954. https://doi.org/10.1214/07-AOAS148
- Herrera, F., Carmona, C. J., González, P., & del Jesus, M. J. (2011). An overview on Subgroup Discovery. *Knowledge and Information Systems*, 29, 495–525. https://doi.org/10.1007/s10115-010-0356-2
- Athey, S., & Imbens, G. (2016). Recursive partitioning for heterogeneous causal effects. *Proceedings of the National Academy of Sciences*, 113, 7353–7360. https://doi.org/10.1073/pnas.1510489113
- Wager, S., & Athey, S. (2018). Estimation and inference of heterogeneous treatment effects using random forests. *Journal of the American Statistical Association*, 113, 1228–1242. https://doi.org/10.1080/01621459.2017.1319839
- Budhathoki, K., Janzing, D., Bloebaum, P., & Ng, H. Y. (2021). Why did the distribution change? *AISTATS 2021*. [PMLR paper](https://proceedings.mlr.press/v130/budhathoki21a.html)
- Zhang, H., Singh, H., Ghassemi, M., & Joshi, S. (2023). Why did the model fail? Attributing model performance changes to distribution shifts. *ICML 2023*. [PMLR paper](https://proceedings.mlr.press/v202/zhang23ai.html)
- Sundararajan, M., Dhamdhere, K., & Agarwal, A. (2020). The Shapley–Taylor Interaction Index. *ICML 2020*. [PMLR paper](https://proceedings.mlr.press/v119/sundararajan20a.html)
- Bewley, T., Amoukou, S. I., Albini, E., Mishra, S., & Veloso, M. (2026). ShapShift: Explaining model prediction shifts with subgroup conditional Shapley values. [arXiv:2604.11200](https://arxiv.org/abs/2604.11200)
- Zhou, C., Chen, D., Shen, Z., Jiang, W., Li, Y., & Di, P. (2026). Explaining the “Why”: A unified framework for the additive attribution of changes in arbitrary measures. [arXiv:2604.26266](https://arxiv.org/abs/2604.26266)
- Li, M., Li, Z., Yin, K., Nie, X., Zhang, W., Sui, K., & Pei, D. (2022). Causal inference-based root cause analysis for online service systems with intervention recognition. [arXiv:2206.05871](https://arxiv.org/abs/2206.05871)
