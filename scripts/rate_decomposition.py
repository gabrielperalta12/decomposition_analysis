"""Production-oriented rate decomposition methods from Notebook 01.

The functions in this module allocate observed changes. They do not identify
causal effects. Inputs are validated aggressively because algebraic exactness
does not protect an analysis from malformed shares, missing rates, or an
ill-defined entry/exit convention.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass
from itertools import permutations, product
from math import factorial
from typing import Any, Literal

import numpy as np
import pandas as pd

Number = int | float | np.number
RatePolicy = Literal["error", "separate", "reference"]


class DecompositionError(ValueError):
    """Raised when inputs do not define a valid decomposition."""


@dataclass(frozen=True)
class KitagawaResult:
    """Detailed and aggregate output from a two-period Kitagawa decomposition."""

    detail: pd.DataFrame
    summary: pd.Series

    def assert_exact(self, atol: float = 1e-12) -> None:
        """Raise when allocated and observed changes do not agree."""
        if not np.isclose(
            self.summary["allocated_change"],
            self.summary["observed_change"],
            atol=atol,
            rtol=0.0,
        ):
            raise DecompositionError(
                f"Decomposition is not exact: error={self.summary['identity_error']!r}"
            )


@dataclass(frozen=True)
class PathDecompositionResult:
    """Output from stepwise, all-orders, or hierarchical path allocation."""

    contributions: pd.Series
    observed_change: float
    identity_error: float
    paths_evaluated: int
    exact_enumeration: bool
    path_details: pd.DataFrame | None = None

    def assert_exact(self, atol: float = 1e-12) -> None:
        """Raise when factor contributions do not reproduce the total change."""
        if not np.isclose(self.identity_error, 0.0, atol=atol, rtol=0.0):
            raise DecompositionError(
                f"Path allocation is not exact: error={self.identity_error!r}"
            )


@dataclass(frozen=True)
class MultiperiodResult:
    """Direct and adjacent-link Kitagawa outputs for several periods."""

    links: pd.DataFrame
    direct: pd.Series
    chained: pd.Series
    comparison: pd.DataFrame


@dataclass(frozen=True)
class ChevanSutherlandResult:
    """Exact two-variable categorical refinement of a Das Gupta rate decomposition."""

    categories: pd.DataFrame
    parents: pd.Series
    summary: pd.Series

    def assert_exact(self, atol: float = 1e-12) -> None:
        """Raise when category and parent allocations do not reproduce the change."""
        if not np.isclose(self.summary["identity_error"], 0.0, atol=atol, rtol=0.0):
            raise DecompositionError(
                f"Chevan-Sutherland decomposition is not exact: "
                f"error={self.summary['identity_error']!r}"
            )


def _require_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise DecompositionError(f"Missing required columns: {missing}")


def _numeric_column(frame: pd.DataFrame, column: str) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    bad = values.isna() & frame[column].notna()
    if bad.any():
        raise DecompositionError(f"Column {column!r} contains non-numeric values")
    return values.astype(float)


def _weighted_total(weights: pd.Series, rates: pd.Series) -> float:
    """Calculate sum(weight * rate), treating zero-weight missing rates as zero."""
    invalid = (weights > 0) & rates.isna()
    if invalid.any():
        raise DecompositionError("A positive-weight segment has a missing rate")
    products = np.where(weights.to_numpy() == 0, 0.0, weights * rates)
    return float(np.sum(products))


def _resolve_reference(
    reference_rate: float | Mapping[Hashable, float] | None,
    segment: Hashable,
) -> float:
    if reference_rate is None:
        raise DecompositionError(
            "missing_rate_policy='reference' requires reference_rate"
        )
    value = (
        reference_rate[segment]
        if isinstance(reference_rate, Mapping)
        else reference_rate
    )
    value = float(value)
    if not np.isfinite(value):
        raise DecompositionError(f"Reference rate for {segment!r} is not finite")
    return value


def kitagawa_two_period(
    data: pd.DataFrame,
    *,
    segment_col: str = "segment",
    weight0_col: str = "w0",
    weight1_col: str = "w1",
    rate0_col: str = "r0",
    rate1_col: str = "r1",
    missing_rate_policy: RatePolicy = "error",
    reference_rate: float | Mapping[Hashable, float] | None = None,
    validate_shares: bool = True,
    share_atol: float = 1e-10,
) -> KitagawaResult:
    """Decompose a two-period aggregate-rate change into mix and rate effects.

    Parameters
    ----------
    data:
        One row per segment in the union of both periods.
    missing_rate_policy:
        ``"error"`` rejects missing rates. ``"separate"`` records the entire
        observed contribution of entrant/exit segments in ``entry_exit`` and
        leaves their mix/rate fields undefined. ``"reference"`` fills a rate
        missing at zero weight using a declared scalar or segment mapping.
    reference_rate:
        Used only by the ``"reference"`` policy. A scalar applies to every
        missing zero-weight rate; a mapping can vary by segment.

    Returns
    -------
    KitagawaResult
        Segment-level effects and an aggregate summary. ``observed_change`` is
        always calculated from the original observed data.
    """
    required = [segment_col, weight0_col, weight1_col, rate0_col, rate1_col]
    _require_columns(data, required)
    if data[segment_col].isna().any() or data[segment_col].duplicated().any():
        raise DecompositionError("Segment identifiers must be non-missing and unique")
    if missing_rate_policy not in {"error", "separate", "reference"}:
        raise DecompositionError(f"Unknown missing_rate_policy={missing_rate_policy!r}")

    work = data[required].copy()
    work.columns = ["segment", "w0", "w1", "r0", "r1"]
    for column in ["w0", "w1", "r0", "r1"]:
        work[column] = _numeric_column(work, column)
    if (~np.isfinite(work[["w0", "w1"]])).any().any():
        raise DecompositionError("Weights must be finite")
    if (work[["w0", "w1"]] < 0).any().any():
        raise DecompositionError("Weights cannot be negative")
    if validate_shares:
        for column in ["w0", "w1"]:
            total = float(work[column].sum())
            if not np.isclose(total, 1.0, atol=share_atol, rtol=0.0):
                raise DecompositionError(
                    f"{column} must sum to one; observed {total:.12g}"
                )

    for period in ["0", "1"]:
        weight = work[f"w{period}"]
        rate = work[f"r{period}"]
        if ((weight > 0) & rate.isna()).any():
            segments = work.loc[(weight > 0) & rate.isna(), "segment"].tolist()
            raise DecompositionError(
                f"Positive-weight segments have missing r{period}: {segments}"
            )
        if ((weight == 0) & rate.isna()).any() and missing_rate_policy == "error":
            segments = work.loc[(weight == 0) & rate.isna(), "segment"].tolist()
            raise DecompositionError(
                f"Zero-weight segments have unidentified r{period}: {segments}; "
                "choose 'separate' or 'reference' explicitly"
            )

    observed_r0 = _weighted_total(work.w0, work.r0)
    observed_r1 = _weighted_total(work.w1, work.r1)
    work["entry_exit"] = 0.0

    missing_mask = work[["r0", "r1"]].isna().any(axis=1)
    if missing_rate_policy == "reference":
        for index in work.index[missing_mask]:
            segment = work.at[index, "segment"]
            value = _resolve_reference(reference_rate, segment)
            if pd.isna(work.at[index, "r0"]):
                work.at[index, "r0"] = value
            if pd.isna(work.at[index, "r1"]):
                work.at[index, "r1"] = value
    elif missing_rate_policy == "separate":
        for index in work.index[missing_mask]:
            contribution = (
                (0.0 if work.at[index, "w1"] == 0 else work.at[index, "w1"] * work.at[index, "r1"])
                - (0.0 if work.at[index, "w0"] == 0 else work.at[index, "w0"] * work.at[index, "r0"])
            )
            work.at[index, "entry_exit"] = contribution

    stable_mask = ~work[["r0", "r1"]].isna().any(axis=1)
    work["mix"] = np.nan
    work["rate"] = np.nan
    work.loc[stable_mask, "mix"] = (
        (work.loc[stable_mask, "w1"] - work.loc[stable_mask, "w0"])
        * (work.loc[stable_mask, "r1"] + work.loc[stable_mask, "r0"])
        / 2
    )
    work.loc[stable_mask, "rate"] = (
        (work.loc[stable_mask, "r1"] - work.loc[stable_mask, "r0"])
        * (work.loc[stable_mask, "w1"] + work.loc[stable_mask, "w0"])
        / 2
    )
    work["total_contribution"] = (
        work[["mix", "rate"]].sum(axis=1, min_count=1).fillna(0.0)
        + work["entry_exit"]
    )

    mix_total = float(work["mix"].sum(skipna=True))
    rate_total = float(work["rate"].sum(skipna=True))
    entry_exit_total = float(work["entry_exit"].sum())
    allocated = mix_total + rate_total + entry_exit_total
    observed_change = observed_r1 - observed_r0
    summary = pd.Series(
        {
            "R0": observed_r0,
            "R1": observed_r1,
            "observed_change": observed_change,
            "mix": mix_total,
            "rate": rate_total,
            "entry_exit": entry_exit_total,
            "allocated_change": allocated,
            "identity_error": allocated - observed_change,
        },
        dtype=float,
    )
    result = KitagawaResult(work, summary)
    result.assert_exact(max(share_atol, 1e-12))
    return result


def chevan_categorical_report(result: KitagawaResult) -> pd.DataFrame:
    """Return additive category effects for one compositional variable.

    This is the one-variable categorical refinement demonstrated in Notebook
    01. It must not be used to add separate one-way reports from several
    cross-classified variables; doing so can double count interactions.
    """
    columns = ["segment", "mix", "rate", "entry_exit", "total_contribution"]
    report = result.detail[columns].set_index("segment").copy()
    report.loc["variable-level total"] = report.sum(numeric_only=True)
    report.loc["observed aggregate change", "total_contribution"] = result.summary[
        "observed_change"
    ]
    return report


def chevan_sutherland_two_variable(
    joint_share0: np.ndarray,
    joint_share1: np.ndarray,
    rate0: np.ndarray,
    rate1: np.ndarray,
    *,
    row_labels: Sequence[Hashable] | None = None,
    column_labels: Sequence[Hashable] | None = None,
    atol: float = 1e-12,
) -> ChevanSutherlandResult:
    """Refine a two-variable Das Gupta decomposition into category effects.

    The four arrays must have the same ``(I, J)`` shape. Joint shares must be
    strictly positive and sum to one in each period. Strict positivity is not
    cosmetic: the original symmetric composition coefficients contain ratios
    of cell and marginal shares, so structural zeros require a separately
    declared pooling, smoothing, or limiting convention.

    Returns category-level composition, rate, and total contributions for both
    variable families. Following Chevan and Sutherland (2009), the overall
    rate effect is divided by ``V=2`` before it is reported by each family.
    """
    w0, w1, r0, r1 = (
        np.asarray(value, dtype=float)
        for value in (joint_share0, joint_share1, rate0, rate1)
    )
    if w0.ndim != 2 or any(value.shape != w0.shape for value in (w1, r0, r1)):
        raise DecompositionError("All inputs must be two-dimensional arrays of equal shape")
    if not all(np.isfinite(value).all() for value in (w0, w1, r0, r1)):
        raise DecompositionError("Shares and rates must be finite")
    if np.any(w0 <= 0) or np.any(w1 <= 0):
        raise DecompositionError(
            "Original Chevan-Sutherland coefficients require positive joint shares"
        )
    for name, weights in (("joint_share0", w0), ("joint_share1", w1)):
        if not np.isclose(weights.sum(), 1.0, atol=atol, rtol=0.0):
            raise DecompositionError(f"{name} must sum to one")

    n_i, n_j = w0.shape
    rows = list(range(n_i)) if row_labels is None else list(row_labels)
    columns = list(range(n_j)) if column_labels is None else list(column_labels)
    if len(rows) != n_i or len(set(rows)) != n_i:
        raise DecompositionError("row_labels must be unique and match the row count")
    if len(columns) != n_j or len(set(columns)) != n_j:
        raise DecompositionError("column_labels must be unique and match the column count")

    def coefficients(weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        row_marginal = weights.sum(axis=1)
        column_marginal = weights.sum(axis=0)
        coef_i = np.sqrt(weights * row_marginal[:, None] / column_marginal[None, :])
        coef_j = np.sqrt(weights * column_marginal[None, :] / row_marginal[:, None])
        if not np.allclose(coef_i * coef_j, weights, atol=atol, rtol=0.0):
            raise DecompositionError("Composition coefficients do not reconstruct shares")
        return coef_i, coef_j

    a, b = coefficients(w0)
    A, B = coefficients(w1)
    rate_bar, weight_bar = (r0 + r1) / 2, (w0 + w1) / 2
    a_bar, b_bar = (a + A) / 2, (b + B) / 2
    composition_i = (rate_bar * b_bar * (A - a)).sum(axis=1)
    composition_j = (rate_bar * a_bar * (B - b)).sum(axis=0)
    rate_i = (weight_bar * (r1 - r0) / 2).sum(axis=1)
    rate_j = (weight_bar * (r1 - r0) / 2).sum(axis=0)

    records = [
        ("I", label, comp, rate, comp + rate)
        for label, comp, rate in zip(rows, composition_i, rate_i)
    ] + [
        ("J", label, comp, rate, comp + rate)
        for label, comp, rate in zip(columns, composition_j, rate_j)
    ]
    categories = pd.DataFrame(
        records,
        columns=["variable_family", "category", "composition", "rate", "total"],
    )
    parents = pd.Series(
        {
            "I_composition": float(composition_i.sum()),
            "J_composition": float(composition_j.sum()),
            "overall_rate": float(rate_i.sum() + rate_j.sum()),
        }
    )
    baseline = float(np.sum(w0 * r0))
    comparison = float(np.sum(w1 * r1))
    allocated = float(categories["total"].sum())
    summary = pd.Series(
        {
            "baseline_rate": baseline,
            "comparison_rate": comparison,
            "observed_change": comparison - baseline,
            "allocated_change": allocated,
            "identity_error": allocated - (comparison - baseline),
        }
    )
    if not np.isclose(rate_i.sum(), rate_j.sum(), atol=atol, rtol=0.0):
        raise DecompositionError("Each variable family must receive half the rate effect")
    if not np.isclose(parents.sum(), comparison - baseline, atol=atol, rtol=0.0):
        raise DecompositionError("Parent effects do not reproduce the observed change")
    result = ChevanSutherlandResult(categories, parents, summary)
    result.assert_exact(atol)
    return result


def stepwise_decomposition(
    base_values: Mapping[Hashable, Number],
    final_values: Mapping[Hashable, Number],
    value_function: Callable[[Mapping[Hashable, float]], Number],
    order: Sequence[Hashable],
) -> PathDecompositionResult:
    """Replace factors once in a declared order and return exact increments."""
    factors = list(base_values)
    if set(factors) != set(final_values):
        raise DecompositionError("Base and final mappings must contain the same factors")
    if len(order) != len(factors) or set(order) != set(factors):
        raise DecompositionError("Order must contain every factor exactly once")
    state = {key: float(value) for key, value in base_values.items()}
    final = {key: float(value) for key, value in final_values.items()}
    rows: list[dict[str, Any]] = []
    for factor in order:
        before = float(value_function(state.copy()))
        state[factor] = final[factor]
        after = float(value_function(state.copy()))
        rows.append(
            {
                "factor": factor,
                "value_before": before,
                "value_after": after,
                "contribution": after - before,
            }
        )
    details = pd.DataFrame(rows)
    contributions = details.set_index("factor")["contribution"]
    observed = float(value_function(final.copy())) - float(
        value_function({key: float(value) for key, value in base_values.items()})
    )
    error = float(contributions.sum() - observed)
    result = PathDecompositionResult(
        contributions=contributions,
        observed_change=observed,
        identity_error=error,
        paths_evaluated=1,
        exact_enumeration=True,
        path_details=details,
    )
    result.assert_exact()
    return result


def all_orders_decomposition(
    base_values: Mapping[Hashable, Number],
    final_values: Mapping[Hashable, Number],
    value_function: Callable[[Mapping[Hashable, float]], Number],
    *,
    max_exact_paths: int = 100_000,
    n_permutations: int | None = None,
    random_state: int | np.random.Generator | None = None,
    retain_paths: bool = False,
) -> PathDecompositionResult:
    """Average marginal increments over all or sampled factor orders.

    Exact enumeration is used when ``factorial(K) <= max_exact_paths`` and
    ``n_permutations`` is omitted. For large problems, set ``n_permutations``;
    every sampled path is telescopically exact, while the allocation is a
    Monte Carlo approximation to the all-orders Shorrocks/Das Gupta value.
    """
    factors = list(base_values)
    if set(factors) != set(final_values):
        raise DecompositionError("Base and final mappings must contain the same factors")
    path_count = factorial(len(factors))
    exact = n_permutations is None and path_count <= max_exact_paths
    if n_permutations is None and not exact:
        raise DecompositionError(
            f"Exact enumeration requires {path_count:,} paths; set n_permutations"
        )
    if n_permutations is not None and n_permutations <= 0:
        raise DecompositionError("n_permutations must be positive")

    if exact:
        orders: Sequence[tuple[Hashable, ...]] = list(permutations(factors))
    else:
        rng = (
            random_state
            if isinstance(random_state, np.random.Generator)
            else np.random.default_rng(random_state)
        )
        orders = [tuple(rng.permutation(factors).tolist()) for _ in range(n_permutations or 0)]

    total = pd.Series(0.0, index=pd.Index(factors, dtype=object), dtype=float)
    path_frames: list[pd.DataFrame] = []
    for path_number, order in enumerate(orders):
        path = stepwise_decomposition(base_values, final_values, value_function, order)
        total = total.add(path.contributions, fill_value=0.0)
        if retain_paths and path.path_details is not None:
            detail = path.path_details.copy()
            detail.insert(0, "path", path_number)
            detail.insert(1, "order", " -> ".join(map(str, order)))
            path_frames.append(detail)
    contributions = total / len(orders)
    base_float = {key: float(value) for key, value in base_values.items()}
    final_float = {key: float(value) for key, value in final_values.items()}
    observed = float(value_function(final_float)) - float(value_function(base_float))
    error = float(contributions.sum() - observed)
    result = PathDecompositionResult(
        contributions=contributions,
        observed_change=observed,
        identity_error=error,
        paths_evaluated=len(orders),
        exact_enumeration=exact,
        path_details=pd.concat(path_frames, ignore_index=True) if path_frames else None,
    )
    result.assert_exact()
    return result


def das_gupta_decomposition(*args: Any, **kwargs: Any) -> PathDecompositionResult:
    """Alias for symmetric all-orders replacement in a Das Gupta-style model."""
    return all_orders_decomposition(*args, **kwargs)


def shorrocks_decomposition(*args: Any, **kwargs: Any) -> PathDecompositionResult:
    """Alias for a Shorrocks/Shapley all-orders indicator decomposition."""
    return all_orders_decomposition(*args, **kwargs)


def hierarchical_owen_decomposition(
    base_values: Mapping[Hashable, Number],
    final_values: Mapping[Hashable, Number],
    value_function: Callable[[Mapping[Hashable, float]], Number],
    groups: Mapping[Hashable, Sequence[Hashable]],
    *,
    max_exact_paths: int = 100_000,
) -> PathDecompositionResult:
    """Enumerate Owen-compatible paths for a declared a-priori hierarchy.

    Groups enter in every possible order, and members enter in every possible
    within-group order. Members of each group remain contiguous in each path.
    """
    factors = list(base_values)
    grouped = [factor for members in groups.values() for factor in members]
    if len(grouped) != len(set(grouped)) or set(grouped) != set(factors):
        raise DecompositionError("Groups must partition the factor set exactly once")
    if set(factors) != set(final_values):
        raise DecompositionError("Base and final mappings must contain the same factors")
    group_names = list(groups)
    path_count = factorial(len(group_names))
    for members in groups.values():
        path_count *= factorial(len(members))
    if path_count > max_exact_paths:
        raise DecompositionError(
            f"Hierarchical enumeration requires {path_count:,} paths; increase "
            "max_exact_paths only after reviewing computational cost"
        )

    orders: list[tuple[Hashable, ...]] = []
    within_options = [list(permutations(groups[name])) for name in group_names]
    for group_order in permutations(group_names):
        option_lookup = dict(zip(group_names, within_options, strict=True))
        for selected in product(*(option_lookup[name] for name in group_order)):
            orders.append(tuple(factor for block in selected for factor in block))

    total = pd.Series(0.0, index=pd.Index(factors, dtype=object), dtype=float)
    for order in orders:
        total = total.add(
            stepwise_decomposition(base_values, final_values, value_function, order).contributions,
            fill_value=0.0,
        )
    contributions = total / len(orders)
    observed = float(value_function(final_values)) - float(value_function(base_values))
    result = PathDecompositionResult(
        contributions=contributions,
        observed_change=observed,
        identity_error=float(contributions.sum() - observed),
        paths_evaluated=len(orders),
        exact_enumeration=True,
    )
    result.assert_exact()
    return result


def multiperiod_kitagawa(
    data: pd.DataFrame,
    *,
    period_col: str = "period",
    segment_col: str = "segment",
    weight_col: str = "weight",
    rate_col: str = "rate",
    missing_rate_policy: RatePolicy = "error",
    reference_rate: float | Mapping[Hashable, float] | None = None,
    validate_shares: bool = True,
) -> MultiperiodResult:
    """Compare direct endpoint and chained adjacent-period decompositions."""
    _require_columns(data, [period_col, segment_col, weight_col, rate_col])
    if data[[period_col, segment_col]].duplicated().any():
        raise DecompositionError("Each period/segment pair must be unique")
    periods = sorted(data[period_col].dropna().unique().tolist())
    if len(periods) < 2:
        raise DecompositionError("At least two periods are required")

    def pair(left_period: Any, right_period: Any) -> KitagawaResult:
        left = data.loc[data[period_col] == left_period, [segment_col, weight_col, rate_col]]
        right = data.loc[data[period_col] == right_period, [segment_col, weight_col, rate_col]]
        joined = left.merge(
            right,
            on=segment_col,
            how="outer",
            suffixes=("0", "1"),
        ).rename(
            columns={
                segment_col: "segment",
                f"{weight_col}0": "w0",
                f"{weight_col}1": "w1",
                f"{rate_col}0": "r0",
                f"{rate_col}1": "r1",
            }
        )
        joined[["w0", "w1"]] = joined[["w0", "w1"]].fillna(0.0)
        return kitagawa_two_period(
            joined,
            missing_rate_policy=missing_rate_policy,
            reference_rate=reference_rate,
            validate_shares=validate_shares,
        )

    link_rows: list[pd.Series] = []
    for left_period, right_period in zip(periods[:-1], periods[1:], strict=True):
        result = pair(left_period, right_period)
        row = result.summary[["mix", "rate", "entry_exit", "observed_change"]].rename(
            " -> ".join(map(str, [left_period, right_period]))
        )
        link_rows.append(row)
    links = pd.DataFrame(link_rows)
    chained = links.sum().rename("chained")
    direct_result = pair(periods[0], periods[-1])
    direct = direct_result.summary[
        ["mix", "rate", "entry_exit", "observed_change"]
    ].rename("direct")
    comparison = pd.concat([direct, chained], axis=1)
    comparison["chained_minus_direct"] = comparison["chained"] - comparison["direct"]
    if not np.isclose(
        direct["observed_change"], chained["observed_change"], atol=1e-12, rtol=0.0
    ):
        raise DecompositionError("Direct and chained observed changes do not telescope")
    return MultiperiodResult(links, direct, chained, comparison)


__all__ = [
    "ChevanSutherlandResult",
    "DecompositionError",
    "KitagawaResult",
    "MultiperiodResult",
    "PathDecompositionResult",
    "all_orders_decomposition",
    "chevan_categorical_report",
    "chevan_sutherland_two_variable",
    "das_gupta_decomposition",
    "hierarchical_owen_decomposition",
    "kitagawa_two_period",
    "multiperiod_kitagawa",
    "shorrocks_decomposition",
    "stepwise_decomposition",
]
