"""Production methods for the Shapley reading route in Notebook 04.

The module separates the allocation operator from the coalition value. None of
the functions is causal merely because it uses counterfactual vocabulary. A
causal interpretation of ``counterfactual_shapley`` requires the supplied
oracle to identify valid structural counterfactuals.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations, permutations
from math import factorial
from typing import Any

import numpy as np
import pandas as pd

Number = int | float | np.number
Coalition = frozenset[Hashable]


class ShapleyDecompositionError(ValueError):
    """Raised when inputs do not define a valid decomposition."""


@dataclass(frozen=True)
class ShapleyResult:
    """Generic Shapley allocation and its conservation diagnostics."""

    contributions: pd.Series
    baseline_value: float
    full_value: float
    observed_change: float
    allocated_change: float
    identity_error: float
    coalitions_evaluated: int

    def assert_exact(self, atol: float = 1e-12) -> None:
        if not np.isclose(self.identity_error, 0.0, atol=atol, rtol=0.0):
            raise ShapleyDecompositionError(
                f"Allocation is not exact: error={self.identity_error!r}"
            )


@dataclass(frozen=True)
class BiewenResult:
    """Pure main and interaction effects from Möbius inversion."""

    interactions: pd.DataFrame
    shapley_from_interactions: pd.Series
    observed_change: float
    identity_error: float

    def assert_exact(self, atol: float = 1e-12) -> None:
        if not np.isclose(self.identity_error, 0.0, atol=atol, rtol=0.0):
            raise ShapleyDecompositionError(
                f"Biewen interactions are not exact: error={self.identity_error!r}"
            )


@dataclass(frozen=True)
class ZhaoAttributionResult:
    """Channel and ordered-touchpoint attribution for observed journeys."""

    channels: pd.Series
    touchpoints: pd.DataFrame
    total_value: float
    identity_error: float

    def assert_exact(self, atol: float = 1e-12) -> None:
        if not np.isclose(self.identity_error, 0.0, atol=atol, rtol=0.0):
            raise ShapleyDecompositionError(
                f"Journey attribution is not exact: error={self.identity_error!r}"
            )


def _players(players: Sequence[Hashable]) -> tuple[Hashable, ...]:
    output = tuple(players)
    if not output or len(set(output)) != len(output):
        raise ShapleyDecompositionError("players must be non-empty and unique")
    return output


def _all_coalitions(players: Sequence[Hashable]) -> list[Coalition]:
    return [
        frozenset(coalition)
        for size in range(len(players) + 1)
        for coalition in combinations(players, size)
    ]


def shapley_value(
    players: Sequence[Hashable],
    value_function: Callable[[Coalition], Number],
    *,
    atol: float = 1e-12,
) -> ShapleyResult:
    """Calculate an exact Shapley allocation by caching all coalition values."""
    names = _players(players)
    coalitions = _all_coalitions(names)
    values: dict[Coalition, float] = {}
    for coalition in coalitions:
        value = float(value_function(coalition))
        if not np.isfinite(value):
            raise ShapleyDecompositionError(
                f"Non-finite value returned for coalition {set(coalition)!r}"
            )
        values[coalition] = value

    p = len(names)
    contributions = pd.Series(0.0, index=pd.Index(names, dtype=object), dtype=float)
    for player in names:
        for coalition in coalitions:
            if player in coalition:
                continue
            size = len(coalition)
            weight = factorial(size) * factorial(p - size - 1) / factorial(p)
            contributions[player] += weight * (
                values[coalition | {player}] - values[coalition]
            )

    baseline, full = values[frozenset()], values[frozenset(names)]
    observed, allocated = full - baseline, float(contributions.sum())
    result = ShapleyResult(
        contributions=contributions,
        baseline_value=baseline,
        full_value=full,
        observed_change=observed,
        allocated_change=allocated,
        identity_error=allocated - observed,
        coalitions_evaluated=len(values),
    )
    result.assert_exact(atol)
    return result


def shorrocks_decomposition(
    base_values: Mapping[Hashable, Number],
    comparison_values: Mapping[Hashable, Number],
    indicator: Callable[[Mapping[Hashable, float]], Number],
) -> ShapleyResult:
    """Shorrocks all-orders decomposition for declared neutral and active values."""
    if set(base_values) != set(comparison_values):
        raise ShapleyDecompositionError("base and comparison factors must match")
    names = tuple(base_values)
    base = {key: float(value) for key, value in base_values.items()}
    comparison = {key: float(value) for key, value in comparison_values.items()}

    def coalition_value(coalition: Coalition) -> float:
        state = {
            key: comparison[key] if key in coalition else base[key]
            for key in names
        }
        return float(indicator(state))

    return shapley_value(names, coalition_value)


def biewen_interactions(
    base_values: Mapping[Hashable, Number],
    comparison_values: Mapping[Hashable, Number],
    outcome: Callable[[Mapping[Hashable, float]], Number],
    *,
    atol: float = 1e-12,
) -> BiewenResult:
    """Report every pure ceteris-paribus and interaction contribution."""
    if set(base_values) != set(comparison_values):
        raise ShapleyDecompositionError("base and comparison factors must match")
    names = _players(tuple(base_values))
    base = {key: float(value) for key, value in base_values.items()}
    comparison = {key: float(value) for key, value in comparison_values.items()}
    coalitions = _all_coalitions(names)

    def evaluate(coalition: Coalition) -> float:
        state = {
            key: comparison[key] if key in coalition else base[key]
            for key in names
        }
        value = float(outcome(state))
        if not np.isfinite(value):
            raise ShapleyDecompositionError("outcome returned a non-finite value")
        return value

    values = {coalition: evaluate(coalition) for coalition in coalitions}
    effects: dict[Coalition, float] = {}
    for target in coalitions[1:]:
        effects[target] = sum(
            (-1) ** (len(target) - len(subset)) * values[subset]
            for subset in coalitions
            if subset.issubset(target)
        )

    table = pd.DataFrame(
        [
            {
                "order": len(coalition),
                "factors": tuple(player for player in names if player in coalition),
                "effect": effect,
            }
            for coalition, effect in effects.items()
        ]
    ).sort_values("order", kind="stable", ignore_index=True)
    allocated = float(sum(effects.values()))
    observed = values[frozenset(names)] - values[frozenset()]
    shapley = pd.Series(
        {
            player: sum(
                effect / len(coalition)
                for coalition, effect in effects.items()
                if player in coalition
            )
            for player in names
        },
        dtype=float,
    )
    result = BiewenResult(table, shapley, observed, allocated - observed)
    result.assert_exact(atol)
    return result


def israeli_r2_shapley(
    features: pd.DataFrame | np.ndarray,
    outcome: pd.Series | np.ndarray,
    *,
    feature_names: Sequence[Hashable] | None = None,
    fit_intercept: bool = True,
) -> ShapleyResult:
    """Allocate linear-regression R-squared across predictors as in Israeli (2007)."""
    if isinstance(features, pd.DataFrame):
        x = features.to_numpy(dtype=float)
        names = tuple(features.columns) if feature_names is None else tuple(feature_names)
    else:
        x = np.asarray(features, dtype=float)
        names = tuple(range(x.shape[1])) if feature_names is None else tuple(feature_names)
    y = np.asarray(outcome, dtype=float).reshape(-1)
    if x.ndim != 2 or x.shape[0] != y.size:
        raise ShapleyDecompositionError("features and outcome have incompatible shapes")
    if len(names) != x.shape[1] or len(set(names)) != len(names):
        raise ShapleyDecompositionError("feature names must be unique and match columns")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ShapleyDecompositionError("features and outcome must be finite")
    total_sum_squares = float(np.sum((y - y.mean()) ** 2))
    if total_sum_squares <= 0:
        raise ShapleyDecompositionError("R-squared is undefined for a constant outcome")
    column = {name: position for position, name in enumerate(names)}

    def r_squared(coalition: Coalition) -> float:
        if not coalition:
            return 0.0
        design = x[:, [column[name] for name in names if name in coalition]]
        if fit_intercept:
            design = np.column_stack([np.ones(len(y)), design])
        coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
        residual_sum_squares = float(np.sum((y - design @ coefficients) ** 2))
        return 1.0 - residual_sum_squares / total_sum_squares

    return shapley_value(names, r_squared)


def zhao_journey_attribution(
    journeys: Iterable[Sequence[Hashable]],
    values: Iterable[Number],
    *,
    atol: float = 1e-12,
) -> ZhaoAttributionResult:
    """Attribute observed journey value to channels and ordered touchpoints.

    Each journey's value is divided uniformly over its unique channels, which
    is Zhao et al.'s simplified Shapley result for their user-type game. If a
    channel repeats, its channel credit is divided uniformly over its observed
    touchpoint occurrences.
    """
    journey_list = [tuple(journey) for journey in journeys]
    value_array = np.asarray(list(values), dtype=float)
    if len(journey_list) != len(value_array):
        raise ShapleyDecompositionError("journeys and values must have equal length")
    if not np.isfinite(value_array).all():
        raise ShapleyDecompositionError("journey values must be finite")
    if any(not journey for journey in journey_list):
        raise ShapleyDecompositionError("journeys cannot be empty")

    channel_credit: dict[Hashable, float] = {}
    touchpoint_credit: dict[tuple[Hashable, int], float] = {}
    for journey, value in zip(journey_list, value_array):
        unique_channels = tuple(dict.fromkeys(journey))
        per_channel = float(value) / len(unique_channels)
        for channel in unique_channels:
            channel_credit[channel] = channel_credit.get(channel, 0.0) + per_channel
            positions = [i + 1 for i, observed in enumerate(journey) if observed == channel]
            for position in positions:
                key = (channel, position)
                touchpoint_credit[key] = (
                    touchpoint_credit.get(key, 0.0) + per_channel / len(positions)
                )

    channels = pd.Series(channel_credit, dtype=float, name="attribution")
    touchpoints = pd.DataFrame(
        [
            {"channel": channel, "position": position, "attribution": credit}
            for (channel, position), credit in touchpoint_credit.items()
        ]
    ).sort_values("position", kind="stable", ignore_index=True)
    total = float(value_array.sum())
    error = float(channels.sum() - total)
    result = ZhaoAttributionResult(channels, touchpoints, total, error)
    result.assert_exact(atol)
    if not np.isclose(touchpoints["attribution"].sum(), total, atol=atol, rtol=0.0):
        raise ShapleyDecompositionError("Ordered touchpoints do not conserve value")
    return result


def counterfactual_shapley(
    variables: Sequence[Hashable],
    counterfactual_outcome: Callable[[Coalition], Number],
) -> ShapleyResult:
    """Allocate an observation-specific change using a counterfactual oracle.

    ``counterfactual_outcome(active)`` must return the same unit/timestamp's
    outcome when variables in ``active`` take observed values and all remaining
    variables take declared reference values. The empty coalition is therefore
    the all-reference counterfactual and the full coalition is the observed
    structural outcome. This wrapper supplies no identification by itself.
    """
    return shapley_value(variables, counterfactual_outcome)


__all__ = [
    "BiewenResult",
    "ShapleyDecompositionError",
    "ShapleyResult",
    "ZhaoAttributionResult",
    "biewen_interactions",
    "counterfactual_shapley",
    "israeli_r2_shapley",
    "shapley_value",
    "shorrocks_decomposition",
    "zhao_journey_attribution",
]
