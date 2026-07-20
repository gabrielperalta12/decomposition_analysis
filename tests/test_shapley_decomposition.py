from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from scripts.shapley_decomposition import (
    ShapleyDecompositionError,
    biewen_interactions,
    counterfactual_shapley,
    israeli_r2_shapley,
    shapley_value,
    shorrocks_decomposition,
    zhao_journey_attribution,
)


BASE = {"traffic": 1_000.0, "cvr": 0.04, "price": 50.0}
FINAL = {"traffic": 1_200.0, "cvr": 0.05, "price": 48.0}


def revenue(values):
    return values["traffic"] * values["cvr"] * values["price"]


class ShapleyRouteTests(unittest.TestCase):
    def test_generic_and_shorrocks_are_exact(self):
        players = tuple(BASE)

        def value(coalition):
            state = {key: FINAL[key] if key in coalition else BASE[key] for key in players}
            return revenue(state)

        generic = shapley_value(players, value)
        shorrocks = shorrocks_decomposition(BASE, FINAL, revenue)
        self.assertAlmostEqual(generic.observed_change, 880.0)
        np.testing.assert_allclose(generic.contributions, shorrocks.contributions)

    def test_biewen_interactions_reconstruct_shapley(self):
        interactions = biewen_interactions(BASE, FINAL, revenue)
        shapley = shorrocks_decomposition(BASE, FINAL, revenue)
        self.assertAlmostEqual(interactions.identity_error, 0.0)
        np.testing.assert_allclose(
            interactions.shapley_from_interactions.loc[shapley.contributions.index],
            shapley.contributions,
        )
        self.assertEqual(set(interactions.interactions["order"]), {1, 2, 3})

    def test_israeli_r2_allocates_full_r_squared(self):
        rng = np.random.default_rng(7)
        x1 = rng.normal(size=300)
        x2 = 0.7 * x1 + rng.normal(scale=0.5, size=300)
        y = 2 * x1 - x2 + rng.normal(scale=0.4, size=300)
        result = israeli_r2_shapley(pd.DataFrame({"x1": x1, "x2": x2}), y)
        self.assertAlmostEqual(result.allocated_change, result.full_value)
        self.assertGreater(result.contributions.min(), 0.0)

    def test_israeli_rejects_constant_outcome(self):
        with self.assertRaises(ShapleyDecompositionError):
            israeli_r2_shapley(np.ones((10, 2)), np.ones(10))

    def test_zhao_channel_and_ordered_outputs_conserve_value(self):
        result = zhao_journey_attribution(
            [["Search", "Email"], ["Social"], ["Search", "Search", "Email"]],
            [100.0, 40.0, 60.0],
        )
        self.assertAlmostEqual(result.channels.sum(), 200.0)
        self.assertAlmostEqual(result.touchpoints["attribution"].sum(), 200.0)
        self.assertAlmostEqual(result.channels["Search"], 80.0)

    def test_counterfactual_shapley_uses_declared_oracle(self):
        observed = {"demand": 12.0, "supply": 8.0}
        reference = {"demand": 10.0, "supply": 5.0}

        def structural_counterfactual(active):
            state = {
                key: observed[key] if key in active else reference[key]
                for key in observed
            }
            return state["demand"] * state["supply"]

        result = counterfactual_shapley(tuple(observed), structural_counterfactual)
        self.assertAlmostEqual(result.observed_change, 46.0)
        self.assertAlmostEqual(result.allocated_change, 46.0)


if __name__ == "__main__":
    unittest.main()
