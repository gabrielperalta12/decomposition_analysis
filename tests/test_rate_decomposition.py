from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from scripts.rate_decomposition import (
    DecompositionError,
    all_orders_decomposition,
    chevan_categorical_report,
    chevan_sutherland_two_variable,
    hierarchical_owen_decomposition,
    kitagawa_two_period,
    multiperiod_kitagawa,
    stepwise_decomposition,
)


def revenue(values):
    return values["traffic"] * values["cvr"] * values["aov"]


class KitagawaTests(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame(
            {
                "segment": ["New", "Returning", "Enterprise"],
                "w0": [.50, .35, .15],
                "w1": [.42, .38, .20],
                "r0": [.08, .18, .31],
                "r1": [.10, .17, .34],
            }
        )

    def test_notebook_example_is_exact(self):
        result = kitagawa_two_period(self.data)
        self.assertAlmostEqual(result.summary["observed_change"], .0251)
        self.assertAlmostEqual(result.summary["mix"], .0143)
        self.assertAlmostEqual(result.summary["rate"], .0108)
        self.assertAlmostEqual(result.summary["identity_error"], 0.0)

    def test_chevan_categories_add(self):
        result = kitagawa_two_period(self.data)
        report = chevan_categorical_report(result)
        self.assertAlmostEqual(
            report.loc["variable-level total", "total_contribution"],
            result.summary["observed_change"],
        )

    def test_chevan_sutherland_two_variable_is_exact(self):
        p = [np.array([.55, .45]), np.array([.48, .52])]
        q = [
            np.array([[.70, .30], [.60, .40]]),
            np.array([[.76, .24], [.64, .36]]),
        ]
        rates = [
            np.array([[.045, .080], [.035, .065]]),
            np.array([[.041, .086], [.040, .070]]),
        ]
        result = chevan_sutherland_two_variable(
            p[0][:, None] * q[0],
            p[1][:, None] * q[1],
            rates[0],
            rates[1],
            row_labels=["Paid Search", "Organic"],
            column_labels=["Mobile", "Desktop"],
        )
        self.assertAlmostEqual(result.summary["identity_error"], 0.0)
        self.assertAlmostEqual(
            result.categories["total"].sum(), result.summary["observed_change"]
        )
        self.assertEqual(set(result.categories["variable_family"]), {"I", "J"})

    def test_chevan_sutherland_rejects_structural_zero(self):
        shares = np.array([[.5, .0], [.2, .3]])
        rates = np.full((2, 2), .1)
        with self.assertRaises(DecompositionError):
            chevan_sutherland_two_variable(shares, shares, rates, rates)

    def test_invalid_shares_are_rejected(self):
        bad = self.data.copy()
        bad.loc[0, "w0"] = .7
        with self.assertRaises(DecompositionError):
            kitagawa_two_period(bad)

    def test_entrant_is_separate(self):
        data = pd.DataFrame(
            {
                "segment": ["Core", "Entrant"],
                "w0": [1.0, 0.0],
                "w1": [.92, .08],
                "r0": [.10, np.nan],
                "r1": [.10, .24],
            }
        )
        result = kitagawa_two_period(data, missing_rate_policy="separate")
        entrant = result.detail.set_index("segment").loc["Entrant"]
        self.assertTrue(np.isnan(entrant["mix"]))
        self.assertAlmostEqual(entrant["entry_exit"], .0192)
        self.assertAlmostEqual(result.summary["identity_error"], 0.0)

    def test_entrant_reference_changes_labels_not_total(self):
        data = pd.DataFrame(
            {
                "segment": ["Core", "Entrant"],
                "w0": [1.0, 0.0],
                "w1": [.92, .08],
                "r0": [.10, np.nan],
                "r1": [.10, .24],
            }
        )
        low = kitagawa_two_period(
            data, missing_rate_policy="reference", reference_rate=0.0
        )
        high = kitagawa_two_period(
            data, missing_rate_policy="reference", reference_rate=.24
        )
        self.assertAlmostEqual(low.summary["observed_change"], high.summary["observed_change"])
        self.assertNotAlmostEqual(low.summary["mix"], high.summary["mix"])

    def test_exiting_segment_is_separate(self):
        data = pd.DataFrame(
            {
                "segment": ["Core", "Exited"],
                "w0": [.90, .10],
                "w1": [1.0, 0.0],
                "r0": [.10, .30],
                "r1": [.10, np.nan],
            }
        )
        result = kitagawa_two_period(data, missing_rate_policy="separate")
        exited = result.detail.set_index("segment").loc["Exited"]
        self.assertAlmostEqual(exited["entry_exit"], -.03)
        self.assertAlmostEqual(result.summary["identity_error"], 0.0)


class PathTests(unittest.TestCase):
    def setUp(self):
        self.base = {"traffic": 1_000., "cvr": .04, "aov": 50.}
        self.final = {"traffic": 1_200., "cvr": .05, "aov": 48.}

    def test_stepwise_is_exact_and_order_dependent(self):
        first = stepwise_decomposition(
            self.base, self.final, revenue, ["traffic", "cvr", "aov"]
        )
        second = stepwise_decomposition(
            self.base, self.final, revenue, ["aov", "cvr", "traffic"]
        )
        self.assertAlmostEqual(first.observed_change, 880.0)
        self.assertAlmostEqual(first.identity_error, 0.0)
        self.assertFalse(first.contributions.equals(second.contributions))

    def test_all_orders_matches_notebook(self):
        result = all_orders_decomposition(self.base, self.final, revenue)
        self.assertEqual(result.paths_evaluated, 6)
        self.assertTrue(result.exact_enumeration)
        self.assertAlmostEqual(result.contributions["traffic"], 440.6666666667)
        self.assertAlmostEqual(result.contributions.sum(), 880.0)

    def test_sampled_orders_remain_efficient(self):
        result = all_orders_decomposition(
            self.base,
            self.final,
            revenue,
            n_permutations=25,
            random_state=42,
        )
        self.assertFalse(result.exact_enumeration)
        self.assertEqual(result.paths_evaluated, 25)
        self.assertAlmostEqual(result.contributions.sum(), 880.0)

    def test_owen_hierarchy_is_exact(self):
        result = hierarchical_owen_decomposition(
            self.base,
            self.final,
            revenue,
            groups={"acquisition": ["traffic"], "economics": ["cvr", "aov"]},
        )
        self.assertAlmostEqual(result.contributions.sum(), 880.0)
        self.assertEqual(result.paths_evaluated, 4)


class MultiperiodTests(unittest.TestCase):
    def test_direct_and_chained_totals_telescope(self):
        panel = pd.DataFrame(
            {
                "period": np.repeat([0, 1, 2, 3], 3),
                "segment": ["New", "Returning", "Enterprise"] * 4,
                "weight": [.50,.35,.15, .44,.38,.18, .39,.40,.21, .42,.36,.22],
                "rate": [.08,.18,.31, .10,.17,.33, .09,.20,.35, .11,.19,.36],
            }
        )
        result = multiperiod_kitagawa(panel)
        self.assertAlmostEqual(
            result.direct["observed_change"], result.chained["observed_change"]
        )
        self.assertNotAlmostEqual(result.direct["mix"], result.chained["mix"])
        self.assertAlmostEqual(
            result.comparison.loc["observed_change", "chained_minus_direct"], 0.0
        )


if __name__ == "__main__":
    unittest.main()
