import unittest

import numpy as np
from fastapi import HTTPException

from main import (
    AnalyzeRequest,
    DistributionRequest,
    TestRequest,
    _describe,
    _histogram,
    _regression,
    _validate_dataset,
    analyze,
    distribution,
    statistical_test,
)


class AnalysisTests(unittest.TestCase):
    def test_descriptive_statistics_and_iqr_outliers(self):
        result = _describe(np.array([1, 2, 3, 4, 5, 100], dtype=float))
        self.assertEqual(result["count"], 6)
        self.assertEqual(result["median"], 3.5)
        self.assertEqual(result["outliers"], [100.0])
        self.assertAlmostEqual(result["variance"], np.var([1, 2, 3, 4, 5, 100], ddof=1))

    def test_histogram_counts_cover_all_observations(self):
        result = _histogram(np.array([1, 2, 3, 4, 5], dtype=float))
        self.assertEqual(sum(item["count"] for item in result["bins"]), 5)

    def test_multiple_regression_recovers_coefficients(self):
        x1 = np.array([0, 1, 2, 3, 4], dtype=float)
        x2 = np.array([3, 1, 4, 2, 5], dtype=float)
        y = 2 + 3 * x1 - 0.5 * x2
        result = _regression({"x1": x1, "x2": x2, "y": y}, "y", ["x1", "x2"])
        self.assertAlmostEqual(result["intercept"], 2)
        self.assertAlmostEqual(result["coefficients"]["x1"], 3)
        self.assertAlmostEqual(result["coefficients"]["x2"], -0.5)
        self.assertAlmostEqual(result["r_squared"], 1)

    def test_dataset_rejects_unequal_column_lengths(self):
        with self.assertRaises(HTTPException):
            _validate_dataset({"x": [1, 2], "y": [1]})

    def test_analysis_returns_statistics_and_correlations(self):
        result = analyze(AnalyzeRequest(
            data={"x": [1, 2, 3, 4], "y": [3, 5, 7, 9]},
            target="y",
            predictors=["x"],
        ))
        self.assertEqual(result["rows"], 4)
        self.assertAlmostEqual(result["correlations"]["x"]["y"], 1)
        self.assertAlmostEqual(result["spearman_correlations"]["x"]["y"], 1)
        self.assertAlmostEqual(result["regression"]["coefficients"]["x"], 2)

    def test_normal_distribution_returns_density_and_cdf(self):
        result = distribution(DistributionRequest(type="normal", mean=0, std=1))
        center = min(result["points"], key=lambda point: abs(point["x"]))
        self.assertAlmostEqual(center["cdf"], 0.5, places=5)
        self.assertGreater(center["density"], 0.39)

    def test_statistical_tests_return_p_values(self):
        t_result = statistical_test(TestRequest(
            type="ttest",
            values_a=[10, 11, 12, 13],
            values_b=[1, 2, 3, 4],
        ))
        chi_result = statistical_test(TestRequest(
            type="chi_square",
            values_a=[10, 10, 10],
            expected=[10, 10, 10],
        ))
        self.assertLess(t_result["p_value"], 0.05)
        self.assertAlmostEqual(chi_result["statistic"], 0)
        self.assertAlmostEqual(chi_result["p_value"], 1)


if __name__ == "__main__":
    unittest.main()
