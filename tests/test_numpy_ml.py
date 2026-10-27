import unittest
import numpy as np
from f1tyre import numpy_ml as ml


class MetricTests(unittest.TestCase):
    def test_known_values(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        pred = np.array([1.0, 2.0, 3.0, 6.0])
        self.assertAlmostEqual(ml.mae(y, pred), 0.5)
        self.assertAlmostEqual(ml.rmse(y, pred), 1.0)
        self.assertAlmostEqual(ml.median_ae(y, pred), 0.0)
        self.assertAlmostEqual(ml.r2(y, y), 1.0)
        self.assertAlmostEqual(ml.r2(y, np.full(4, y.mean())), 0.0)


class LinearModelTests(unittest.TestCase):
    def test_ols_and_ridge_recover_coefficients(self):
        rng = np.random.RandomState(0)
        x = rng.randn(2000, 3)
        y = x @ np.array([2.0, -1.0, 0.5]) + 3.0 + 0.01 * rng.randn(2000)
        np.testing.assert_allclose(ml.ols_fit(x, y), [2.0, -1.0, 0.5, 3.0], atol=0.01)
        ridge = ml.RidgeRegression(alpha=1e-6).fit(x, y)
        self.assertGreater(ml.r2(y, ridge.predict(x)), 0.999)

    def test_ridge_shrinks_coefficients(self):
        rng = np.random.RandomState(1)
        x = rng.randn(500, 3)
        y = x @ np.array([2.0, -1.0, 0.5]) + 0.1 * rng.randn(500)
        weak = np.abs(ml.RidgeRegression(alpha=1e-6).fit(x, y).coef_)
        strong = np.abs(ml.RidgeRegression(alpha=1e4).fit(x, y).coef_)
        self.assertTrue((strong < weak).all())


class GradientBoostedTreesTests(unittest.TestCase):
    def test_learns_nonlinear_function_with_missing_values(self):
        rng = np.random.RandomState(1)
        x = rng.randn(6000, 4)
        y = np.sin(2 * x[:, 0]) + (x[:, 1] > 0) * 1.5 + x[:, 2] * x[:, 3] + 0.1 * rng.randn(6000)
        x[rng.rand(6000) < 0.1, 3] = np.nan
        model = ml.GradientBoostedTrees(n_trees=200, max_depth=4, learning_rate=0.1, subsample=0.8, colsample=1.0, min_samples_leaf=10, reg_lambda=1.0).fit(x[:4500], y[:4500])
        self.assertGreater(ml.r2(y[4500:], model.predict(x[4500:])), 0.8)
        self.assertAlmostEqual(model.feature_importances_.sum(), 1.0)

    def test_median_boosting_is_robust_to_outliers(self):
        rng = np.random.RandomState(2)
        x = rng.randn(4000, 2)
        y = 2 * x[:, 0] + 0.1 * rng.randn(4000)
        y[:200] += 50.0
        model = ml.GradientBoostedTrees(n_trees=100, max_depth=3, learning_rate=0.1, subsample=1.0, colsample=1.0, min_samples_leaf=20).fit(x, y)
        self.assertLess(ml.mae(y[200:], model.predict(x[200:])), 0.3)

    def test_quantile_boosting_coverage(self):
        rng = np.random.RandomState(3)
        x = rng.randn(8000, 2)
        y = x[:, 0] + (0.5 + np.abs(x[:, 1])) * rng.randn(8000)
        params = dict(n_trees=150, max_depth=3, learning_rate=0.1, subsample=0.8, colsample=1.0, min_samples_leaf=30)
        low = ml.GradientBoostedTrees(alpha=0.05, **params).fit(x[:6000], y[:6000])
        high = ml.GradientBoostedTrees(alpha=0.95, **params).fit(x[:6000], y[:6000])
        inside = (y[6000:] >= low.predict(x[6000:])) & (y[6000:] <= high.predict(x[6000:]))
        self.assertTrue(0.85 < inside.mean() < 0.95)

    def test_same_seed_gives_same_predictions(self):
        rng = np.random.RandomState(4)
        x = rng.randn(500, 3)
        y = x[:, 0] + rng.randn(500)
        first = ml.GradientBoostedTrees(n_trees=20, seed=7).fit(x, y).predict(x)
        second = ml.GradientBoostedTrees(n_trees=20, seed=7).fit(x, y).predict(x)
        np.testing.assert_array_equal(first, second)

    def test_input_validation(self):
        with self.assertRaises(ValueError):
            ml.GradientBoostedTrees(alpha=1.5)
        with self.assertRaises(RuntimeError):
            ml.GradientBoostedTrees().predict(np.zeros((2, 3)))
        model = ml.GradientBoostedTrees(n_trees=3).fit(np.random.randn(100, 3), np.random.randn(100))
        with self.assertRaises(ValueError):
            model.predict(np.zeros((2, 5)))


if __name__ == "__main__":
    unittest.main()
