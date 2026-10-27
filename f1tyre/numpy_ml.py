"""Machine learning implemented from scratch with NumPy: metrics, least squares, ridge regression and gradient-boosted trees."""
import numpy as np


def mae(y, pred):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(pred))))


def median_ae(y, pred):
    return float(np.median(np.abs(np.asarray(y) - np.asarray(pred))))


def rmse(y, pred):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(pred)) ** 2)))


def r2(y, pred):
    y = np.asarray(y, dtype=float)
    ss_res = np.sum((y - np.asarray(pred)) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot)


def ols_fit(x, y):
    x = np.asarray(x, dtype=float)
    design = np.column_stack([x, np.ones(len(x))])
    coef = np.linalg.lstsq(design, np.asarray(y, dtype=float), rcond=None)[0]
    return coef


def ols_predict(coef, x):
    x = np.asarray(x, dtype=float)
    return x @ coef[:-1] + coef[-1]


class RidgeRegression:
    def __init__(self, alpha=10.0):
        self.alpha = alpha

    def fit(self, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if x.ndim != 2 or len(x) != len(y):
            raise ValueError("x must be 2-D with one row per target value")
        self.mean_ = np.nanmean(x, axis=0)
        self.std_ = np.nanstd(x, axis=0)
        self.std_[self.std_ == 0] = 1.0
        z = self._standardize(x)
        self.intercept_ = y.mean()
        gram = z.T @ z + self.alpha * np.eye(z.shape[1])
        self.coef_ = np.linalg.solve(gram, z.T @ (y - self.intercept_))
        return self

    def _standardize(self, x):
        z = (np.asarray(x, dtype=float) - self.mean_) / self.std_
        return np.nan_to_num(z, nan=0.0)

    def predict(self, x):
        return self._standardize(x) @ self.coef_ + self.intercept_


def make_bin_edges(column, n_bins):
    values = column[~np.isnan(column)]
    if len(values) == 0:
        return np.array([])
    cut_points = np.linspace(0, 1, n_bins + 1)[1:-1]
    return np.unique(np.quantile(values, cut_points))


def bin_matrix(x, edges):
    binned = np.zeros(x.shape, dtype=np.int32)
    for j in range(x.shape[1]):
        column = x[:, j]
        binned[:, j] = np.searchsorted(edges[j], column, side="right") + 1
        binned[np.isnan(column), j] = 0
    return binned


class Tree:
    def __init__(self, feature, threshold, left, right, value, max_depth):
        self.feature = np.array(feature, dtype=np.int64)
        self.threshold = np.array(threshold, dtype=np.int64)
        self.left = np.array(left, dtype=np.int64)
        self.right = np.array(right, dtype=np.int64)
        self.value = np.array(value, dtype=float)
        self.max_depth = max_depth

    def predict_binned(self, binned):
        rows = np.arange(binned.shape[0])
        node = np.zeros(binned.shape[0], dtype=np.int64)
        for _ in range(self.max_depth):
            go_left = binned[rows, self.feature[node]] <= self.threshold[node]
            node = np.where(go_left, self.left[node], self.right[node])
        return self.value[node]


class GradientBoostedTrees:
    """Gradient boosting with pinball loss (alpha=0.5 is least absolute deviation).

    Each round: compute the negative gradient of the pinball loss, grow a
    histogram regression tree on it (variance-reduction splits), then set each
    leaf to the alpha-quantile of the current residuals in that leaf (exact line
    search for the pinball loss), scaled by the learning rate.
    """

    def __init__(self, n_trees=200, max_depth=6, learning_rate=0.05, subsample=0.8, colsample=0.8, min_samples_leaf=20, reg_lambda=5.0, alpha=0.5, n_bins=64, seed=42):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample = colsample
        self.min_samples_leaf = min_samples_leaf
        self.reg_lambda = reg_lambda
        self.alpha = alpha
        self.n_bins = n_bins
        self.seed = seed
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        if n_trees < 1 or max_depth < 1 or min_samples_leaf < 1:
            raise ValueError("n_trees, max_depth and min_samples_leaf must be at least 1")
        if not (0.0 < subsample <= 1.0 and 0.0 < colsample <= 1.0):
            raise ValueError("subsample and colsample must be in (0, 1]")

    def fit(self, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if x.ndim != 2 or len(x) != len(y):
            raise ValueError("x must be 2-D with one row per target value")
        if np.isnan(y).any():
            raise ValueError("y must not contain NaN")
        self.n_features_ = x.shape[1]
        self.edges_ = [make_bin_edges(x[:, j], self.n_bins) for j in range(x.shape[1])]
        binned = bin_matrix(x, self.edges_)
        n, n_features = binned.shape
        rng = np.random.RandomState(self.seed)
        self.base_ = float(np.quantile(y, self.alpha))
        self.gain_ = np.zeros(n_features)
        self.trees_ = []
        pred = np.full(n, self.base_)
        n_cols = max(1, int(round(self.colsample * n_features)))
        for _ in range(self.n_trees):
            rows = np.flatnonzero(rng.rand(n) < self.subsample)
            cols = np.sort(rng.choice(n_features, size=n_cols, replace=False))
            residual = y - pred
            grad = np.where(residual > 0, self.alpha, -(1.0 - self.alpha))
            tree = self._grow(binned[np.ix_(rows, cols)], grad[rows], residual[rows], cols)
            pred += tree.predict_binned(binned)
            self.trees_.append(tree)
        self.feature_importances_ = self.gain_ / self.gain_.sum() if self.gain_.sum() > 0 else self.gain_
        return self

    def _best_split(self, xs, grad, offsets, n_total_bins):
        n, n_features = xs.shape
        flat = (xs + offsets).ravel()
        size = n_features * n_total_bins
        counts = np.bincount(flat, minlength=size).reshape(n_features, n_total_bins)
        sums = np.bincount(flat, weights=np.repeat(grad, n_features), minlength=size).reshape(n_features, n_total_bins)
        left_n = np.cumsum(counts, axis=1)[:, :-1].astype(float)
        left_s = np.cumsum(sums, axis=1)[:, :-1]
        right_n = n - left_n
        right_s = grad.sum() - left_s
        lam = self.reg_lambda
        gain = left_s ** 2 / (left_n + lam) + right_s ** 2 / (right_n + lam) - grad.sum() ** 2 / (n + lam)
        gain[(left_n < self.min_samples_leaf) | (right_n < self.min_samples_leaf)] = -np.inf
        best = int(np.argmax(gain))
        feature, threshold = divmod(best, gain.shape[1])
        if not np.isfinite(gain[feature, threshold]) or gain[feature, threshold] <= 0:
            return None
        return feature, threshold, float(gain[feature, threshold])

    def _grow(self, xs, grad, residual, cols):
        n_total_bins = self.n_bins + 1
        offsets = (np.arange(xs.shape[1]) * n_total_bins).astype(np.int64)
        feature, threshold, left, right, value = [0], [n_total_bins], [0], [0], [0.0]
        stack = [(0, np.arange(len(grad)), 0)]
        while stack:
            node, idx, depth = stack.pop()
            split = None
            if depth < self.max_depth and len(idx) >= 2 * self.min_samples_leaf:
                split = self._best_split(xs[idx], grad[idx], offsets, n_total_bins)
            if split is None:
                left[node] = node
                right[node] = node
                value[node] = self.learning_rate * float(np.quantile(residual[idx], self.alpha))
                continue
            local_feature, local_threshold, gain = split
            self.gain_[cols[local_feature]] += gain
            goes_left = xs[idx, local_feature] <= local_threshold
            left_node = len(feature)
            right_node = left_node + 1
            for _ in range(2):
                feature.append(0)
                threshold.append(n_total_bins)
                left.append(0)
                right.append(0)
                value.append(0.0)
            feature[node] = int(cols[local_feature])
            threshold[node] = int(local_threshold)
            left[node] = left_node
            right[node] = right_node
            stack.append((left_node, idx[goes_left], depth + 1))
            stack.append((right_node, idx[~goes_left], depth + 1))
        return Tree(feature, threshold, left, right, value, self.max_depth)

    def predict(self, x):
        if not hasattr(self, "trees_"):
            raise RuntimeError("call fit before predict")
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or x.shape[1] != self.n_features_:
            raise ValueError(f"expected {self.n_features_} feature columns, got shape {x.shape}")
        binned = bin_matrix(x, self.edges_)
        pred = np.full(binned.shape[0], self.base_)
        for tree in self.trees_:
            pred += tree.predict_binned(binned)
        return pred
