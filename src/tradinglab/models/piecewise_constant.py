"""Exact one-dimensional piecewise-constant least-squares regression.

The model chooses ``n_splits`` boundaries, giving ``n_splits + 1`` nonempty
regions. Equal input values always belong to the same region. Dynamic
programming finds a global minimum of training squared error, subject to these
constraints; this does not guarantee optimal performance on unseen data.

Run this file to compare the model with an independent exhaustive search:

    python src/tradinglab/models/piecewise_constant.py --points 40 --splits 3
    python src/tradinglab/models/piecewise_constant.py --save comparison.png

For runs from an editor, set DEMO_POINTS and DEMO_SPLITS below. Command-line
--points and --splits override those defaults. Keep 0 <= splits < points;
the exhaustive comparison can be slow for large counts.

NumPy is required by the estimator. Matplotlib is required only by the demo.
"""

from itertools import combinations
from numbers import Integral
from time import perf_counter

import numpy as np


# Demo settings: edit these when running this file directly from an editor.
DEMO_POINTS = 40  # Number of randomly generated observations.
DEMO_SPLITS = 3   # Number of boundaries (produces DEMO_SPLITS + 1 regions).


def _vector(values, name):
    """Validate and convert a finite, real, one-dimensional numeric array."""
    values = np.asarray(values)
    if values.ndim != 1 or values.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a one-dimensional real numeric array")
    values = values.astype(np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values")
    return values


def _thresholds(unique_x, cuts):
    """Place boundaries between groups, with the right group owning equality."""
    left, right = unique_x[cuts - 1], unique_x[cuts]
    midpoint = left / 2 + right / 2  # Avoid overflow in left + right.
    # Adjacent floating-point values may have no representable midpoint.
    return np.where(midpoint > left, midpoint, right)


class PiecewiseConstantRegressor:
    """Fit exactly ``n_splits + 1`` constant regions to one input feature.

    Parameters
    ----------
    n_splits : int, default=3
        Number of boundaries, not number of regions. Zero fits a single mean.
        Must be smaller than the number of distinct training input values.

    Attributes after fitting
    ------------------------
    thresholds_ : ndarray
        Increasing boundaries. A value equal to a boundary goes to its right.
    values_ : ndarray
        Mean target for each region, ordered by increasing input.
    training_sse_ : float
        Sum of squared training residuals.

    Notes
    -----
    For n observations, m distinct inputs, and k = n_splits + 1 regions,
    fitting takes O(n log n + k*m**2) time and O(n + k*m) memory. It uses
    cumulative counts, target sums, and squared target sums to score a region
    in constant time. No quadratic-size table of region costs is stored.

    Targets are centered and accumulated in extended precision where available
    to reduce cancellation in the squared-error formula. As with other numeric
    estimators, the optimum is subject to floating-point precision. Exact ties
    choose the earliest final boundary, then the earliest preceding boundary.
    """

    def __init__(self, n_splits=3):
        self.n_splits = n_splits

    def fit(self, x, y):
        """Fit from finite 1D arrays of equal nonzero length; return self.

        Inputs are not modified. Missing values are rejected rather than
        silently removed. Repeated x values retain their observation counts.
        """
        if (isinstance(self.n_splits, bool)
                or not isinstance(self.n_splits, Integral)
                or self.n_splits < 0):
            raise ValueError("n_splits must be a non-negative integer")
        x, y = _vector(x, "x"), _vector(y, "y")
        if len(x) == 0 or len(x) != len(y):
            raise ValueError("x and y must have equal, nonzero lengths")

        order = np.argsort(x, kind="stable")
        unique_x, starts, counts = np.unique(
            x[order], return_index=True, return_counts=True
        )
        m, k = len(unique_x), self.n_splits + 1
        if k > m:
            raise ValueError("n_splits must be smaller than the number of distinct x values")

        targets = y[order].astype(np.longdouble)
        offset = np.mean(targets)
        centered = targets - offset
        count = np.r_[0, np.cumsum(counts)]
        sums = np.r_[np.longdouble(0), np.cumsum(np.add.reduceat(centered, starts))]
        squares = np.r_[
            np.longdouble(0), np.cumsum(np.add.reduceat(centered ** 2, starts))
        ]

        # previous[j]: minimum SSE for the first j groups with r - 1 regions.
        previous = np.full(m + 1, np.inf, dtype=np.longdouble)
        previous[0] = 0
        parents = np.full((k + 1, m + 1), -1, dtype=np.intp)
        for r in range(1, k + 1):
            current = np.full(m + 1, np.inf, dtype=np.longdouble)
            for end in range(r, m + 1):
                begin = np.arange(r - 1, end)
                region_sum = sums[end] - sums[begin]
                errors = (squares[end] - squares[begin]
                          - region_sum ** 2 / (count[end] - count[begin]))
                # A true SSE is nonnegative; cancellation can make it tiny-negative.
                candidates = previous[begin] + np.maximum(errors, 0)
                best = int(np.argmin(candidates))
                current[end] = candidates[best]
                parents[r, end] = begin[best]
            previous = current

        edges = [m]
        for r in range(k, 0, -1):
            edges.append(int(parents[r, edges[-1]]))
        edges = np.array(edges[::-1], dtype=np.intp)
        thresholds = _thresholds(unique_x, edges[1:-1])
        values = np.array([
            np.mean(targets[count[a]:count[b]])
            for a, b in zip(edges[:-1], edges[1:])
        ], dtype=np.float64)
        residuals = y.astype(np.longdouble) - values[np.searchsorted(thresholds, x, side="right")]

        self.thresholds_ = thresholds
        self.values_ = values
        self.training_sse_ = float(np.sum(residuals ** 2))
        return self

    def predict(self, x):
        """Return 1D predictions; extrapolate with the nearest end region.

        An empty input returns an empty array. Call fit before predict.
        """
        if not hasattr(self, "thresholds_"):
            raise RuntimeError("Call fit before predict")
        x = _vector(x, "x")
        return self.values_[np.searchsorted(self.thresholds_, x, side="right")]


def _brute_force(x, y, n_splits):
    """Independent reference search for the demo; enumerate every partition.

    Score direct residuals instead of reusing the DP cumulative-sum formula.
    This is intentionally expensive and intended only for small datasets.
    """
    order = np.argsort(x, kind="stable")
    unique_x, starts = np.unique(x[order], return_index=True)
    targets = y[order].astype(np.longdouble)
    best_score, best_key = np.inf, None
    for cuts in combinations(range(1, len(unique_x)), n_splits):
        row_edges = [0, *(int(starts[c]) for c in cuts), len(x)]
        groups = [targets[a:b] for a, b in zip(row_edges[:-1], row_edges[1:])]
        means = [np.mean(group) for group in groups]
        score = sum(np.sum((group - mean) ** 2) for group, mean in zip(groups, means))
        key = tuple(reversed(cuts))
        if score < best_score or (score == best_score and (best_key is None or key < best_key)):
            best_score, best_key = score, key
            best_cuts, best_means = cuts, means
    return (
        _thresholds(unique_x, np.array(best_cuts, dtype=np.intp)),
        np.asarray(best_means, dtype=np.float64),
        float(best_score),
    )


def _demo():
    import argparse
    from math import comb

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=int, default=DEMO_POINTS,
                        help="Number of random observations (default: %(default)s)")
    parser.add_argument("--splits", type=int, default=DEMO_SPLITS,
                        help="Number of boundaries (regions minus one; default: %(default)s)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save", help="Save the plot to this path instead of opening a window")
    args = parser.parse_args()
    if args.points < 1 or not 0 <= args.splits < args.points:
        parser.error("Require points >= 1 and 0 <= splits < points")

    import matplotlib.pyplot as plt

    rng = np.random.default_rng(args.seed)
    x = rng.uniform(-3, 3, args.points)
    y = np.sin(2 * x) + rng.normal(0, 0.25, args.points)
    print(f"{args.points} points, {args.splits} splits, {args.splits + 1} regions")
    print(f"Brute-force partitions: {comb(args.points - 1, args.splits):,}", flush=True)

    start = perf_counter()
    model = PiecewiseConstantRegressor(args.splits).fit(x, y)
    dp_seconds = perf_counter() - start
    start = perf_counter()
    thresholds, values, brute_sse = _brute_force(x, y, args.splits)
    brute_seconds = perf_counter() - start

    grid = np.unique(np.r_[np.linspace(-3.2, 3.2, 2000), x, model.thresholds_, thresholds])
    dp_prediction = model.predict(grid)
    brute_prediction = values[np.searchsorted(thresholds, grid, side="right")]
    np.testing.assert_allclose(model.training_sse_, brute_sse, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(model.thresholds_, thresholds, rtol=0, atol=0)
    np.testing.assert_allclose(dp_prediction, brute_prediction, rtol=1e-10, atol=1e-10)
    print(f"Dynamic programming: {dp_seconds:.6f} s; SSE = {model.training_sse_:.12g}")
    print(f"Brute force:         {brute_seconds:.6f} s; SSE = {brute_sse:.12g}")
    print("Checks passed: boundaries and predictions match (within numerical tolerance).")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(x, y, color="black", s=25, alpha=0.6, label="Training points")
    ax.step(grid, dp_prediction, where="post", linewidth=3,
            label=f"Dynamic programming ({dp_seconds:.4f} s)")
    ax.step(grid, brute_prediction, where="post", linestyle="--", linewidth=2,
            label=f"Brute force ({brute_seconds:.4f} s)")
    ax.set(xlabel="x", ylabel="y", title=f"Optimal piecewise-constant fit: {args.splits} splits")
    ax.legend()
    fig.tight_layout()
    if args.save:
        fig.savefig(args.save, dpi=150)
        print(f"Plot saved to {args.save}")
    else:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    _demo()
