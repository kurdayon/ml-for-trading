# Models

This directory contains model implementations. Each entry documents the method,
its interface, computational cost, and how to test it.

## `piecewise_constant.py`

### What it does

[`PiecewiseConstantRegressor`](piecewise_constant.py) fits a step function to one
numeric input feature. Given training pairs `(x, y)` and a requested number of
splits, it divides the sorted input values into exactly `n_splits + 1` nonempty
regions. Each region predicts the mean of its training targets.

The model chooses boundaries that minimize the total training sum of squared
errors (SSE). Equal input values always stay in the same region, and every
observation contributes to the objective, including repeated inputs. Zero splits
produces a single constant prediction: the overall target mean.

Dynamic programming finds a global optimum under these constraints, subject to
floating-point precision. This is an optimum for training error; it does not
guarantee good predictions on unseen data. More splits can overfit.

### Interface

The estimator requires NumPy. Matplotlib is needed only for the comparison demo.

| Interface | Expected input or result |
| --- | --- |
| `PiecewiseConstantRegressor(n_splits=3)` | A nonnegative integer number of boundaries. It must be smaller than the number of distinct training input values; validated during `fit`. |
| `fit(x, y)` | Two one-dimensional real numeric arrays or lists of equal, nonzero length. Values must be finite; NaN, infinity, and complex values are rejected. Inputs may be unsorted and are not modified. Returns the fitted model. |
| `predict(x)` | A one-dimensional array or list of finite real inputs. Returns a one-dimensional NumPy array of predictions in the input order. An empty input returns an empty array. Calling before `fit` raises `RuntimeError`. |

After fitting, the model exposes:

- `thresholds_`: the `n_splits` boundaries in increasing order.
- `values_`: the `n_splits + 1` region means in increasing input order.
- `training_sse_`: the sum of squared training residuals.

A value exactly equal to a threshold belongs to the region on its right.
Inputs outside the training range use the nearest end region's prediction.

With `src` on the Python import path (for example, start Python with
`PYTHONPATH=src python3` from the repository root):

```python
from tradinglab.models.piecewise_constant import PiecewiseConstantRegressor

x = [0, 1, 2, 3, 4, 5]
y = [2, 2, -1, -1, 9, 9]

model = PiecewiseConstantRegressor(n_splits=2).fit(x, y)
print(model.thresholds_)          # [1.5 3.5]
print(model.values_)              # [ 2. -1.  9.]
print(model.predict([0, 2, 5]))   # [ 2. -1.  9.]
print(model.training_sse_)        # 0.0
```

### How dynamic programming makes it efficient

The algorithm sorts the observations and groups equal input values. It then
builds cumulative counts, target sums, and squared target sums, allowing the SSE
of any contiguous region to be calculated in constant time.

For each number of regions and each prefix of the sorted groups, it stores the
best achievable error. To extend a solution, it considers each possible start of
the final region, combining that region's error with a previously solved prefix.
Saved boundary choices allow the optimal partition to be reconstructed.
Reusing these smaller solutions avoids enumerating every complete partition.

Let `n` be the number of observations, `m` the number of distinct input values,
and `k = n_splits + 1` the number of regions. The current implementation uses:

- **Fitting time: `O(n log n + k * m²)`** — sorting plus dynamic programming.
- **Fitting memory: `O(n + k * m)`** — observation arrays, cumulative statistics,
  rolling error arrays, and saved boundary choices. It does not store an
  `m × m` table of region costs.

If all input values are distinct, `m = n`, so fitting time is
`O(n log n + k * n²)`. This is efficient compared with exhaustive search, but
the quadratic dependence on distinct inputs can still be costly for large data.

### Why brute force becomes intractable

With `m` distinct input values, there are `m - 1` possible boundary positions.
Choosing `s = n_splits` boundaries requires checking
`C(m - 1, s)` complete partitions by brute force. The included `_brute_force`
reference computes means and residuals directly for each partition, taking
`O(n log n + n * C(m - 1, s))` time.

For example, 40 distinct points and 3 splits require 9,139 partitions, while
100 distinct points and 10 splits require 15,579,278,510,796 partitions.
This combinatorial growth makes exhaustive search computationally intractable
for many larger combinations of point and split counts. Dynamic programming
avoids that explosion while still finding the optimal training partition.

### How to test it

Run the existing unit tests from the repository root:

```sh
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_piecewise_constant.py'
```

The [tests](../../../tests/test_piecewise_constant.py) compare the model with an
independent exhaustive search on small random datasets, including duplicate
inputs. They also cover known step functions, extrapolation, threshold equality,
zero and maximum splits, a single point, constant-target ties, large target
offsets, adjacent floating-point inputs, and invalid inputs.

For a visual comparison on random data, run:

```sh
python3 src/tradinglab/models/piecewise_constant.py --points 40 --splits 3 --seed 42
```

The demo generates noisy sine-wave observations, fits both methods, prints
timings and SSE, checks agreement of boundaries and predictions, and plots the
results. To save the plot instead of opening a window:

```sh
python3 src/tradinglab/models/piecewise_constant.py --points 40 --splits 3 --save comparison.png
```

For runs from an editor, change `DEMO_POINTS` and `DEMO_SPLITS` near the top of
the Python file. Command-line `--points` and `--splits` override those defaults.
Use `points >= 1` and `0 <= splits < points`.

**Keep demo datasets small: the demo always runs brute force as well as dynamic
programming.** To fit larger datasets, use `PiecewiseConstantRegressor` directly;
its `fit` method does not run the exhaustive comparison.
