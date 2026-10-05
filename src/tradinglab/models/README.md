# Models

## `piecewise_constant.py`

[`PiecewiseConstantRegressor`](piecewise_constant.py) fits a step function to
one numeric feature, choosing exactly `n_splits + 1` nonempty regions to minimize
training squared error. Each region predicts its target mean.

| Method | Interface |
| --- | --- |
| `__init__(n_splits=3)` | Number of boundaries: a nonnegative integer smaller than the number of distinct training input values. |
| `fit(x, y)` | Input feature values `x` and targets `y`: one-dimensional arrays or lists of finite real numbers, with equal, nonzero lengths. Unsorted and repeated inputs are allowed. Returns the fitted model. |
| `predict(x)` | One-dimensional array or list of finite real feature values. Returns a NumPy array of predictions. Requires a fitted model. |

The implementation sorts inputs and uses dynamic programming with cumulative
statistics to reuse optimal solutions for smaller regions. Fitting takes
`O(n log n + k * m²)` time for `n` points, `m` distinct inputs, and
`k = n_splits + 1` regions, avoiding the combinatorial cost of brute force.

Run tests from the repository root (NumPy required):

```sh
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_piecewise_constant.py'
```

For a plot and timing comparison against brute force, run the demo (also requires
Matplotlib). Keep counts small because the demo runs both methods:

```sh
python3 src/tradinglab/models/piecewise_constant.py --points 40 --splits 3
```

See [piecewise_constant.md](piecewise_constant.md) for examples, implementation
details, complexity analysis, and further testing options.
