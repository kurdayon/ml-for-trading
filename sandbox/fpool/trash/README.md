# Retired experiments

This folder keeps old implementations for reference.

- [`step_preds_to_position_strat.py`](step_preds_to_position_strat.py) was a very
  simple wrapper around `int_to_float_steps_model.py`: it delegated fitting and
  applied a supplied transformation to the model's predictions. It added little
  functionality and was therefore not very useful.
- [`int_to_float_steps_model.py`](int_to_float_steps_model.py) was moved here
  because it used a very inefficient brute-force search over all combinations of
  split boundaries. This becomes impractical as the number of points and splits
  grows.
- [`pwm.py`](pwm.py) contains an unfinished attempt to fit a piecewise-monotonic
  function with a limit on the number of turning points (peaks and valleys).
  It was moved here because the implementation was never completed.
- [`test_opos.py`](test_opos.py) is an empirical check of the known position-sizing
  formula `p = mean / (mean² + variance)`, using numerical optimization. It also
  compares monotonic positions based on that formula against random search.
  It is retained here as an exploratory experiment rather than a proof.

For the old brute-force step model, we now have a much more efficient
implementation using dynamic programming:
[`PiecewiseConstantRegressor` in `src/tradinglab/models/piecewise_constant.py`](../../../src/tradinglab/models/piecewise_constant.py).
See the [models README](../../../src/tradinglab/models/README.md) for its interface
and test instructions.
