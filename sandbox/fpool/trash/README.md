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

We now have a much more efficient implementation using dynamic programming:
[`PiecewiseConstantRegressor` in `src/tradinglab/models/piecewise_constant.py`](../../../src/tradinglab/models/piecewise_constant.py).
See the [models README](../../../src/tradinglab/models/README.md) for its interface
and test instructions.
