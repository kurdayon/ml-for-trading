# Nested cross-validation

This note preserves the design idea from `../meta_fold.py`. It describes an
evaluation procedure; it does not provide an implementation.

## Why use it?

Ordinary N-fold cross-validation divides a dataset into N folds. Each fold is
predicted by a model trained on the other N − 1 folds, producing out-of-sample
predictions across the dataset.

If we evaluate several candidate models this way, choose the best, and report
its score from the same evaluation, the estimate can be optimistic. Selection
favors candidates that benefited from chance as well as those that learned
useful relationships.

Nested cross-validation separates model selection from evaluation. The inner
loop chooses a model; the outer loop evaluates the selection and fitting
procedure on observations that were not used to make that choice.

## Procedure

For each outer split:

1. Set aside the outer test fold and use the remaining observations as the
   outer training set.
2. Run inner cross-validation entirely within the outer training set to compare
   candidate models or hyperparameter settings.
3. Choose the candidate with the best inner validation score, using a metric
   specified in advance.
4. Refit that candidate on the entire outer training set.
5. Predict the outer test fold and retain its predictions and evaluation results.

Summarize performance across the outer test folds. Different outer splits may
select different candidates: the results evaluate the selection procedure,
not a single fixed model. Keep outer test results out of the selection process.

All learned preprocessing, including feature selection, scaling, and probability
calibration, must be fitted using only the training portion of each split.

After evaluation, the same selection procedure can be applied to all available
training data, and the selected candidate refitted for subsequent use.

## Time-series forecasting

For forecasting, use chronological splits in both loops so that training
observations precede the observations being evaluated. Ordinary folds, even
without shuffling, can include future observations in a training set.

Expanding or rolling training windows can implement this arrangement. Unlike
ordinary N-fold cross-validation, they typically leave an initial training
period without out-of-sample predictions. Where target intervals overlap split
boundaries, exclude training observations whose labels use information from
the evaluation period.
