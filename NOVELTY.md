# Research novelty

## Research gap

The base paper predicts flight delay using operational flight variables,
including information such as `DEPARTURE_DELAY` that is only known after a
flight has started. This can produce strong benchmark scores but is not
strictly suitable for pre-departure prediction.

## Proposed contribution

This project introduces a leakage-aware evaluation framework that separates:

- **Operational baseline:** follows the base-paper-style setup and measures
  performance when operational variables are available.
- **Strict pre-departure model (Path B):** uses only information available
  before departure.
- **Leakage diagnostic (Path A):** demonstrates how post-departure information
  inflates performance. It is not used for deployment.

The proposed improvement also evaluates engineered pre-departure features such
as route, scheduled speed, peak-hour, and red-eye indicators.

## Evaluation plan

The comparison uses the same dataset, target, train/test split, and metrics
where possible:

1. Reproduce the base-paper operational model.
2. Train the strict pre-departure model.
3. Compare basic features with the engineered pre-departure features.
4. Report accuracy, precision, recall, F1-score, MAE, RMSE, or R² only where
   the task and model type are comparable.

## Current conclusion

The current results show that post-departure information produces much higher
scores than strict pre-departure information. Therefore, the contribution is
not a claim of higher raw accuracy. It is a claim of improved practical
validity, transparent leakage detection, and a deployable pre-departure
prediction workflow.

The implementation is available in
[`ml/scripts/train_model.py`](ml/scripts/train_model.py) and
[`ml/scripts/reproduce_base_paper.py`](ml/scripts/reproduce_base_paper.py).
