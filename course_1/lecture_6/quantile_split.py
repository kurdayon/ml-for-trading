import numpy as np

class QuantileSplit:
    """
    A 1D feature selector + binner model based on quantile splits and a mean/std score.

    Parameters
    ----------
    n_splits : int
        Number of intervals (bins). Must be >= 2.

    Attributes (after fit)
    ----------------------
    best_feature_ : int
        Index of the selected feature.
    thresholds_ : ndarray of shape (n_splits-1,)
        Quantile split values for the selected feature.
    bin_predictions_ : ndarray of shape (n_splits,)
        Prediction per bin: mean(y_bin) / mean(y_bin**2).
    best_score_ : float
        Maximal score achieved: mean(pred_i * y_i) / std(pred_i * y_i).
    """

    def __init__(self, n_splits: int):
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        self.n_splits = n_splits
        self.best_feature_ = None
        self.thresholds_ = None
        self.bin_predictions_ = None
        self.best_score_ = None

    @staticmethod
    def _quantile_thresholds(x, n_splits):
        # Compute (n_splits-1) quantile cut points at i/n_splits, i=1..n_splits-1
        qs = np.linspace(0, 1, n_splits + 1)[1:-1]
        return np.quantile(x, qs, interpolation="linear")

    @staticmethod
    def _bin_indices(x, thresholds):
        # Map values to bin indices 0..n_splits-1 using left-closed, right-open bins
        # np.digitize returns indices in 0..len(thresholds); that’s exactly our bin id.
        return np.digitize(x, thresholds, right=False)

    @staticmethod
    def _bin_predictions(y, bins, n_splits, eps=1e-12):
        preds = np.zeros(n_splits, dtype=float)
        for b in range(n_splits):
            mask = (bins == b)
            if not np.any(mask):
                preds[b] = 0.0  # empty bin → neutral prediction
            else:
                yb = y[mask]
                num = float(np.mean(yb))
                den = float(np.mean(yb ** 2))
                preds[b] = num / (den + eps)  # protect against zero division
        return preds

    @staticmethod
    def _mean_over_std(z, eps=1e-12):
        return float(np.mean(z)) / (float(np.std(z)) + eps)

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y).reshape(-1)
        if X.ndim != 2:
            raise ValueError("X must be 2D (n_samples, n_features)")
        if y.ndim != 1 or y.shape[0] != X.shape[0]:
            raise ValueError("y must be 1D with length equal to n_samples")

        n_samples, n_features = X.shape
        best = {
            "feature": None,
            "thresholds": None,
            "bin_preds": None,
            "score": -np.inf,
        }

        for j in range(n_features):
            xj = X[:, j]
            thresholds = self._quantile_thresholds(xj, self.n_splits)
            bins = self._bin_indices(xj, thresholds)
            bin_preds = self._bin_predictions(y, bins, self.n_splits)
            per_sample_pred = bin_preds[bins]  # map each sample to its bin prediction
            score = self._mean_over_std(per_sample_pred * y)

            if score > best["score"]:
                best.update({
                    "feature": j,
                    "thresholds": thresholds,
                    "bin_preds": bin_preds,
                    "score": score,
                })

        # Store
        self.best_feature_ = best["feature"]
        self.thresholds_ = best["thresholds"]
        self.bin_predictions_ = best["bin_preds"]
        self.best_score_ = best["score"]
        return self

    def predict(self, X):
        if self.best_feature_ is None:
            raise RuntimeError("Model is not fitted yet.")
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        xj = X[:, self.best_feature_]
        bins = self._bin_indices(xj, self.thresholds_)
        return self.bin_predictions_[bins]

    # Optional convenience:
    def get_params(self, deep=True):
        return {"n_splits": self.n_splits}

    def set_params(self, **params):
        if "n_splits" in params:
            self.n_splits = params["n_splits"]
        return self

