from sklearn.linear_model import LogisticRegression
import numpy as np


class InstanceWeightingLogisticRegression(LogisticRegression):
    def __init__(self, penalty="l2", dual=False, tol=1e-4, C=1.0, fit_intercept=True, intercept_scaling=1,
                 class_weight=None, random_state=None, solver="lbfgs", max_iter=100, multi_class="auto", verbose=0,
                 warm_start=False, n_jobs=None, l1_ratio=None, instance_weight: float = 0.0):
        super().__init__(penalty=penalty, dual=dual, tol=tol, C=C, fit_intercept=fit_intercept,
                         intercept_scaling=intercept_scaling, class_weight=class_weight, random_state=random_state,
                         solver=solver, max_iter=max_iter, multi_class=multi_class, verbose=verbose,
                         warm_start=warm_start, n_jobs=n_jobs, l1_ratio=l1_ratio)
        self.instance_weight = instance_weight

    def fit(self, X, y, sample_weight=None):
        weights = np.squeeze(y[:, -1])
        weights = np.where(weights == 0, np.ones_like(weights), np.ones_like(weights) * self.instance_weight)
        targets = np.squeeze(y[:, :-1])

        return super().fit(X, targets, sample_weight=weights)
