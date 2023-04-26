import copy
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from skopt import BayesSearchCV
from skopt.space import Real
import itertools

from model.sklearn_models import InstanceWeightingLogisticRegression
from model.models import TorchLabelRelaxationNNClassifier, TorchCrossEntropyNNClassifier, \
    TorchLabelSmoothingNNClassifier

# Hyperparameter search spaces
LR_GRID = {'alpha': [0.05, 0.1, 0.25, 0.4, 0.5], 'learning_rate': [1e-3, 1e-4],
           'l2_penalty': [0.0, 1e-4, 1e-3]}
LS_GRID = {'alpha': [0.05, 0.1, 0.25, 0.4, 0.5], 'learning_rate': [1e-3, 1e-4],
           'l2_penalty': [0.0, 1e-4, 1e-3]}
CE_GRID = {'learning_rate': [1e-3, 1e-4], 'l2_penalty': [0.0, 1e-4, 1e-3]}
IWCE_GRID = {'learning_rate': [1e-3, 1e-4], 'instance_weight': [0.1, 0.25, 0.5, 0.75, 1.0],
             'l2_penalty': [0.0, 1e-4, 1e-3]}
INST_LOG_REG_GRID = {'instance_weight': [0.1, 0.25, 0.5, 0.75, 1.0]}

LR_GRID_BAYES = {'alpha': Real(low=1e-3, high=0.5, prior="uniform"),
                 'learning_rate': Real(low=1e-4, high=1e-1, prior="log-uniform"),
                 'l2_penalty': Real(low=1e-5, high=1e-0, prior="log-uniform")}
LS_GRID_BAYES = {'alpha': Real(low=1e-3, high=0.5, prior="uniform"),
                 'learning_rate': Real(low=1e-4, high=1e-1, prior="log-uniform"),
                 'l2_penalty': Real(low=1e-5, high=1e-0, prior="log-uniform")}
CE_GRID_BAYES = {'learning_rate': Real(low=1e-4, high=1e-1, prior="log-uniform"),
                 'l2_penalty': Real(low=1e-5, high=1e-0, prior="log-uniform")}
IWCE_GRID_BAYES = {'learning_rate': Real(low=1e-4, high=1e-1, prior="log-uniform"),
                   'l2_penalty': Real(low=1e-5, high=1e-0, prior="log-uniform"),
                   'instance_weight': Real(low=1e-3, high=1, prior='uniform')}
INST_LOG_REG_GRID_BAYES = {'instance_weight': Real(low=1e-3, high=1, prior='uniform')}


def accuracy_scoring_fn(estimator, X, y, sample_weight=None):
    if len(y.shape) == 2:
        # Deal with alpha data
        y = y[:, 0]

    return accuracy_score(y, estimator.predict(X), sample_weight=sample_weight)


def balanced_accuracy_scoring_fn(estimator, X, y, sample_weight=None):
    if len(y.shape) == 2:
        # Deal with alpha data
        y = y[:, 0]

    return balanced_accuracy_score(y, estimator.predict(X), sample_weight=sample_weight)


def get_scoring_fn(args):
    if args.ho_target == "acc":
        return accuracy_scoring_fn
    elif args.ho_target == "bal_acc":
        return balanced_accuracy_scoring_fn


def get_opt_model(model, args, grid_bayes, grid_random, verbose, n_jobs, bayes_n_points=10):
    if args.ho_method == "bayes":
        cv_model = BayesSearchCV(model, grid_bayes, n_iter=args.ho_n_iters, cv=args.n_folds, verbose=verbose,
                                 n_points=bayes_n_points, n_jobs=n_jobs, random_state=args.seed,
                                 scoring=get_scoring_fn(args))
    else:
        cv_model = GridSearchCV(estimator=model, param_grid=grid_random, cv=args.n_folds, n_jobs=n_jobs,
                                scoring=get_scoring_fn(args), verbose=verbose)
    return cv_model


def single_split_cv(args, param_grid, model_template, X_train, y_train, X_val, y_val):
    keys, values = zip(*param_grid.items())
    candidates = [dict(zip(keys, v)) for v in itertools.product(*values)]
    best_score = -np.inf
    best_model = None
    for candidate in candidates:
        # Deepcopy model template to new object
        model = copy.deepcopy(model_template)
        model.set_params(**candidate)
        model.fit(X_train, y_train)
        val_score = get_scoring_fn(args)(model, X_val, y_val)
        print("Validation score for {}: {}".format(candidate, val_score))

        if val_score > best_score:
            best_score = val_score
            best_model = model
    return best_model


def optimize_lr_model(X, y, args, instance_weighted: bool, n_jobs=-1, ret_hist=False, validation_split=0.0,
                      n_classes: int = 2, verbose: int = 1, X_val=None, y_val=None):
    # lr_int = LabelRelaxationNNClassifier(n_classes=2, hidden_layer_sizes=(), provide_alphas=True,
    #                                      validation_split=validation_split)
    lr_int = TorchLabelRelaxationNNClassifier(n_classes=n_classes, hidden_layer_sizes=(128,),
                                              provide_alphas=instance_weighted, validation_split=validation_split)
    if X_val is not None and y_val is not None:
        lr = single_split_cv(args, LR_GRID, lr_int, X, y, X_val, y_val)
    else:
        lr = get_opt_model(lr_int, args, LR_GRID_BAYES, LR_GRID, verbose, n_jobs)
        lr.fit(X, y)

        print("CV results (LR):", lr.cv_results_)
        print("Best params (LR):", lr.best_params_)

    if ret_hist:
        return lr, lr.history()
    return lr


def optimize_ce_model(X, y, args, instance_weighted: bool, n_jobs=-1, ret_hist=False, validation_split=0.0,
                      n_classes: int = 2, verbose: int = 1, X_val=None, y_val=None):
    if instance_weighted:
        param_grid = IWCE_GRID
        param_grid_bayes = IWCE_GRID_BAYES
    else:
        param_grid = CE_GRID
        param_grid_bayes = CE_GRID_BAYES

    # ce_int = CrossEntropyNNClassifier(n_classes=2, hidden_layer_sizes=(), validation_split=validation_split)
    ce_int = TorchCrossEntropyNNClassifier(n_classes=n_classes, hidden_layer_sizes=(128,),
                                           validation_split=validation_split)

    if X_val is not None and y_val is not None:
        ce = single_split_cv(args, CE_GRID if not instance_weighted else IWCE_GRID, ce_int, X, y, X_val, y_val)
    else:
        ce = get_opt_model(ce_int, args, param_grid_bayes, param_grid, verbose, n_jobs)
        ce.fit(X, y)

        print("CV results (CE):", ce.cv_results_)
        print("Best params (CE):", ce.best_params_)

    if ret_hist:
        return ce, ce.history()
    return ce


def optimize_ls_model(X, y, args, instance_weighted: bool, n_jobs=-1, ret_hist=False, validation_split=0.0,
                      n_classes: int = 2, verbose: int = 1, X_val=None, y_val=None):
    # ls_int = CrossEntropyNNClassifier(n_classes=2, hidden_layer_sizes=(), validation_split=validation_split)
    ls_int = TorchLabelSmoothingNNClassifier(n_classes=n_classes, hidden_layer_sizes=(128,),
                                             provide_alphas=instance_weighted, validation_split=validation_split)

    if X_val is not None and y_val is not None:
        ls = single_split_cv(args, LS_GRID, ls_int, X, y, X_val, y_val)
    else:
        ls = get_opt_model(ls_int, args, LS_GRID_BAYES, LS_GRID, verbose, n_jobs)
        ls.fit(X, y)

        print("CV results (LS):", ls.cv_results_)
        print("Best params (LS):", ls.best_params_)

    if ret_hist:
        return ls, ls.history()
    return ls


def optimize_instance_weighting_logistic_regression_model(X, y, args, n_jobs=-1, verbose: int = 1, X_val=None,
                                                          y_val=None):
    log_reg_int = InstanceWeightingLogisticRegression(penalty='none', solver='lbfgs', max_iter=1000)

    if X_val is not None and y_val is not None:
        log_reg = single_split_cv(args, INST_LOG_REG_GRID, log_reg_int, X, y, X_val, y_val)
    else:
        log_reg = get_opt_model(log_reg_int, args, INST_LOG_REG_GRID_BAYES, INST_LOG_REG_GRID, verbose, n_jobs)
        log_reg.fit(X, y)

    return log_reg
