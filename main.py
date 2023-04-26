import argparse

from dataloader import DataLoader
from model.training import optimize_ce_model, optimize_lr_model
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import accuracy_score


def assess_model(args):
    # Load data
    loader = DataLoader(source_state=args.source_state, source_year=args.source_year, target_state=args.target_state,
                        target_year=args.target_year, prediction_target=args.prediction_target, feature_sets_to_drop=[],
                        n_source_train=args.n_source_train, n_target_train=args.n_target_train)

    X_train, y_train, X_source_test, y_source_test, X_target_val, y_target_val, X_target_test, y_target_test, mask = loader.load_data(
        return_target_mask=True)

    y_train = y_train.astype(float).squeeze()
    y_source_test = y_source_test.astype(float).squeeze()
    y_target_val = y_target_val.astype(float).squeeze()
    y_target_test = y_target_test.astype(float).squeeze()

    # Use standard scaler for the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_source_test = scaler.transform(X_source_test)
    X_target_val = scaler.transform(X_target_val)
    X_target_test = scaler.transform(X_target_test)

    # Fit CE model
    ce_model = optimize_ce_model(X_train, y_train, args, instance_weighted=False, X_val=X_target_val,
                                 y_val=y_target_val, n_jobs=args.n_threads)
    # print(ce_model)
    print("CE test accuracy: ", accuracy_score(y_source_test, ce_model.predict(X_source_test)), " (source)")
    print("CE test accuracy: ", accuracy_score(y_target_test, ce_model.predict(X_target_test)), " (target)")

    # Instance weighted
    y_train_weights = np.concatenate((np.expand_dims(y_train, axis=-1), np.expand_dims(mask, axis=-1)), axis=-1)
    weighted_ce_model = optimize_ce_model(X_train, y_train_weights, args, instance_weighted=True, X_val=X_target_val,
                                          y_val=y_target_val, n_jobs=args.n_threads)
    # print(weighted_ce_model)
    print("Inst. weighted CE test accuracy: ", accuracy_score(y_source_test, weighted_ce_model.predict(X_source_test)),
          " (source)")
    print("Inst. weighted CE test accuracy: ", accuracy_score(y_target_test, weighted_ce_model.predict(X_target_test)),
          " (target)")

    # Fit LR model
    lr_model = optimize_lr_model(X_train, y_train, args, instance_weighted=False, X_val=X_target_val,
                                 y_val=y_target_val, n_jobs=args.n_threads)
    # print(lr_model)
    print("LR test accuracy: ", accuracy_score(y_source_test, lr_model.predict(X_source_test)), " (source)")
    print("LR test accuracy: ", accuracy_score(y_target_test, lr_model.predict(X_target_test)), " (target)")

    weighted_lr_model = optimize_lr_model(X_train, y_train_weights, args, instance_weighted=True, X_val=X_target_val,
                                          y_val=y_target_val, n_jobs=args.n_threads)
    # print(weighted_lr_model)
    print("Inst. weighted LR test accuracy: ", accuracy_score(y_source_test, weighted_lr_model.predict(X_source_test)),
          " (source)")
    print("Inst. weighted LR test accuracy: ", accuracy_score(y_target_test, weighted_lr_model.predict(X_target_test)),
          " (target)")


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ho_target", type=str, default='acc')

    parser.add_argument("--source_state", type=str, default='FL')
    parser.add_argument("--source_year", type=str, default='2014')
    parser.add_argument("--target_state", type=str, default='CA')
    parser.add_argument("--target_year", type=str, default='2014')
    parser.add_argument("--prediction_target", type=str, default='ACSPublicCoverage')
    parser.add_argument("--n_source_train", type=int, default=50)
    parser.add_argument("--n_target_train", type=int, default=10)

    parser.add_argument("--n_threads", type=int, default=-1)

    args = parser.parse_args()

    assess_model(args)
