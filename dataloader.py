from folktables import ACSDataSource, ACSIncome, ACSPublicCoverage, ACSEmployment, ACSMobility, ACSTravelTime
import numpy as np
from sklearn.model_selection import train_test_split
from features import FEATURE_SETS


class DataLoader:
    def __init__(self, source_year, source_state, target_year, target_state, prediction_target, feature_sets_to_drop,
                 n_target_train, n_source_train, horizon='1-Year', survey='person', random_state=137):
        """Dataloader class for loading data from folktables package for learning under distributional shifts.

        Args:
            source_year (int): The year of the source data.
            source_state (int): The US state of the source data.
            target_year (str): The year of the target data.
            target_state (str): The US state of the target data.
            prediction_target (str): The target variable for prediction. Must be one of the following: 'ACSIncome',
                                    'ACSPublicCoverage', 'ACSEmployment', 'ACSMobility', 'ACSTravelTime'
            feature_sets_to_drop (list[str]): The feature sets that should be removed from the data. If empty, the
                                    default feature set of folktables is used. Can contain the following: 'WORK',
                                     'SENSITIVE'.
        """
        self.source_year = source_year
        self.source_state = source_state
        self.target_year = target_year
        self.target_state = target_state

        self.prediction_target = prediction_target
        self.n_source_train = n_source_train
        self.n_target_train = n_target_train

        self.horizon = horizon
        self.survey = survey

        self.feature_sets_to_drop = feature_sets_to_drop

        self.random_state = random_state

    def load_acs_data(self):

        # download target and source data for the respective state & year
        source_data = ACSDataSource(survey_year=self.source_year,
                                    horizon=self.horizon, survey=self.survey).get_data(states=[self.source_state],
                                                                                       download=True)
        target_data = ACSDataSource(survey_year=self.target_year,
                                    horizon=self.horizon, survey=self.survey).get_data(states=[self.target_state],
                                                                                       download=True)

        # select prediction target
        if self.prediction_target == 'ACSIncome':
            X_source, y_source, _ = ACSIncome.df_to_pandas(source_data)
            X_target, y_target, _ = ACSIncome.df_to_pandas(target_data)
        elif self.prediction_target == 'ACSPublicCoverage':
            X_source, y_source, _ = ACSPublicCoverage.df_to_pandas(source_data)
            X_target, y_target, _ = ACSPublicCoverage.df_to_pandas(target_data)
        elif self.prediction_target == 'ACSEmployment':
            X_source, y_source, _ = ACSEmployment.df_to_pandas(source_data)
            X_target, y_target, _ = ACSEmployment.df_to_pandas(target_data)
        elif self.prediction_target == 'ACSPublicCoverage':
            X_source, y_source, _ = ACSPublicCoverage.df_to_pandas(source_data)
            X_target, y_target, _ = ACSPublicCoverage.df_to_pandas(target_data)
        elif self.prediction_target == 'ACSMobility':
            X_source, y_source, _ = ACSMobility.df_to_pandas(source_data)
            X_target, y_target, _ = ACSMobility.df_to_pandas(target_data)
        elif self.prediction_target == 'ACSTravelTime':
            X_source, y_source, _ = ACSTravelTime.df_to_pandas(source_data)
            X_target, y_target, _ = ACSTravelTime.df_to_pandas(target_data)
        else:
            raise ValueError("Unrecognized prediction target '{}'.".format(self.prediction_target))

        return X_source, y_source, X_target, y_target

    def train_test_split(self, X_source, y_source, X_target, y_target, return_target_mask: bool = False):

        # split
        X_source_train, X_source_test, y_source_train, y_source_test = \
            train_test_split(X_source, y_source, train_size=self.n_source_train, random_state=self.random_state)

        X_target_train, X_target_test, y_target_train, y_target_test = \
            train_test_split(X_target, y_target, train_size=self.n_target_train, random_state=self.random_state)

        # create training dataset out of source & target
        X_train = np.concatenate((X_source_train, X_target_train), axis=0)
        y_train = np.concatenate((y_source_train, y_target_train), axis=0)

        # split remaining target data into validation & test set
        X_target_test, X_target_val, y_target_test, y_target_val = \
            train_test_split(X_target_test, y_target_test, test_size=0.5, random_state=self.random_state)

        if return_target_mask:
            mask = np.concatenate((np.ones(X_source_train.shape[0]), np.zeros(X_target_train.shape[0])), axis=0)
            return X_train, y_train, X_source_test, y_source_test, X_target_val, y_target_val, X_target_test, y_target_test, mask

        return X_train, y_train, X_source_test, y_source_test, X_target_val, y_target_val, X_target_test, y_target_test

    def drop_feature_sets(self, X_source, X_target):

        columns_to_drop = \
            [item for sublist in [FEATURE_SETS[key] for key in self.feature_sets_to_drop] for item in sublist]

        filter_source = X_source.filter(columns_to_drop)
        filter_target = X_target.filter(columns_to_drop)

        X_source = X_source.drop(filter_source, axis=1)
        X_target = X_target.drop(filter_target, axis=1)

        return X_source, X_target

    def load_data(self, return_target_mask: bool = False):

        # load ACS source & target data
        X_source, y_source, X_target, y_target = self.load_acs_data()

        # drop feature sets from default folktables features
        if self.feature_sets_to_drop:
            X_source, X_target = self.drop_feature_sets(X_source, X_target)

        X_source, y_source, X_target, y_target = X_source.to_numpy(), y_source.to_numpy(), X_target.to_numpy(), y_target.to_numpy()

        # split data into train, validation and test set
        return self.train_test_split(X_source, y_source, X_target, y_target, return_target_mask=return_target_mask)
