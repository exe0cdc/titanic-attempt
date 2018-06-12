import datetime
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.special import binom
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, \
    cross_val_score
from sklearn.pipeline import FeatureUnion, Pipeline

import titanic_src.utils as utils
from titanic_src.features.transform_features import DataFrameSelector


class ModelTester(object):
    """
    A class that allows for easily constructing and testing different classifier pipelines.
    """

    def __init__(self, classifier, num_columns, cat_columns):
        self.classifier = classifier
        self.num_columns = num_columns
        self.cat_columns = cat_columns
        self.model_pipeline = self._make_model_pipeline()
        self.base_params = {}
        self.additional_actions = [{'ad_action': [None]}]

    def _make_model_pipeline(self):
        num_columns = self.num_columns
        cat_columns = self.cat_columns
        classifier = self.classifier

        num_pipe = Pipeline([
            ('num_selector', DataFrameSelector(num_columns)),
            ('scaler', None),
        ])

        union = FeatureUnion([
            ('num_pipe', num_pipe),
            ('cat_selector', DataFrameSelector(cat_columns))
        ])

        clf_pipeline = Pipeline([
            ('union', union),
            ('ad_action', None),
            ('clf', classifier)
        ])
        return clf_pipeline

    def get_clf_param_options(self):
        return sorted(
            list(self.model_pipeline.steps[-1][1].get_params().keys()))

    @classmethod
    def _fix_dict(cls, dict_, prefix):
        new_dict = {}
        for k, v in dict_.items():
            new_dict['{}{}'.format(prefix, k)] = v
        return new_dict

    def set_scalers(self, scaler_list):
        self.base_params['union__num_pipe__scaler'] = scaler_list

    def set_clf_params(self, params):
        new_dict = self._fix_dict(params, 'clf__')
        self.base_params.update(new_dict)

    def add_additional_action(self, action, action_params=None,
                              clear_previous=False):
        new_action = {'ad_action': [action]}
        # new_dict = self._fix_dict(action_params, 'ad_action__')
        # new_action.update(new_dict)
        if clear_previous:
            self.additional_actions = []
        self.additional_actions.append(new_action)

    @staticmethod
    def union_dicts(dict_1, dict_2):
        new_dict = {}
        new_dict.update(dict_1)
        new_dict.update(dict_2)
        return new_dict

    def _make_param_grid(self):
        return [self.union_dicts(ad_action, self.base_params) for ad_action in
                self.additional_actions]

    def do_grid_search(self, X, y, repeats=5, scoring='accuracy', cv=3,
                       n_jobs=-1, *args, **kwargs):
        param_grid = self._make_param_grid()

        skf = StratifiedKFold(n_splits=cv)
        best_models = {}
        verbose = True
        print('Starting {} repeats of grid search procedure'.format(repeats))
        for i in range(repeats):
            if i != 0:
                verbose = False
            grid_result = GridSearchCV(self.model_pipeline,
                                       param_grid=param_grid, scoring=scoring,
                                       verbose=verbose,
                                       cv=skf, n_jobs=n_jobs, iid=False, *args,
                                       **kwargs).fit(X, y)
            if str(grid_result.best_params_) not in best_models:
                best_models[str(grid_result.best_params_)] = [
                    [grid_result.best_score_], 1, grid_result.best_estimator_,
                    grid_result.best_params_]
            else:
                best_models[str(grid_result.best_params_)][0].append(
                    grid_result.best_score_)
                best_models[str(grid_result.best_params_)][1] += 1
            print('Repeat {} of {} done'.format(i + 1, repeats))

        cut_string = lambda x: str(x).replace('\n', '').replace(' ', '')[:30]

        if type(grid_result.param_grid) is list:
            main_set = set()
            for each in grid_result.param_grid:
                main_set = main_set.union(set(each.keys()))
            actual_columns = list(main_set)
        else:
            actual_columns = sorted(list(grid_result.param_grid.keys()))
        df_columns = ['score', 'count'] + [utils.get_last_two(each) for each in
                                           actual_columns] + ['model',
                                                              'params']

        results = []
        for k, v in best_models.items():
            current_res = [np.mean(v[0]), v[1]]
            param_vals = [cut_string(v[3].get(j)) for j in actual_columns]
            current_res += param_vals
            current_res += [v[2], v[3]]
            results.append(current_res)

        best_df = pd.DataFrame(results, columns=df_columns).sort_values(
            by=['score', 'count'],
            ascending=False).reset_index().drop('index',
                                                axis=1)
        self.best_df_ = best_df
        self.best_model_ = best_df['model'].iloc[0]


def find_best_voter(estimators, X, y, voting='hard', scoring='accuracy',
                    calc_diff=True, cv=5):
    """Finds the best combination from a list of estimators for use as a
       voting classifier.

    :param estimators: list of estimators
    :param X: training features
    :param y: training targets
    :param voting: voting (either 'hard' or 'soft')
    :param scoring: default metric is 'accuracy'
    :param calc_diff: if True calculates difference between training predictions and validation predictions
    :param cv: number of splits for cross validation
    :return: best_score (most accurate score), best_combo (best estimator combo), combo_score_std (dict with estimator combos as keys, and lists with scores and stdevs as values)
    """
    start_time = datetime.datetime.now()
    best_score = 0
    num_ests = len(estimators)
    num_of_combos = sum((binom(num_ests, i)) for i in range(2, num_ests + 1))
    count = 0
    per_divider = 10
    per_div_step = per_divider
    per_increment = num_of_combos / per_divider
    per_target = per_increment
    combo_score_std = {}

    print('Starting search for best estimator combination for voting.')
    print('Num of estimators : {}'.format(num_ests))
    print('Num of combinations : {}'.format(num_of_combos))
    for i in range(0, num_ests - 1):
        for each in combinations(estimators, num_ests - i):
            voting_clf = VotingClassifier(estimators=each,
                                          voting=voting)
            cross_val_raw = cross_val_score(voting_clf, X, y, cv=cv,
                                            scoring=scoring, n_jobs=-1)
            current_score = np.mean(cross_val_raw)
            current_std = np.std(cross_val_raw)
            if calc_diff:
                current_diff = train_val_diff(voting_clf, X, y, cv=cv)
            else:
                current_diff = None
            key = str([k for k, _ in each]).replace(' ', '')
            combo_score_std[key] = [current_score, current_std, current_diff]
            if current_score > best_score:
                best_score = current_score
                best_combo = each

            if count == int(np.floor(per_target)):
                print('{} % complete; {} elapsed '.format(per_div_step, str(
                    datetime.datetime.now() - start_time)))
                per_target += per_increment
                per_div_step += per_divider
            count += 1

    print('Best score: {}'.format(best_score))
    return best_score, best_combo, combo_score_std


def train_val_diff(model, X, y, metric=None, cv=5):
    """Calculates the difference in training and validation scores of a
       model using cross validation.


    :param model: estimator to test
    :param X: training features
    :param y: training targets
    :param metric: defaults to `sklearn.metrics.accuracy_score`
    :param cv: defaults to 5
    :return: mean_diff (mean difference between training and validation)
    """
    if not metric:
        metric = accuracy_score

    if type(y) != pd.Series:
        y = pd.Series(y)

    skf = StratifiedKFold(n_splits=cv)
    splits = list(skf.split(X=X, y=y))

    train_errors, val_errors = [], []
    for train_index, val_index in splits:
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        model.fit(X_train, y_train)

        y_train_predict = model.predict(X_train)
        y_val_predict = model.predict(X_val)

        y_train_error = metric(y_train, y_train_predict)
        y_val_error = metric(y_val, y_val_predict)

        train_errors.append(y_train_error)
        val_errors.append(y_val_error)

    mean_diff = np.mean(train_errors) - np.mean(val_errors)
    return mean_diff


def score_df(combo_score_std, lowest_precision=2, highest_precision=2):
    """Converts to combo_score_std from find_best_voter to a pandas
       dataframe.
    """
    df = pd.DataFrame.from_dict(combo_score_std, orient='index').reset_index()
    df.columns = ['estimator', 'score', 'stdev', 'diff']
    df = pd.DataFrame(df)

    df['lowest'] = df['score'] - df['stdev']
    df['lowest'] = np.round(df['lowest'], lowest_precision)

    df['highest'] = df['score'] + df['stdev']
    df['highest'] = np.round(df['highest'], highest_precision)

    return df


def get_estimators(selected_row):
    """Extracts the estimators from a row of the output of `score_df`"""
    return selected_row['estimator'].replace("'", '').replace('[', '').replace(
        ']', '').split(',')
