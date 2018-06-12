import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, StratifiedKFold

from titanic_src.utils import interpolate_array


def plot_learning_curves_cv(model, X, y, metric=None, cv=5, increments=False):
    """Plots learning curves for a given model/pipeline using cross
       validation.

    An average score is calculated for each point using cross
    validation. Increments smaller than the length of the input data
    can be used to speed up calculation of scores.

    :param model: a sklearn estimator to test
    :param X: training features (pandas dataframe)
    :param y: training targets (np array or pd dataframe)
    :param metric: metric to use (defaults to `sklearn.metrics.accuracy`)
    :param cv: (Int, optional) default 5
    :param increments: Bool or int (number of increments to split data)
    """
    if not metric:
        metric = accuracy_score

    skf = StratifiedKFold(n_splits=cv)
    splits = list(skf.split(X=X, y=y))

    max_train_len = np.max([len(split[0]) for split in splits])

    final_train_errors, final_val_errors = [], []
    for train_index, val_index in splits:
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        train_errors, val_errors = [], []

        if increments:
            increment_size = len(X_train) // increments
        else:
            increment_size = 1

        for m in list(range(increment_size + 1, len(X_train), increment_size)):
            model.fit(X_train[:m], y_train[:m])
            y_train_predict = model.predict(X_train[:m])
            y_val_predict = model.predict(X_val)
            train_errors.append(metric(y_train_predict, y_train[:m]))
            val_errors.append(metric(y_val_predict, y_val))
        final_train_errors.append(train_errors)
        final_val_errors.append(val_errors)

    train_interpolation = [
        interpolate_array(np.array(train_error), max_train_len) for train_error
        in final_train_errors]
    val_interpolation = [interpolate_array(np.array(val_error), max_train_len)
                         for val_error in final_val_errors]
    final_train_errors = np.mean(train_interpolation, axis=0)
    final_train_std = np.std(train_interpolation, axis=0)
    final_val_errors = np.mean(val_interpolation, axis=0)
    final_val_std = np.std(val_interpolation, axis=0)

    plt.plot(final_train_errors, color='red', linewidth=2, label="train")
    plt.plot(final_train_errors + final_train_std, color='red', linestyle=':',
             linewidth=2, )
    plt.plot(final_train_errors - final_train_std, color='red', linestyle=':',
             linewidth=2, )
    plt.plot(final_val_errors, color='blue', linewidth=2, label="val")
    plt.plot(final_val_errors + final_val_std, color='blue', linestyle=':',
             linewidth=2)
    plt.plot(final_val_errors - final_val_std, color='blue', linestyle=':',
             linewidth=2)
    plt.fill_between(x=range(len(final_train_errors)),
                     y1=final_train_errors + final_train_std,
                     y2=final_train_errors - final_train_std, alpha=0.5,
                     color='red')
    plt.fill_between(x=range(len(final_val_errors)),
                     y1=final_val_errors + final_val_std,
                     y2=final_val_errors - final_val_std, alpha=0.5,
                     color='blue')
    plt.legend()
    plt.ylim(0.6, 1)


def plot_learning_curves(model, X, y, metric=None, increments=False, ):
    """Plots learning curves for a given model/pipeline.

    Increments smaller than the length of the input data can be used
    to speed up calculation of scores.

    :param model: a sklearn estimator to test
    :param X: training features (pandas dataframe)
    :param y: training targets (np array or pd dataframe)
    :param metric: metric to use (defaults to `sklearn.metrics.accuracy`)
    :param increments: Bool or int (number of increments to split data)
    """
    if not metric:
        metric = accuracy_score

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    train_errors, val_errors = [], []
    if increments:
        increment_size = len(X_train) // increments
    else:
        increment_size = 1
    for m in list(range(increment_size + 1, len(X_train), increment_size)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(metric(y_train_predict, y_train[:m]))
        val_errors.append(metric(y_val_predict, y_val))

    plt.plot(train_errors, "r-+", linewidth=2, label="train")
    plt.plot(val_errors, "b-", linewidth=3, label="val")
    plt.legend()
    plt.ylim(0.6, 1)
