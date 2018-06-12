import logging
from os import path
from pathlib import Path

import click
from dotenv import find_dotenv, load_dotenv

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.preprocessing import CategoricalEncoder, QuantileTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.impute import MICEImputer
from titanic_src import utils
import numpy as np
import pandas as pd

CATEGORICAL_COLS = ['Title', 'Sex', 'Embarked', 'HighestDeck', 'Pclass']
NUMERICAL_COLS = ['Age', 'SibSp', 'Fare', 'FamilySize', 'NumOfCabins', 'Parch']


class DataFrameSelector(BaseEstimator, TransformerMixin):
    """
    Selects columns from a dataframe.
    """
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names]


class CustomImputer(BaseEstimator, TransformerMixin):
    """
    Imputer that allows for imputing most common categorical values.
    """
    # copied from https://stackoverflow.com/questions/25239958/impute-categorical-missing-values-in-scikit-learn
    def __init__(self, strategy='mean', filler='NA'):
        self.strategy = strategy
        self.fill = filler

    def fit(self, X, y=None):
        if self.strategy in ['mean', 'median']:
            if not all(X.dtypes == np.number):
                raise ValueError('dtypes mismatch np.number dtype is \
                                  required for ' + self.strategy)
        if self.strategy == 'mean':
            self.fill = X.mean()
        elif self.strategy == 'median':
            self.fill = X.median()
        elif self.strategy == 'mode':
            self.fill = X.mode().iloc[0]
        elif self.strategy == 'fill':
            if type(self.fill) is list and type(X) is pd.DataFrame:
                self.fill = dict([(cname, v) for cname, v in zip(X.columns, self.fill)])
        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)


class ColumnSelector(BaseEstimator, TransformerMixin):
    """
    Selects columns from numpy array.
    """
    def __init__(self, col):
        self.col = col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if len(self.col) == 1:
            return X[:, self.col].reshape(-1, 1)
        else:
            return X[:, self.col]


class SelectiveAction(BaseEstimator, TransformerMixin):
    """
    Applies a transformation to a selection of columns of the original data.
    """
    def __init__(self, col, action):
        self.col = col
        self.action = action

    def fit(self, X, y=None):
        if len(self.col) == 1:
            fit_X = X[:, self.col].reshape(-1, 1)
        else:
            fit_X = X[:, self.col]

        self.action = self.action.fit(fit_X)
        return self

    def transform(self, X):
        ar_shape = X.shape
        other_cols = [col for col in range(ar_shape[1]) if col not in self.col]
        if len(self.col) == 1:
            trans_X = X[:, self.col].reshape(-1, 1)
        else:
            trans_X = X[:, self.col]

        if len(self.col) == 1:
            rem_X = X[:, other_cols].reshape(-1, 1)
        else:
            rem_X = X[:, other_cols]

        trans_X = self.action.transform(trans_X)
        return np.hstack([trans_X, rem_X])


def find_categories(X, cols):
    unique_cats = [list(np.unique(X[~X[col].isnull()][col].as_matrix())) for col in cols]
    return unique_cats


def inverse_func(X, transformer):
    """Performs an inverse_transform of a particular transformer on X"""
    return transformer.inverse_transform(X)


def make_impute_pipeline():
    categorical_cols = CATEGORICAL_COLS
    numerical_cols = NUMERICAL_COLS

    categorical_pre = Pipeline([
        ('selector', DataFrameSelector(categorical_cols)),
        ('impute', CustomImputer(strategy='mode')),
    ])

    categorical_pipeline = Pipeline([
        ('categorical_pre', categorical_pre),
        ('encoder', CategoricalEncoder(encoding='onehot-dense')),
    ])

    num_init_quantile_transformer = QuantileTransformer(output_distribution='normal')

    numerical_pipeline = Pipeline([
        ('selector', DataFrameSelector(numerical_cols)),
        ('scale', num_init_quantile_transformer),
    ])

    combined_features = FeatureUnion([
        ('numerical_pipeline', numerical_pipeline),
        ('cat_ordinal_pipeline', categorical_pipeline),
    ])

    mice_pipeline = Pipeline([
        ('combined_features', combined_features),
        ('mice_impute', MICEImputer(verbose=True)),
    ])

    impute_pipeline = Pipeline([
        ('mice_pipeline', mice_pipeline),
        ('inverse_qt', SelectiveAction(col=list(range(len(numerical_cols))),
                                       action=FunctionTransformer(inverse_func,
                                                                  kw_args={'transformer':
                                                                               num_init_quantile_transformer}))),
        ('numerical_selection', ColumnSelector(range(len(numerical_cols))))
    ])

    final_pipeline = FeatureUnion([
        ('impute_pipeline', impute_pipeline),
        ('categorical_pre', categorical_pre)
    ])

    return final_pipeline


def make_encode_pipeline():
    numerical_cols = NUMERICAL_COLS
    categorical_cols = CATEGORICAL_COLS + ['AgeBin', 'FamilyBin']

    cat_pipe = Pipeline([
        ('cat_selector', DataFrameSelector(categorical_cols)),
        ('cat_encoder', CategoricalEncoder('onehot-dense')),
    ])
    final_pipe = FeatureUnion([
        ('num_selector', DataFrameSelector(numerical_cols)),
        ('cat_pipe', cat_pipe),
    ])
    return final_pipe


@click.command()
@click.option('--input_path', default=path.join('data', 'interim', 'added_features_train.csv'))
@click.argument('pipeline_output_path', type=click.Path())
def fit_imputer(input_path, pipeline_output_path):
    logger = logging.getLogger(__name__)
    logger.info('Fitting impute pipeline to training data:')
    logger.info('input data- {}'.format(input_path))
    logger.info('pipeline output path- {}'.format(pipeline_output_path))

    data_set = utils.read_csv(input_path, index_col=0)

    impute_encode_pipeline = make_impute_pipeline()
    impute_encode_pipeline = impute_encode_pipeline.fit(data_set)
    utils.save_pipeline(impute_encode_pipeline, pipeline_output_path)


@click.command()
@click.option('--pipeline_input_path', default=path.join('models', 'impute_pipeline.pkl'))
@click.argument('input_path', type=click.Path(exists=True))
@click.argument('output_path', type=click.Path())
@click.option('--add_target', is_flag=True, default=False)
@utils.unclick
def transform_imputer(input_path, output_path, pipeline_input_path, add_target):
    logger = logging.getLogger(__name__)
    logger.info('Imputing missing values:')
    logger.info('input data- {}'.format(input_path))
    logger.info('pipeline input path - {}'.format(pipeline_input_path))
    logger.info('output data - {}'.format(output_path))

    data_set = utils.read_csv(input_path, index_col=0)

    impute_encode_pipeline = utils.load_pipeline(pipeline_input_path)

    columns = NUMERICAL_COLS + CATEGORICAL_COLS

    imputed_encoded_data = pd.DataFrame(impute_encode_pipeline.transform(data_set), columns=columns,
                                        index=data_set.index)
    if add_target:
        logger.info('adding target to data set:')
        try:
            survived = data_set['Survived']
            imputed_encoded_data['Survived'] = survived
            logger.info('success!')
        except KeyError as ke:
            logger.info('failed - column {} not found in data set'.format(ke.args[0]))

    utils.save_csv(imputed_encoded_data, output_path)


@click.command()
@click.argument('input_path', type=click.Path(exists=True))
@click.argument('output_path', type=click.Path())
@click.option('--add_target', is_flag=True, default=False)
@utils.unclick
def fit_transform_encoder(input_path, output_path, add_target):
    logger = logging.getLogger(__name__)
    logger.info('Encoding categorical features:')
    logger.info('input data- {}'.format(input_path))
    logger.info('output data - {}'.format(output_path))

    data_set = utils.read_csv(input_path, index_col=0)

    encode_pipeline = make_encode_pipeline()
    encode_pipeline.fit(data_set)
    categorical_cols = list(np.concatenate(encode_pipeline.transformer_list[1][1].steps[1][1].categories_))
    numerical_cols = NUMERICAL_COLS
    columns = numerical_cols + categorical_cols

    encoded_data = pd.DataFrame(encode_pipeline.transform(data_set), columns=columns,
                                index=data_set.index)
    if add_target:
        logger.info('adding target to data set:')
        try:
            survived = data_set['Survived']
            encoded_data['Survived'] = survived
            logger.info('success!')
        except KeyError as ke:
            logger.info('failed - column {} not found in data set'.format(ke.args[0]))

    utils.save_csv(encoded_data, output_path)


@click.group()
def main():
    pass


main.add_command(transform_imputer)
main.add_command(fit_imputer)
main.add_command(fit_transform_encoder)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
