# -*- coding: utf-8 -*-
import logging
from os import path
from pathlib import Path

import click
from dotenv import find_dotenv, load_dotenv
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

import titanic_src.utils as utils


def replace_title(title):
    """For a string containing a title, return either the same string or
       replace it with a more generic title (e.g. Mlle. - Miss.)
    """
    replacements = {'Mlle.': 'Miss.',
                    'Sir.': 'Royal.',
                    'Dr.': 'Offcr.',
                    'Rev.': 'Offcr.',
                    'Col.': 'Offcr.',
                    'Major.': 'Offcr.',
                    'Mme.': 'Mrs.',
                    'Ms.': 'Miss.',
                    'Countess.': 'Royal.',
                    'Don.': 'Royal.',
                    'Dona.': 'Royal.',
                    'Jonkheer.': 'Master.',
                    'Lady.': 'Royal.',
                    'Capt.': 'Offcr.'}
    if title in replacements:
        return replacements[title]
    else:
        return title


def get_title(name):
    """Return the title substring for a name string in the titanic
       dataset."""
    return [word for word in name.split(' ') if '.' in word][0]


def count_cabins(cabin):
    """Counts the number of cabins that a person has in the titanic
       dataset."""
    if type(cabin) is str:
        return len(cabin.split(' '))
    else:
        return 0


def get_highest_deck(cabin):
    """Returns the highest deck letter from the cabin string in the
       titanic dataset."""
    if type(cabin) is str:
        cabins = cabin.split(' ')
        cabin_letters = [cab[0].upper() for cab in cabins]
        return sorted(cabin_letters)[0]
    else:
        return 'ZZ'


class ExtractFeature(BaseEstimator, TransformerMixin):
    """Apply a arbitrary function on a dataframe column and create a new
       column to save the result.
    """

    def __init__(self, function, on, new):
        self.function = function
        self.on = on
        self.new = new

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        new_X = X.copy()
        new_X[self.new] = new_X[self.on].apply(self.function)
        return new_X


class CollapseUncommon(BaseEstimator, TransformerMixin):
    """Replaces uncommon categories with an 'other' category for any
       given pandas dataframe.
    """

    def __init__(self, on, threshold=4, replace_with='other'):
        self.on = on
        self.threshold = threshold
        self.replace_with = '{}_{}'.format(on, replace_with)

    def fit(self, X, y=None):
        freqs = dict(X[self.on].value_counts())
        self.to_keep = [k for k, v in freqs.items() if v > self.threshold]
        return self

    def transform(self, X):
        new_X = X.copy()
        not_in = ~new_X[self.on].isin(self.to_keep)
        new_X.loc[not_in, self.on] = self.replace_with
        return new_X


class GetFamilySize(BaseEstimator, TransformerMixin):
    """Counts number of family members based on Parch and SibSp."""
    def __init__(self, on=['SibSp', 'Parch']):
        self.on = on

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        new_X = X.copy()
        new_X['FamilySize'] = new_X[self.on].sum(axis=1) + 1
        return new_X


class GetAlone(BaseEstimator, TransformerMixin):
    """Adds a column 'IsAlone' for when FamilySize is 0."""
    def __init(self, on='FamilySize'):
        self.on = on

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        new_X = X.copy()
        new_X['IsAlone'] = new_X['FamilySize'] == 0
        return new_X


def make_add_features_pipeline():
    """Creates a pipeline for adding additional features to the data.



    """
    add_features_pipeline = Pipeline([
        ('get_family_size', GetFamilySize()),
        ('extract_title', ExtractFeature(get_title, 'Name', 'Title')),
        ('get_num_of_cabins',
         ExtractFeature(count_cabins, 'Cabin', 'NumOfCabins')),
        ('get_highest_deck',
         ExtractFeature(get_highest_deck, 'Cabin', 'HighestDeck')),
        ('fix_title', ExtractFeature(replace_title, 'Title', 'Title')),
    ])
    return add_features_pipeline


def make_bucket_pipeline():
    bucket_pipeline = Pipeline([
        ('age_binner', ExtractFeature(make_age_bins, 'Age', 'AgeBin')),
        ('family_size_binner',
         ExtractFeature(make_family_bins, 'FamilySize', 'FamilyBin')),
    ])
    return bucket_pipeline


def make_family_bins(family_size):
    if family_size == 1:
        return 'Alone'
    if family_size <= 3:
        return 'Small'
    if family_size <= 4:
        return 'Kernel'
    else:
        return 'Large'


def get_ticket_pre(ticket):
    ticket = ticket.replace('/', '').replace('.', '')
    pre_suf = ticket.split(' ')
    if len(pre_suf) != 2:
        return 'XXXXX'
    else:
        return pre_suf[0]


def get_ticket_suf(ticket):
    pre_suf = ticket.split(' ')
    if len(pre_suf) != 2:
        return pre_suf[0]
    else:
        return pre_suf[1]


def get_surname(name):
    return name.split(',')[0]


def make_age_bins(age):
    return str(age // 15 * 15).replace('.', '_')


@click.group()
def main():
    pass


@click.command()
@click.option('--pipeline_input_path',
              default=path.join('models', 'new_features_pipeline.pkl'))
@click.argument('input_path', type=click.Path(exists=True))
@click.argument('output_path', type=click.Path())
def transform_new_features(input_path, output_path, pipeline_input_path):
    logger = logging.getLogger(__name__)
    logger.info('Adding new features:')
    logger.info('input data - {}'.format(input_path))
    logger.info('pipeline input path - {}'.format(pipeline_input_path))
    logger.info('output data - {}'.format(output_path))

    data_set = utils.read_csv(input_path, index_col=0)

    add_features_pipeline = utils.load_pipeline(pipeline_input_path)
    transformed_data = add_features_pipeline.transform(data_set)
    utils.save_csv(transformed_data, output_path)


@click.command()
@click.option('--input_path', default=path.join('data', 'raw', 'train.csv'))
@click.argument('pipeline_output_path', type=click.Path())
def fit_new_features(input_path, pipeline_output_path):
    logger = logging.getLogger(__name__)
    logger.info('Fitting additional feature pipeline to training data:')
    logger.info('input data- {}'.format(input_path))
    logger.info('pipeline output path- {}'.format(pipeline_output_path))

    data_set = utils.read_csv(input_path, index_col=0)

    add_features_pipeline = make_add_features_pipeline()
    add_features_pipeline = add_features_pipeline.fit(data_set)
    utils.save_pipeline(add_features_pipeline, pipeline_output_path)


@click.command()
@click.option('--pipeline_input_path',
              default=path.join('models', 'bins_pipeline.pkl'))
@click.argument('input_path', type=click.Path(exists=True))
@click.argument('output_path', type=click.Path())
def transform_bins(input_path, output_path, pipeline_input_path):
    logger = logging.getLogger(__name__)
    logger.info('Adding Bins:')
    logger.info('input data - {}'.format(input_path))
    logger.info('pipeline input path - {}'.format(pipeline_input_path))
    logger.info('output data - {}'.format(output_path))

    data_set = utils.read_csv(input_path, index_col=0)

    bucket_pipeline = utils.load_pipeline(pipeline_input_path)
    transformed_data = bucket_pipeline.transform(data_set)
    utils.save_csv(transformed_data, output_path)


@click.command()
@click.option('--input_path',
              default=path.join('data', 'interim', 'imputed_train.csv'))
@click.argument('pipeline_output_path', type=click.Path())
def fit_bins(input_path, pipeline_output_path):
    logger = logging.getLogger(__name__)
    logger.info('Fitting binning pipeline to training data:')
    logger.info('input data- {}'.format(input_path))
    logger.info('pipeline output path- {}'.format(pipeline_output_path))

    data_set = utils.read_csv(input_path, index_col=0)

    bucket_pipeline = make_bucket_pipeline()
    bucket_pipeline = bucket_pipeline.fit(data_set)
    utils.save_pipeline(bucket_pipeline, pipeline_output_path)


main.add_command(fit_new_features)
main.add_command(transform_new_features)
main.add_command(fit_bins)
main.add_command(transform_bins)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
