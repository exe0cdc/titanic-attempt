# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import kaggle
from os import path

@click.group()
def main():
    pass


@click.command()
@click.argument('output_path', type=click.Path())
@click.option('--competition', default='titanic')
def download_data(competition, output_path):
    """ Downloads data file from kaggle titanic data set.
    """
    logger = logging.getLogger(__name__)
    logger.info('Downloading data:')
    logger.info('data set - {}'.format(competition))
    logger.info('data path - {}'.format(output_path))
    outputdir, file_name = path.split(output_path)
    kaggle.api.competition_download_file(competition,file=file_name,path=outputdir)

main.add_command(download_data)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
