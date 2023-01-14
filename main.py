import yaml

from data_producers.bitcoin_price_dataproducer import BitcoinPriceDataProducer
from data_producers.bitcoin_metric_dataproducer import BitcoinMetricDataProducer
from data_producers.bitcoin_training_dataproducer import BitcoinTrainingDataProducer
from optimizers.bitcoin_price_regression_optimizer import (
    BitcoinPriceRegressionOptimizer,
)

from predictors.bitcoin_price_predictor import BitcoinPriceModelPredictor
from models.register_models import ModelRegistry
from models.bitcoin_price_regression_models import *

from dotmap import DotMap

import pandas as pd

import click
import datetime
from datetime import date


def get_config():
    with open(r"config.yaml") as file:
        config = yaml.safe_load(file)

    return config


@click.group()
def cli1():
    pass


@cli1.command()
@click.option(
    "--pull-data",
    help="load saved data or pull new data. Default is False",
    default=False,
    type=bool,
)
def price_data_retrieve(pull_data):

    config = DotMap()
    config.dataproducer.pull_data = pull_data
    config.dataproducer.apikey = get_config()["alphavantage_api_key"]

    print("Create the data producers.")
    price_data_producer = BitcoinPriceDataProducer(config)
    metric_data_producer = BitcoinMetricDataProducer(config)

    df_price_raw = price_data_producer.get_raw_data()
    df_price_processed = price_data_producer.get_processed_data()
    df_metric_raw = metric_data_producer.get_raw_data()
    df_metric_processed = metric_data_producer.get_processed_data()

    print(df_price_raw.head())
    print(df_price_processed.head())
    print(df_metric_raw.head())
    print(df_metric_processed.head())


@click.group()
def cli2():
    pass


@cli2.command()
def price_training_data_create():

    config = DotMap()
    config.dataproducer.pull_data = True
    config.dataproducer.end_timestamp = pd.Timestamp.now()

    print("Create the data producers.")
    training_data_producer = BitcoinTrainingDataProducer(config)

    df_training_data = training_data_producer.get_processed_data()

    print(df_training_data.head())


@click.group()
def cli3():
    pass


@cli3.command()
@click.option(
    "--model-type", help="LSTM, CNN. Default uses all", default="LSTM", type=str
)
@click.option(
    "--pull-data",
    help="load saved data or pull new data. Default is False",
    default=False,
    type=bool,
)
@click.option(
    "--n-trials", help="number of study trials. Default is 2", default=2, type=int
)
@click.option(
    "--timesteps",
    help="Number of timesteps, if not set the optimizer will select the timesteps. Default is 0",
    default=0,
    type=int,
)
@click.option(
    "--prediction-length",
    help="The number of predictions in the test set. The size of the test set will be prediction length + timesteps + 1. Be careful not to go above the total size of data set. Default is 5",
    default=5,
    type=int,
)
@click.option("--version", help="version of study. Default is 1", default=1, type=int)
@click.option(
    "--experiment-name",
    help="Name of the experiment",
    default="crypto-reg-experiment",
    type=str,
)
def price_regression_optimize(
    model_type,
    pull_data,
    n_trials,
    timesteps,
    prediction_length,
    version,
    experiment_name,
):

    config = DotMap()
    config.pull_data = pull_data
    config.dataloader.start_timestamp = pd.Timestamp(2021, 1, 10)
    config.dataloader.end_timestamp = pd.Timestamp.now()
    config.trainer.model_type = model_type
    config.trainer.model_task = "regression"
    config.optimizer.n_trials = n_trials
    config.optimizer.version = version
    config.optimizer.experiment_name = experiment_name
    config.trainer.model_name_prefix = "./artifacts/reg"
    config.trainer.timesteps = timesteps
    if timesteps == 0:
        config.trainer.timestep_set = False
    else:
        config.trainer.timestep_set = True
    config.trainer.prediction_length = prediction_length

    optimizer = BitcoinPriceRegressionOptimizer(config)

    optimizer.optimize()


@click.group()
def cli4():
    pass


@cli4.command()
@click.option(
    "--date-end",
    help="Final date for time series prior to prediction date",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    default=str(date.today()),
)
@click.option(
    "--registered-model-name",
    help="Registered model name without version. The latest version will be selected from the registry",
    default="bitcoin-regression",
    type=str,
)
@click.option(
    "--timesteps",
    help="Number of timesteps the model was trained with. Default is 10",
    default=10,
    type=int,
)
def price_predict(date_end, registered_model_name, timesteps):

    config = DotMap()
    config.dataloader.start_timestamp = date_end - datetime.timedelta(days=timesteps)
    config.dataloader.end_timestamp = date_end
    config.registered_model_name = registered_model_name
    config.trainer.timesteps = timesteps
    predictor = BitcoinPriceModelPredictor(config)
    prediction = predictor.predict()
    print(prediction)
    print("done")


@click.group()
def cli5():
    pass


@cli5.command()
@click.option(
    "--model-name",
    help="Model name",
    type=str,
    default="bitcoin-regression",
)
@click.option(
    "--experiment-id",
    help="Experiment id",
    type=str,
    default="2",
)
def register_best_model(model_name, experiment_id):

    modelregistry = ModelRegistry()
    print(modelregistry.register_best_model(model_name, experiment_id))


cli = click.CommandCollection(sources=[cli1, cli2, cli3, cli4, cli5])

if __name__ == "__main__":
    cli()
