# Bitcoin Price Prediction with MLFlow and Feast

## Overview

This repository contains code that predicts the price of Bitcoin using various blockchain features combined from two sources. The project includes various deep learning model architectures (primarily CNN and LSTM) to predict Bitcoin prices using different size time steps. The full machine learning lifecycle is managed by [MLFlow](https://mlflow.org/) and [Feast](https://feast.dev/). 

Feast feature views for Bitcoin price and metric data are combined and pulled for training and batch model scoring. MLFlow Projects is used to orchestrate each step in the process; ETL, model training using hyperparameter optimization, model selection and deployment, and model scoring. [Optuna](https://optuna.org/) is used to perform hyperparameter optimization, which is integrated with MLFlow Tracking using Optuna's callback integration. The best performing model is registered programatically and made available for scoring inside the MLFlow Model Registry.

## Project Structure

All methods can be executed via the command line using `mlflow run .` or `main.py`. Using `mlflow run .` ensures all features of MLFlow are integrated properly.

```nohighlight
├── artifacts                               <- tokenizers, scalers, and optuna studies 
├── base                                    <- base classes 
├── data                                    <- bitcoin metric and price data
    ├── feature_repo                        <- feast feature repository
      ├── data                              <- feast data home
    ├── mlflow_backend                      <- backend for mlflow server 
    ├── processed                           <- cleansed, transformed, scaled data
    ├── raw                                 <- raw price and metric data from apis 
├── data_loaders                            <- classes that load preprocessed data and splits it 
├── data_producers                          <- classes that load data from apis and preprocesses data
├── mlruns                                  <- default MLFlow artifact folder created automatically
├── models                                  <- price models
├── optimizers                              <- optuna optimizers for price models
├── predictors                              <- model prediction classes
├── trainers                                <- model training classes
├── bitcoin-price-prediction-eda.ipynb      <- Jupyter notebook for project eda
├── main.py                                 <- command line methods
├── MLproject                               <- MLflow Project configuration
├── price-prediction-experiments.sh         <- script for price prediction experiments
├── conda.yaml                              <- conda environment requirements
```

## Data Sources

### Alpha Vantage API

Daily bitcoin price data, open, high, low, close, volume, and market, are pulled using the Alpha Vantage API. A free API key must be obtained and set the value to an alphavantage_api_key key in a config.yaml file in the root of the project.

### Blockchain.com API

The blockchain metrics are obtained from the blockchain.com API. More information about the information is available at https://www.blockchain.com/charts. An API key is not required.


## Envrionment Setup

The following are required to run the machine learning pipelines locally.

### Setup MLFlow Environment in your command-line

These environment variables need to be set prior to using `mlflow run .` commands.
```
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
export MLFLOW_EXPERIMENT_NAME="crypto-reg-experiment"
```

### Running MLFlow Server

This command starts MLFlow server locally. See MLFlow documentation for production grade deployments options.

```bash
crypto-price-prediction % mlflow server --backend-store-uri sqlite:///data/mlflow_backend/mlflow.db --host 0.0.0.0
```

### Feast Setup

These commands will create or update the Feast feature repository. Feast teardowm can also be run to reset the repository.

```bash
cd ./data/feature_repo
feast apply
```

## Model Pipelines

The command line methods below represent the steps in the model pipeline for retrieving the bitcoin blockchain metrics and price data, training the regression Bitcoin price prediction models in the context of an experiment, registering the best performing model, and scoring against the registered model. The MLproject file defines the entry points into `main.py` so the options are the same and only listed once in each step of the pipeline. You can run all the commands below together using `price-prediction-regression-experiment.sh`

### Retrieve Data

These commands will pull data from Blockchain.com and Alpha Vantage when --pull-data is True and save them as parquet files. They are used as Feast file sources for the feature views and feature services. See Feast [docs](https://docs.feast.dev/getting-started/concepts) for a review of key concepts.

```
Usage: mlflow run . -e price-data-retrieve -P pull-data=True --env-manager=local

Options:
  --pull-data BOOLEAN  load saved data or pull new data. Default is False
```

OR 

```
Usage: main.py price-data-retrieve [OPTIONS]

Options:
  --pull-data BOOLEAN  load saved data or pull new data. Default is False
  --help               Show this message and exit.
```

### Price Regression Model Training with Hyperparameter Optimization

These commands create an Optuna study with the number of trials specified. Model metrics and parameters are logged, and models are saved for each trial as a run under the specified experiment in MLFlow. Subsequent command execution can use the same experiment. If a new experiment is desired, be sure to modify the environment variable for experiment name as described above. The data is loaded using Feast.

```
mlflow run . -P model_type='LSTM' -P n_trials=4 --env-manager=local --experiment-name="crypto-reg-experiment"
```

OR 

```
Usage: main.py price-regression-optimize [OPTIONS]

Options:
  --model-type TEXT            LSTM, CNN. Default uses all
  --pull-data BOOLEAN          load saved data or pull new data. Default is
                               False
  --n-trials INTEGER           number of study trials. Default is 2
  --timesteps INTEGER          Number of timesteps, if not set the optimizer
                               will select the timesteps. Default is 0
  --prediction-length INTEGER  The number of predictions in the test set. The
                               size of the test set will be prediction length
                               + timesteps + 1. Be careful not to go above the
                               total size of data set. Default is 5
  --version INTEGER            version of study. Default is 1
  --experiment-name TEXT       Name of the experiment
  --help                       Show this message and exit.
```

## Register Best Model in MLFlow Model Registry

These commands select the best model based on the model in an experiment that had the best MAPE (Mean Absolute Percentage Error) and registers in for scoring.

```
mlflow run . -e register-best-model -P model-name="bitcoin-regression" -P experiment-id="2" --env-manager=local
```

OR

```
Usage: main.py register-best-model [OPTIONS]

Options:
  --model-name TEXT     Model name
  --experiment-id TEXT  Experiment id
  --help                Show this message and exit.
```

## Price Prediction

This command will load the latest version of a registered model and predict the Bitcoin price for the next day. The prediciton is printed in the console and logged in the experiment as a run.

```
mlflow run . -e price-predict -P date-end="2023-01-02" -P registered-model-name="bitcoin-regression" -P timesteps=10 --env-manager=local
```

OR

```
Usage: main.py price-predict [OPTIONS]

Options:
  --date-end [%Y-%m-%d]         Final date for time series prior to prediction
                                date
  --registered-model-name TEXT  Registered model name without version. The
                                latest version will be selected from the
                                registry
  --timesteps INTEGER           Number of timesteps the model was trained
                                with. Default is 10
  --help                        Show this message and exit.
```

## Tools

### Running Optuna Dashboard

```bash
optuna-dashboard sqlite:///artifacts/db.sqlite3    
```

### Running the Feast UI

```bash
cd ./data/feature_repo
feast ui
```