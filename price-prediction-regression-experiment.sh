#!/bin/bash
echo "Set Environment Variables"
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
export MLFLOW_EXPERIMENT_NAME="crypto-reg-experiment"
echo "Pull Data"
mlflow run . -e price-data-retrieve -P pull-data=True --env-manager=local
echo "Train model using hyperparameter optimization"
mlflow run . -P model_type='CNN' -P n_trials=4 --env-manager=local --experiment-name="crypto-reg-experiment"
echo "Register the best model"
mlflow run . -e register-best-model -P model-name="bitcoin-regression" -P experiment-id="2" --env-manager=local
echo "Use the best model to predict the price of Bitcoin"
mlflow run . -e price-predict -P date-end="2023-01-02" -P registered-model-name="bitcoin-regression" -P timesteps=10 --env-manager=local

