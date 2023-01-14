from base.base_predictor import BasePredictor
from data_loaders.bitcoin_price_dataloader import BitcoinPriceDataLoader
from models.register_models import ModelRegistry

import mlflow
import mlflow.keras

import numpy as np

from sklearn.preprocessing import MinMaxScaler
from joblib import load, dump
from pathlib import Path

import datetime


class BitcoinPriceModelPredictor(BasePredictor):
    def __init__(self, config):
        super(BitcoinPriceModelPredictor, self).__init__(config)

    def _load_model(self):

        modelregistry = ModelRegistry()
        loaded_model = mlflow.pyfunc.load_model(modelregistry.get_latest_registered_model_uri(self.config.registered_model_name))

        return loaded_model

    def _load_data(self):
        data_loader = BitcoinPriceDataLoader(self.config)
        self.nscaler_features = int(data_loader.get_train_data()[0].shape[1])
        X, y = data_loader.get_full_data()
        X, _, self.nscaler_features = self._scale_data(X, y)

        dataX = []
        for i in range(len(X) - self.config.trainer.timesteps):
            a = X[i : (i + self.config.trainer.timesteps)]
            dataX.append(a)

        pred_X = np.array(dataX)
        pred_X.reshape(
            pred_X.shape[0],
            pred_X.shape[1],
            pred_X.shape[2],
            1,
        )

        return pred_X

    def _scale_data(self, X_train, y_train):

        train = np.concatenate((X_train, y_train), axis=1)
        scaler = MinMaxScaler().fit(train)
        scaled_train = scaler.transform(train)

        print("Saving artifacts...")
        Path("artifacts").mkdir(exist_ok=True)
        dump(scaler, "artifacts/scaler.pkl")

        self.nscaler_features = scaled_train.shape[1]

        return (
            scaled_train[:, :-1],
            scaled_train[:, -1].reshape(-1, 1),
            self.nscaler_features,
        )

    def _inverse_data(self, X_test_pred):

        scaler = load("artifacts/scaler.pkl")

        trainPredict_dataset_like = np.zeros(
            shape=(len(X_test_pred), self.nscaler_features)
        )
        # put the predicted values in the right field
        trainPredict_dataset_like[:, -1] = X_test_pred[:, -1]
        # inverse transform and then select the right field
        X_test_pred_inverse = scaler.inverse_transform(trainPredict_dataset_like)[:, -1]

        return X_test_pred_inverse

    def predict(self):
        prediction = self.model.predict(self.data)
        prediction = self._inverse_data(prediction)
        mlflow.set_tag(
                "mlflow.runName",
                "prediction-{}".format(datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")),
            )
        mlflow.log_metric("prediction", prediction)

        return prediction

    def get_data(self):
        return self.data
