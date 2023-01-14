from base.base_optimizer import BaseOptimizer
from models.bitcoin_price_regression_models import *
from data_loaders.bitcoin_price_dataloader import BitcoinPriceDataLoader
from trainers.bitcoin_price_trainer import BitcoinPriceModelTrainer
from keras.backend import clear_session
import tensorflow as tf
from tensorflow import keras

import optuna
from optuna.integration.mlflow import MLflowCallback
import joblib

import mlflow
import mlflow.keras

import datetime


class BitcoinPriceRegressionOptimizer(BaseOptimizer):
    def __init__(self, config):
        super(BitcoinPriceRegressionOptimizer, self).__init__(config)
        self.study = optuna.create_study(
            storage="sqlite:///artifacts/db.sqlite3",
            study_name=config.optimizer.experiment_name,
            direction="minimize",
            load_if_exists=True,
        )
        mlflow.set_experiment(config.optimizer.experiment_name)

    def optimize(self):

        mlflc = MLflowCallback(
            tracking_uri="http://127.0.0.1:5000",
            metric_name="MAPE",
            create_experiment=False,
            mlflow_kwargs={"nested": True},
        )

        @mlflc.track_in_mlflow()
        def _objective(trial):
            return_val = 0

            clear_session()
            mlflow.set_tag(
                "mlflow.runName",
                "{}-{}".format(trial.number, self.config.trainer.model_type),
            )

            data_loader = BitcoinPriceDataLoader(self.config)
            self.config.trainer.nfeatures = int(
                data_loader.get_train_data()[0].shape[1]
            )

            if self.config.trainer.model_type == "":
                self.config.trainer.model_type_current = trial.suggest_categorical(
                    "model_type",
                    [
                        "LSTM",
                        "CNN",
                        "GRU",
                        "CNNLSTM",
                        "Attention",
                        "LSTM_Encoder_Decoder",
                        "TCN",
                        "ConvLSTM2D_Encoder_Decoder",
                    ],
                )
            else:
                self.config.trainer.model_type_current = self.config.trainer.model_type

            trial.set_user_attr("model_type", self.config.trainer.model_type_current)

            self.config.trainer.verbose_training = 0
            self.config.trainer.validation_split = 0.1

            if self.config.trainer.timestep_set == False:
                self.config.trainer.timesteps = trial.suggest_categorical(
                    "timesteps", [5, 10, 20, 30]
                )
            else:
                print(self.config.trainer.timesteps)
                self.config.trainer.timesteps = trial.suggest_categorical(
                    "timesteps", [self.config.trainer.timesteps]
                )

            self.config.trainer.batch_size = trial.suggest_categorical(
                "batch_size", [8, 16, 32]
            )

            mlflow.log_param("model_type", self.config.trainer.model_type_current)

            self.config.trainer.num_epochs = trial.suggest_categorical(
                "epochs", [10, 20, 50, 100]
            )
            mlflow.log_param("epochs", self.config.trainer.num_epochs)

            self.config.trainer.activation_function = trial.suggest_categorical(
                "activation_function", ["tanh", "relu"]
            )

            self.config.trainer.optimizer_name = trial.suggest_categorical(
                "optimizer", ["adam"]
            )
            mlflow.log_param("optimizer", self.config.trainer.optimizer_name)

            if self.config.trainer.optimizer_name == "adam":
                adam_lr = trial.suggest_loguniform("adam_lr", 1e-5, 1e-1)
                mlflow.log_param("adam_lr", adam_lr)
                optimizer = keras.optimizers.Adam(learning_rate=adam_lr)
            else:
                sgd_lr = trial.suggest_loguniform("sgd_lr", 1e-5, 1e-1)
                mlflow.log_param("sgd_lr", sgd_lr)
                sgd_momentum = trial.suggest_loguniform("sgd_momentum", 1e-5, 1e-1)
                mlflow.log_param("sgd_momentum", sgd_momentum)
                sgd_nesterov = trial.suggest_categorical("sgd_nesterov", [False, True])
                mlflow.log_param("sgd_nesterov", sgd_nesterov)
                optimizer = keras.optimizers.SGD(
                    lr=sgd_lr,
                    momentum=sgd_momentum,
                    nesterov=sgd_nesterov,
                    clipvalue=0.5,
                )

            self.config.trainer.model_name = "{}-tmp-{}.h5".format(
                self.config.trainer.model_name_prefix, trial.number
            )

            print("Create the model.")
            if self.config.trainer.model_type_current == "LSTM":
                self.config.trainer.units = trial.suggest_categorical(
                    "units", [16, 32, 64]
                )
                mlflow.log_param("units", self.config.trainer.units)
                model = BitcoinPriceLSTMRegressionModel(optimizer, self.config)
            elif self.config.trainer.model_type_current == "GRU":
                self.config.trainer.units = trial.suggest_categorical("units", [32, 64])
                mlflow.log_param("units", self.config.trainer.units)
                model = BitcoinPriceGRURegressionModel(optimizer, self.config)
            elif self.config.trainer.model_type_current == "CNN":
                model = BitcoinPriceCNNRegressionModel(optimizer, self.config)
            elif self.config.trainer.model_type_current == "CNNLSTM":
                self.config.trainer.filters = trial.suggest_categorical(
                    "filters", [40, 50]
                )
                mlflow.log_param("filters", self.config.trainer.filters)
                self.config.trainer.units = trial.suggest_categorical("units", [32, 64])
                mlflow.log_param("units", self.config.trainer.units)
                model = BitcoinPriceCNNLSTMRegressionModel(optimizer, self.config)
            elif self.config.trainer.model_type_current == "ConvLSTM2D_Encoder_Decoder":
                self.config.trainer.units = trial.suggest_categorical("units", [32, 64])
                mlflow.log_param("units", self.config.trainer.units)
                model = BitcoinPriceEncoderDecoderConvLSTM2DRegressionModel(
                    optimizer, self.config
                )
            elif self.config.trainer.model_type_current == "LSTM_Encoder_Decoder":
                self.config.trainer.units = trial.suggest_categorical("units", [32, 64])
                mlflow.log_param("units", self.config.trainer.units)
                model = BitcoinPriceEncoderDecoderLSTMRegressionModel(
                    optimizer, self.config
                )
            elif self.config.trainer.model_type_current == "Attention":
                self.config.trainer.units = trial.suggest_categorical("units", [32, 64])
                mlflow.log_param("units", self.config.trainer.units)
                model = BitcoinPriceAttentionRegressionModel(optimizer, self.config)

            print("Create the trainer")
            trainer = BitcoinPriceModelTrainer(
                model.model,
                data_loader.get_train_data(),
                data_loader.get_test_data(),
                self.config,
            )

            print("Start training the model.")
            trainer.train()

            for step, loss in enumerate(trainer.loss, start=1):
                mlflow.log_metric("loss", loss, step)
            for step, val_loss in enumerate(trainer.val_loss, start=1):
                mlflow.log_metric("val_loss", val_loss, step)

            print("Evaluate model performance")
            return_val = trainer.evaluate()
            mlflow.log_metric("MAPE", return_val)

            mlflow.keras.log_model(model.model, "model")
            # replace custom evaluation with mlflow.evaluate
            # model_uri = mlflow.get_artifact_uri("model")

            return return_val

        # Initiate study
        self.study.optimize(
            _objective, self.config.optimizer.n_trials, callbacks=[mlflc]
        )

        print("Number of finished trials: {}".format(len(self.study.trials)))

        print("Best trial:")
        trial = self.study.best_trial

        print("  Value: {}".format(trial.value))

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

        for key, value in trial.params.items():
            mlflow.set_tag(key, value)

        mlflow.set_tag("best_trial", self.study.best_trial.number)
        mlflow.set_tag(
                "mlflow.runName",
                "study-summary-{}".format(datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")),
            )

        joblib.dump(
            self.study,
            "./artifacts/study-bitcoin-prediction-d-{}-v{}.pkl".format(
                self.config.optimizer.version,
                datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S"),
            ),
        )

        self.clean_models(self.study, self.config.trainer.model_name_prefix)
