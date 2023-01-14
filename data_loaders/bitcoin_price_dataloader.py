from base.base_data_loader import BaseDataLoader
from sklearn.model_selection import train_test_split

from feast import FeatureStore

store = FeatureStore(repo_path="./data/feature_repo/")

import pandas as pd


class BitcoinPriceDataLoader(BaseDataLoader):
    def __init__(self, config):
        super(BitcoinPriceDataLoader, self).__init__(config)

        print("Load data...")
        timestamps = pd.date_range(
            start=self.config.dataloader.start_timestamp,
            end=self.config.dataloader.end_timestamp,
            freq="D",
        ).to_frame(name="event_timestamp", index=False)

        entity_df = pd.DataFrame.from_dict(
            {
                "event_timestamp": [],
            }
        )
        entity_df = pd.concat(objs=[entity_df, timestamps])

        train_df = store.get_historical_features(
            entity_df=entity_df,
            features=store.get_feature_service("daily_bitcoin_v1"),
        ).to_df()
        train_df.set_index("event_timestamp", inplace=True)

        if "prediction_length" in config.trainer:
            test_size = config.trainer.timesteps + (
                config.trainer.prediction_length + 1
            )
        else:
            test_size = 1

        print("Test Size: {}".format(test_size))
        print(train_df["market-price"].head())
        print(train_df["market-price"].tail())

        train_df["price"] = train_df["market-price"]
        train_df.drop(["market-price"], axis=1, inplace=True)
        df = train_df[[c for c in train_df if c not in ["price"]] + ["price"]]
        # get all data in range for batch scoring
        self.X, self.y = df.iloc[:, :-1].to_numpy(), df.iloc[:, -1:].to_numpy()
        # create train test split for training
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            df.iloc[:, :-1].to_numpy(),
            df.iloc[:, -1:].to_numpy(),
            test_size=test_size,
            shuffle=False,
            random_state=1,
        )

    def get_train_data(self):
        return self.X_train, self.y_train

    def get_test_data(self):
        return self.X_test, self.y_test

    def get_full_data(self):
        return self.X, self.y
