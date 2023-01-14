from base.base_data_producer import BaseDataProducer

import pandas as pd
from feast import FeatureStore
from feast.infra.offline_stores.file_source import SavedDatasetFileStorage
import os

store = FeatureStore(repo_path="./data/feature_repo/")


class BitcoinTrainingDataProducer(BaseDataProducer):
    def __init__(self, config):
        super(BitcoinTrainingDataProducer, self).__init__(config)

    def _retrieve_data(self):
        df_all = pd.DataFrame()

        if self.config.dataproducer.pull_data:
            chart_df = pd.read_parquet("./data/processed/metric_data_btc.parquet")
            timestamps = pd.date_range(
                end=self.config.dataproducer.end_timestamp,
                periods=len(chart_df),
                freq="D",
            ).to_frame(name="event_timestamp", index=False)

            entity_df = pd.DataFrame.from_dict(
                {
                    "event_timestamp": [],
                }
            )
            entity_df = pd.concat(objs=[entity_df, timestamps])

            training_data = store.get_historical_features(
                entity_df=entity_df,
                features=store.get_feature_service("daily_bitcoin_v1"),
            )

            os.remove("./data/processed/bitcoin_training_dataset.parquet")
            dataset = store.create_saved_dataset(
                from_=training_data,
                name="bitcoin_training_dataset",
                storage=SavedDatasetFileStorage(
                    "./data/processed/bitcoin_training_dataset.parquet"
                ),
            )

            df_all = dataset.to_df()
        else:
            df_all = pd.read_parquet(
                "./data/processed/bitcoin_training_dataset.parquet"
            )

        print(df_all.head())
        return df_all

    def _preprocess_data(self):

        df = pd.read_parquet("./data/processed/bitcoin_training_dataset.parquet")

        return df
