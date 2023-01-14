from base.base_data_producer import BaseDataProducer

import pandas as pd
import urllib.request, json


class BitcoinPriceDataProducer(BaseDataProducer):
    def __init__(self, config):
        super(BitcoinPriceDataProducer, self).__init__(config)

    def _retrieve_data(self):
        df_all = pd.DataFrame()

        if self.config.dataproducer.pull_data:
            print(
                "https://www.alphavantage.co/query?function=DIGITAL_CURRENCY_DAILY&symbol=BTC&market=USD&apikey="
                + self.config.dataproducer.apikey
            )
            with urllib.request.urlopen(
                "https://www.alphavantage.co/query?function=DIGITAL_CURRENCY_DAILY&symbol=BTC&market=USD&apikey="
                + self.config.dataproducer.apikey
            ) as url:
                data = json.loads(url.read().decode())

            df_all = pd.DataFrame(data["Time Series (Digital Currency Daily)"])
            # df_all = pd.read_json('https://www.alphavantage.co/query?function=DIGITAL_CURRENCY_DAILY&symbol=BTC&market=USD&apikey='+self.config.dataproducer.apikey)

            df_all = df_all.transpose()
            df_all.reset_index(level=0, inplace=True)  # turns the index into a column
            # df_all = df_all.rename(columns={'index':'event_id'})
            df_all = df_all.rename(columns={"index": "event_timestamp"})
            df_all = df_all.rename(columns={"1a. open (USD)": "open-usd"})
            df_all["open-usd"] = pd.to_numeric(
                df_all["open-usd"].astype(float), errors="coerce"
            )
            df_all = df_all.rename(columns={"2a. high (USD)": "high-usd"})
            df_all["high-usd"] = pd.to_numeric(
                df_all["high-usd"].astype(float), errors="coerce"
            )
            df_all = df_all.rename(columns={"3a. low (USD)": "low-usd"})
            df_all["low-usd"] = pd.to_numeric(
                df_all["low-usd"].astype(float), errors="coerce"
            )
            df_all = df_all.rename(columns={"4a. close (USD)": "close-usd"})
            df_all["close-usd"] = pd.to_numeric(
                df_all["close-usd"].astype(float), errors="coerce"
            )
            df_all = df_all.rename(columns={"5. volume": "volume"})
            df_all["volume"] = pd.to_numeric(
                df_all["volume"].astype(float), errors="coerce"
            )
            df_all = df_all.rename(columns={"6. market cap (USD)": "market-cap"})
            df_all["market-cap"] = pd.to_numeric(
                df_all["market-cap"].astype(float), errors="coerce"
            )
            df_all["event_timestamp"] = pd.to_datetime(df_all["event_timestamp"])
            df_all.index.name = "event_id"
            print(df_all.head())
            df_all.to_parquet("./data/raw/price_data_btc.parquet")
        else:
            df_all = pd.read_parquet("./data/raw/price_data_btc.parquet")

        return df_all

    def _preprocess_data(self):

        df = pd.read_parquet("./data/raw/price_data_btc.parquet")
        # df.index.name = 'event_timestamp'

        df.to_parquet("./data/processed/price_data_btc.parquet")

        return df
