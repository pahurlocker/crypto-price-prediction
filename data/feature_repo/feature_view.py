from datetime import timedelta

import pandas as pd

from feast import (
    FeatureService,
    FeatureView,
    Field,
    FileSource,
)
from feast.types import Float64

daily_bitcoin_metric_source = FileSource(
    name="daily_bitcoin_metric_source",
    path="/Users/phurlocker/Documents/mycode/crypto-price-prediction/data/processed/metric_data_btc.parquet",
    timestamp_field="event_timestamp",
)

daily_bitcoin_price_source = FileSource(
    name="daily_bitcoin_price_source",
    path="/Users/phurlocker/Documents/mycode/crypto-price-prediction/data/processed/price_data_btc.parquet",
    timestamp_field="event_timestamp",
)

daily_bitcoin_metric_fv = FeatureView(
    name="daily_bitcoin_metric_source",
    entities=[],
    ttl=timedelta(days=10),
    schema=[
        Field(name="blocks-size", dtype=Float64),
        Field(name="avg-block-size", dtype=Float64),
        Field(name="n-transactions-total", dtype=Float64),
        Field(name="hash-rate", dtype=Float64),
        Field(name="difficulty", dtype=Float64),
        Field(name="transaction-fees-usd", dtype=Float64),
        Field(name="n-unique-addresses", dtype=Float64),
        Field(name="n-transactions", dtype=Float64),
        Field(name="my-wallet-n-users", dtype=Float64),
        Field(name="utxo-count", dtype=Float64),
        Field(name="n-transactions-excluding-popular", dtype=Float64),
        Field(name="estimated-transaction-volume-usd", dtype=Float64),
        Field(name="trade-volume", dtype=Float64),
        Field(name="total-bitcoins", dtype=Float64),
        Field(name="market-price", dtype=Float64),
    ],
    online=False,
    source=daily_bitcoin_metric_source,
    tags={"currency": "btc"},
)

daily_bitcoin_price_fv = FeatureView(
    name="daily_bitcoin_price_source",
    entities=[],
    ttl=timedelta(days=10),
    schema=[
        Field(name="open-usd", dtype=Float64),
        Field(name="high-usd", dtype=Float64),
        Field(name="low-usd", dtype=Float64),
        Field(name="close-usd", dtype=Float64),
        Field(name="volume", dtype=Float64),
        Field(name="market-cap", dtype=Float64),
    ],
    online=False,
    source=daily_bitcoin_price_source,
    tags={"currency": "btc"},
)

daily_bitcoin_v1 = FeatureService(
    name="daily_bitcoin_v1", features=[daily_bitcoin_metric_fv, daily_bitcoin_price_fv]
)
