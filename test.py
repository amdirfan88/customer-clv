from pathlib import Path

import pandas as pd


DATA_DIR = Path(__file__).resolve().parent / 'data'

df_calib = pd.read_csv(
    DATA_DIR / 'working_data/calib_all_transaction.csv',
    usecols=['Customer_ID'],
    dtype={'Customer_ID': 'Int64'},
)
holdout_products = set(pd.read_csv(
    DATA_DIR / 'given_data/prod_catalog_holdout.csv',
    usecols=['StockCode'],
    dtype={'StockCode': 'string'},
)['StockCode'].dropna())
print(f'Calibration customers: {df_calib["Customer_ID"].nunique()}')
print(f'New catalogue products: {len(holdout_products)}')
