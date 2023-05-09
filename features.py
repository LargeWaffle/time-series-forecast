import pandas as pd


def read_sales(data_path):
    train_df = pd.read_csv(data_path + '/store-sales/train.csv')
    stores_df = pd.read_csv(data_path + '/store-sales/stores.csv')
    oil_df = pd.read_csv(data_path + '/store-sales/oil.csv')
    holidays_df = pd.read_csv(data_path + '/store-sales/holidays_events.csv')
    transactions_df = pd.read_csv(data_path + '/store-sales/transactions.csv')
    
    print(train_df.head(5))


def read_energy(data_path):
    train_df = pd.read_parquet(data_path + '/energy/est_hourly.parqet')
