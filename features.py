import pandas as pd
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures


def format_sales(df, stores_df, oil_df, holidays_df, transactions_df):
    df = df.merge(stores_df, on='store_nbr', how='left')
    df = df.merge(oil_df, on='date', how='left')
    df = df.merge(transactions_df, on=['date', 'store_nbr'], how='left')

    lb = LabelEncoder()
    df['family'] = lb.fit_transform(df['family'])
    df['city'] = lb.fit_transform(df['city'])
    df['state'] = lb.fit_transform(df['state'])
    df['type'] = lb.fit_transform(df['type'])

    df = df.merge(holidays_df[['date', 'holiday']], on='date', how='left')
    df['holiday'].fillna(0, inplace=True)
    df['holiday'] = df['holiday'].astype(int)

    lags = [1, 7, 14]
    for lag in lags:
        df[f'oil_lag_{lag}'] = df['dcoilwtico'].shift(lag)

    df['payday'] = ((df['date'].dt.day == 15) | df['date'].dt.is_month_end).astype(int)
    df['weekday'] = df['date'].dt.weekday
    df['day'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year

    df['is_weekday'] = 0
    df.loc[df['weekday'] < 5, 'is_weekday'] = 1

    df.dropna(inplace=True)

    df = df.set_index('date')

    return df


def df_with_dt(f_path):
    df = pd.read_csv(f_path)
    df['date'] = pd.to_datetime(df['date'])
    return df


def read_sales(data_path):

    # df = pd.read_csv('BTC-USD.csv', index_col = 'Date', parse_dates=True)
    sample_sub = pd.read_csv(data_path + '/store-sales/sample_submission.csv')
    stores_df = pd.read_csv(data_path + '/store-sales/stores.csv')

    train_df = df_with_dt(data_path + '/store-sales/train.csv')
    test_df = df_with_dt(data_path + '/store-sales/test.csv')
    oil_df = df_with_dt(data_path + '/store-sales/oil.csv')

    holidays_df = df_with_dt(data_path + '/store-sales/holidays_events.csv')
    holidays_df['holiday'] = 1

    transactions_df = df_with_dt(data_path + '/store-sales/transactions.csv')

    train_df = format_sales(train_df, stores_df, oil_df, holidays_df, transactions_df)
    test_df = format_sales(test_df, stores_df, oil_df, holidays_df, transactions_df)

    return train_df, test_df, sample_sub


def read_energy(data_path):
    train_df = pd.read_parquet(data_path + '/energy/est_hourly.parquet')
    return train_df
