import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures

from statsmodels.tsa.stattools import adfuller


def stationary_test(df):
    ts = df['sales'].iloc[:15000]
    result = adfuller(ts)

    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))


def assign_time_ft(df):
    df['payday'] = ((df['date'].dt.day == 15) | df['date'].dt.is_month_end).astype(int)
    df['weekday'] = df['date'].dt.weekday
    df['day'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year

    df['is_weekday'] = 0
    df.loc[df['weekday'] < 5, 'is_weekday'] = 1

    df["season"] = np.where(df.month.isin([12, 1, 2]), 0, 1)
    df["season"] = np.where(df.month.isin([6, 7, 8]), 2, df["season"])
    df["season"] = pd.Series(np.where(df.month.isin([9, 10, 11]), 3, df["season"])).astype("int8")

    return df


def format_sales(df, test_df, data_path):
    stores_df = pd.read_csv(data_path + '/store-sales/stores.csv')
    oil_df = pd.read_csv(data_path + '/store-sales/oil.csv', parse_dates=['date'])

    holidays_df = pd.read_csv(data_path + '/store-sales/holidays_events.csv', parse_dates=['date'])
    holidays_df['holiday'] = 1

    transactions_df = pd.read_csv(data_path + '/store-sales/transactions.csv', parse_dates=['date'])

    df = df.merge(stores_df, on='store_nbr', how='left')
    df = df.merge(oil_df, on='date', how='left')
    df = df.merge(transactions_df, on=['date', 'store_nbr'], how='left')

    lb = LabelEncoder()
    df['family'] = lb.fit_transform(df['family'])
    test_df['family'] = lb.transform(test_df['family'])

    df['city'] = lb.fit_transform(df['city'])
    df['state'] = lb.fit_transform(df['state'])
    df['type'] = lb.fit_transform(df['type'])

    df = df.merge(holidays_df[['date', 'holiday']], on='date', how='left')
    df['holiday'].fillna(0, inplace=True)
    df['holiday'] = df['holiday'].astype(int)

    lags = [1, 7, 14]
    for lag in lags:
        df[f'oil_lag_{lag}'] = df['dcoilwtico'].shift(lag)

    df = assign_time_ft(df)
    test_df = assign_time_ft(test_df)

    df.dropna(inplace=True)
    test_df.dropna(inplace=True)

    df = df[~((df.store_nbr == 52) & (df.date < "2017-04-20"))]
    df = df[~((df.store_nbr == 22) & (df.date < "2015-10-09"))]
    df = df[~((df.store_nbr == 42) & (df.date < "2015-08-21"))]
    df = df[~((df.store_nbr == 21) & (df.date < "2015-07-24"))]
    df = df[~((df.store_nbr == 29) & (df.date < "2015-03-20"))]
    df = df[~((df.store_nbr == 20) & (df.date < "2015-02-13"))]
    df = df[~((df.store_nbr == 53) & (df.date < "2014-05-29"))]
    df = df[~((df.store_nbr == 36) & (df.date < "2013-05-09"))]

    df = df.set_index('date')
    test_df = test_df.set_index('date')

    return df, test_df


def read_sales(data_path):
    train_df = pd.read_csv(data_path + '/store-sales/train.csv', parse_dates=['date'])
    test_df = pd.read_csv(data_path + '/store-sales/test.csv', parse_dates=['date'])

    return format_sales(train_df, test_df, data_path)


def read_energy(data_path):
    train_df = pd.read_parquet(data_path + '/energy/est_hourly.parquet')
    return train_df
