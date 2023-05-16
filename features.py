import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess


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
    df["dayofyear"] = df['date'].dt.dayofyear
    df['weekday'] = df['date'].dt.weekday
    df['day'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year

    df['is_weekday'] = 0
    df.loc[df['weekday'] < 5, 'is_weekday'] = 1

    return df


def fill_na(df):
    df['transactions'] = df['transactions'].fillna(0)

    df['transferred'] = df['transferred'].fillna(False)
    df['transferred'] = df['transferred'].astype(int)

    df['dcoilwtico'] = df['dcoilwtico'].fillna(method='backfill')

    return df


def lag_ft(df, lag_infos):
    for col_name, lags in lag_infos.items():
        for lag in lags:
            df[f'{col_name}_{lag}'] = df[col_name].shift(lag)

    return df


def unify_types(df):
    df[df.select_dtypes(np.int64).columns] = df.select_dtypes(np.int64).astype(np.int32)
    df[df.select_dtypes(np.float32).columns] = df.select_dtypes(np.float32).astype(np.float64)

    return df


def format_sales(df, data_path):
    stores_df = pd.read_csv(data_path + '/stores.csv')
    oil_df = pd.read_csv(data_path + '/oil.csv', parse_dates=['date'])

    holidays_df = pd.read_csv(data_path + '/holidays_events.csv', parse_dates=['date'])
    holidays_df['holiday'] = 1

    transactions_df = pd.read_csv(data_path + '/transactions.csv', parse_dates=['date'])

    df = df.merge(stores_df, on='store_nbr', how='left')
    df = df.merge(oil_df, on='date', how='left')
    df = df.merge(transactions_df, on=['date', 'store_nbr'], how='left')

    lb = LabelEncoder()
    df['family'] = lb.fit_transform(df['family'])
    df['city'] = lb.fit_transform(df['city'])
    df['state'] = lb.fit_transform(df['state'])
    df['type'] = lb.fit_transform(df['type'])

    df = df.merge(holidays_df[['date', 'holiday', 'transferred']], on='date', how='left')

    df['holiday'].fillna(0, inplace=True)
    df['holiday'] = df['holiday'].astype(int)

    df = fill_na(df)

    lag_features = {
        'dcoilwtico': [1, 2, 3, 7, 14],
        'sales': [1, 2, 3]
    }

    df = lag_ft(df, lag_features)
    df = assign_time_ft(df)

    df = unify_types(df)

    return df


def read_sales(data_path):
    train_df = pd.read_csv(data_path + '/train.csv', parse_dates=['date'])
    test_df = pd.read_csv(data_path + '/test.csv', parse_dates=['date'])

    data_df = pd.concat([train_df, test_df], axis=0)
    data_df = format_sales(data_df, data_path)

    train_df = data_df[data_df.index <= pd.to_datetime("2017-08-15")]
    train_df = train_df.dropna()

    test_df = data_df[data_df.index > pd.to_datetime("2017-08-15")]

    return train_df, test_df


def read_energy(data_path):
    train_df = pd.read_parquet(data_path + '/energy/est_hourly.parquet')
    return train_df
