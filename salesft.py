import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder


def assign_time_ft(df):
    df['payday'] = ((df['date'].dt.day == 15) | df['date'].dt.is_month_end).astype(int)
    df["dayofyear"] = df['date'].dt.dayofyear
    df["weekofyear"] = df['date'].dt.isocalendar().week
    df['weekday'] = df['date'].dt.weekday
    df['day'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year

    df['is_weekday'] = 0
    df.loc[df['weekday'] < 5, 'is_weekday'] = 1

    return df


def handle_na(df):
    df['transferred'] = df['transferred'].fillna(False).astype(int)
    df['is_holiday'] = df['is_holiday'].fillna(0).astype(int)

    df['holiday_type'] = df['holiday_type'].fillna('Common')
    df['locale'] = df['locale'].fillna('Common')
    df['locale_name'] = df['locale_name'].fillna('Ecuador')  # WARNING

    df['transactions'] = df['transactions'].fillna(0).astype(int)
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
    stores_df = stores_df.rename(columns={'type': 'store_type'})

    oil_df = pd.read_csv(data_path + '/oil.csv', parse_dates=['date'])
    transactions_df = pd.read_csv(data_path + '/transactions.csv', parse_dates=['date'])

    df = df.merge(stores_df, on='store_nbr', how='left')
    df = df.merge(oil_df, on='date', how='left')
    df = df.merge(transactions_df, on=['date', 'store_nbr'], how='left')

    holidays_df = pd.read_csv(data_path + '/holidays_events.csv', parse_dates=['date'])
    holidays_df = holidays_df.drop(['description'], axis=1)
    holidays_df = holidays_df.rename(columns={'type': 'holiday_type'})
    holidays_df['is_holiday'] = 1

    df = df.merge(holidays_df, on='date', how='left')

    df = handle_na(df)
    df.loc[df['transferred'] == 1, 'is_holiday'] = 0

    lb = LabelEncoder()
    df['family'] = lb.fit_transform(df['family'])
    df['city'] = lb.fit_transform(df['city'])
    df['state'] = lb.fit_transform(df['state'])
    df['store_type'] = lb.fit_transform(df['store_type'])
    df['holiday_type'] = lb.fit_transform(df['holiday_type'])
    df['locale'] = lb.fit_transform(df['locale'])
    df['locale_name'] = lb.fit_transform(df['locale_name'])

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
    data_df = data_df.set_index('date')

    train_df = data_df[data_df.index <= pd.to_datetime("2017-08-15")]
    train_df = train_df.dropna()

    test_df = data_df[data_df.index > pd.to_datetime("2017-08-15")]
    test_df = test_df.drop(['sales_1', 'sales_2', 'sales_3'], axis=1)

    return train_df, test_df
