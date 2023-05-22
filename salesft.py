import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder

from statsmodels.tsa.deterministic import DeterministicProcess, CalendarFourier


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

    df["season"] = df["month"] % 12 // 3

    df['earthquake_rl'] = np.where(df['description'].str.contains('Terremoto Manabi'), 1, 0)

    return df


def handle_na(df):
    df['transferred'] = df['transferred'].fillna(False).astype(int)
    df['is_holiday'] = df['is_holiday'].fillna(0).astype(int)

    df['description'] = df['description'].fillna('None')

    df['holiday_type'] = df['holiday_type'].fillna('Common')
    df['locale'] = df['locale'].fillna('Common')
    df['locale_name'] = df['locale_name'].fillna('Ecuador')  # WARNING

    df['dcoilwtico'] = df['dcoilwtico'].fillna(method='backfill')

    df['transactions'] = df['transactions'].fillna(0).astype(int)

    return df


def encode_ft(df):
    lb = LabelEncoder()
    df['family'] = lb.fit_transform(df['family'])
    df['city'] = lb.fit_transform(df['city'])
    df['state'] = lb.fit_transform(df['state'])
    df['store_type'] = lb.fit_transform(df['store_type'])
    df['holiday_type'] = lb.fit_transform(df['holiday_type'])
    df['locale'] = lb.fit_transform(df['locale'])
    df['locale_name'] = lb.fit_transform(df['locale_name'])

    return df


def lag_ft(df, lag_infos):
    for col_name, lags in lag_infos.items():
        for lag in lags:
            df[f'{col_name}_{lag}'] = df[col_name].shift(lag)
            df[f'{col_name}_{lag}'] = df[f'{col_name}_{lag}'].fillna(0).astype(float)

    return df


def window_ft(df):
    times = {'week': 7, '2weeks': 14, 'month': 28}

    for key, val in times.items():
        df[f'oil_{key}_avg'] = df['dcoilwtico'].rolling(val).mean()
        df[f'oil_{key}_avg'] = df[f'oil_{key}_avg'].fillna(0).astype(float)

        df[f'oil_{key}_min'] = df['dcoilwtico'].rolling(val).min()
        df[f'oil_{key}_min'] = df[f'oil_{key}_min'].fillna(0).astype(float)

        df[f'oil_{key}_max'] = df['dcoilwtico'].rolling(val).max()
        df[f'oil_{key}_max'] = df[f'oil_{key}_max'].fillna(0).astype(float)

    df['avg_transactions'] = df['transactions'].rolling(15, min_periods=10).mean()
    df['avg_transactions'] = df['avg_transactions'].fillna(0).astype(float)

    df['min_transactions'] = df['transactions'].rolling(15, min_periods=10).min()
    df['min_transactions'] = df['min_transactions'].fillna(0).astype(float)

    df['max_transactions'] = df['transactions'].rolling(15, min_periods=10).max()
    df['max_transactions'] = df['max_transactions'].fillna(0).astype(float)

    return df


def format_sales(df, data_path):
    stores_df = pd.read_csv(data_path + '/stores.csv')
    stores_df = stores_df.rename(columns={'type': 'store_type'})

    transactions_df = pd.read_csv(data_path + '/transactions.csv', parse_dates=['date'])
    oil_df = pd.read_csv(data_path + '/oil.csv', parse_dates=['date'])

    holidays_df = pd.read_csv(data_path + '/holidays_events.csv', parse_dates=['date'])
    holidays_df = holidays_df.rename(columns={'type': 'holiday_type'})
    holidays_df['is_holiday'] = 1

    df = df.merge(stores_df, on='store_nbr', how='left')
    df = df.merge(oil_df, on='date', how='left')
    df = df.merge(transactions_df, on=['date', 'store_nbr'], how='left')
    df = df.merge(holidays_df, on='date', how='left')

    df = handle_na(df)
    df.loc[df['transferred'] == 1, 'is_holiday'] = 0

    df = encode_ft(df)
    df = assign_time_ft(df)
    df = df.drop(['description'], axis=1)

    lag_features = {
        'dcoilwtico': [1, 3, 7, 14],
        'transactions': [1, 3, 7, 14]
    }

    for family in df['family'].unique():

        sub_df = df.loc[df['family'] == family]

        sub_df = lag_ft(sub_df, lag_features)
        sub_df = window_ft(sub_df)
        df.loc[sub_df.index, sub_df.columns] = sub_df

    df[df.select_dtypes(np.int32).columns] = df.select_dtypes(np.int32).astype(np.int64)
    df[df.select_dtypes(np.float32).columns] = df.select_dtypes(np.float32).astype(np.float64)

    return df.set_index('date')


def read_sales(data_path, scaler, use_pca=False):
    train_df = pd.read_csv(data_path + '/train.csv', parse_dates=['date'])
    test_df = pd.read_csv(data_path + '/test.csv', parse_dates=['date'])

    train_df = format_sales(train_df, data_path)
    scaled = train_df.drop(['id', 'sales'], axis=1)
    train_df[scaled.columns] = scaler.fit_transform(scaled[scaled.columns])

    test_df = format_sales(test_df, data_path)
    scaled = test_df.drop(['id'], axis=1)
    test_df[scaled.columns] = scaler.transform(scaled[scaled.columns])

    train_data = {}
    test_data = {}
    for idx, famval in enumerate(train_df['family'].unique()):
        train_data[idx] = train_df.loc[train_df['family'] == famval]
        test_data[idx] = test_df.loc[test_df['family'] == famval]

    return train_data, test_data
