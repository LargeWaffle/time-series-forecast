import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import adfuller


def lreg():
    model = LinearRegression()
    # model.fit(x_train, y)


def xgb_regressor(train_df):
    x = train_df.drop(['id', 'store_nbr', 'sales', 'dcoilwtico'], axis=1)
    y = train_df['sales']

    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)

    pca = PCA(n_components=0.95)
    x_train = pca.fit_transform(x_train)
    x_val = pca.transform(x_val)

    model = xgb.XGBRegressor()
    model.fit(x_train, y_train)
    model.get_booster().feature_names = list(x.columns)

    xgb.plot_importance(model, importance_type='weight', xlabel='weight', ylabel='features',
                        height=0.6, show_values=False, title='XGBoost feature importance')
    plt.show()

    y_pred = model.predict(x_val)
    y_pred[y_pred < 0] = 0

    mae = mean_absolute_error(y_val, y_pred)
    mse = mean_squared_error(y_val, y_pred)
    print(f'MAE: {mae:.2f}')
    print(f'MSE: {mse:.2f}')


def stationary_test(df):
    ts = df['sales'].iloc[:15000]
    result = adfuller(ts)

    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))


def create_submission(sub_df, model):
    scaler = MinMaxScaler()
    sub_df = scaler.fit_transform(sub_df)
    pca = PCA(n_components=0.95)
    sub_df = pca.fit_transform(sub_df)

    test_sales = model.predict(sub_df)

    sub_df['sales'] = test_sales
    sub_df.to_csv('/kaggle/working/submission.csv', index=False)
