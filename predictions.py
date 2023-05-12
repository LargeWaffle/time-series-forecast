import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, ElasticNet, SGDRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, PredictionErrorDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline

from statsmodels.tsa.stattools import adfuller


def plot_predictions(x_val, y_val, y_pred):
    samples = list(range(0, len(x_val)))
    plt.figure(figsize=(10, 6))
    plt.plot(samples, y_val, label='Expected', alpha=0.5)
    plt.plot(samples, y_pred, label='Predicted', alpha=0.5)
    plt.legend(loc="upper right")
    plt.show()

    plt.plot(samples, abs(y_val - y_pred), label='Difference')
    plt.legend(loc="upper right")
    plt.show()


def lreg():
    model = LinearRegression()
    # model.fit(x_train, y)


def xgb_regressor(train_df, show_ft_ip=False, pca=False):
    x = train_df.drop(['id', 'store_nbr', 'sales', 'dcoilwtico'], axis=1)
    y = train_df['sales']

    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.15, random_state=42)

    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)

    feature_names = list(x.columns)

    if pca:
        pca = PCA(n_components=0.95)
        x_train = pca.fit_transform(x_train)
        x_val = pca.transform(x_val)

        n_pcs = pca.components_.shape[0]

        # get the most important feature on EACH component
        most_important = [np.abs(pca.components_[i]).argmax() for i in range(n_pcs)]

        # get the names
        most_names = [feature_names[most_important[i]] for i in range(n_pcs)]

        feature_names = ['PC{}_{}'.format(i + 1, most_names[i]) for i in range(n_pcs)]

    model = xgb.XGBRegressor(eval_metric='rmse', early_stopping_rounds=10)
    model = model.fit(x_train, y_train, eval_set=[(x_val, y_val)])

    if show_ft_ip:
        model.get_booster().feature_names = feature_names

        _, ax = plt.subplots(figsize=(10, 6))
        xgb.plot_importance(model, importance_type='weight', xlabel='weight', ylabel='features',
                            height=0.6, show_values=False, grid=False, ax=ax,
                            title='XGBoost feature importance')
        plt.show()

    y_pred = model.predict(x_val)
    y_pred[y_pred < 0] = 0

    print('MAE: {:.2f}'.format(mean_absolute_error(y_val, y_pred)))
    print('MSE: {:.2f}'.format(mean_squared_error(y_val, y_pred)))
    print('R2: {:.2f}'.format(r2_score(y_val, y_pred)))

    plot_predictions(x_val, y_val, y_pred)


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
