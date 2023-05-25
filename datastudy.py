import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_squared_log_error
from statsmodels.tsa.stattools import adfuller


def show_metrics(y_val, y_pred, mdict):
    mae = mean_absolute_error(y_val, y_pred)
    mse = mean_squared_error(y_val, y_pred, squared=True)
    rmsle = mean_squared_log_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)

    print("\nRegression metrics")
    print('MAE: {:.2f}'.format(mae))
    print('MSE: {:.2f}'.format(mse))
    print('RMSLE: {:.2f}'.format(rmsle))
    print('R2: {:.2f}'.format(r2))

    if mdict is not None:
        mdict["mae"].append(mae)
        mdict["mse"].append(mse)
        mdict["rmsle"].append(rmsle)
        mdict["r2"].append(r2)


def show_feature_importance(feature_names, ft_val):
    plt.figure(figsize=(12, 6))

    (pd.Series(ft_val, index=feature_names)
     .sort_values(ascending=True)
     .plot(kind='barh'))

    plt.show()


def plot_predictions(nb_samples, y_val, y_pred):
    sp_list = list(range(0, nb_samples))
    plt.figure(figsize=(10, 6))

    plt.plot(sp_list, y_val, label='Expected', alpha=0.5)
    plt.plot(sp_list, y_pred, label='Predicted', alpha=0.5)
    plt.legend(loc="upper right")
    plt.show()

    plt.plot(sp_list, abs(y_val - y_pred), label='Difference')
    plt.legend(loc="upper right")
    plt.show()


def corr_matrix(df):
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=False, linewidth=.5, vmin=-1, vmax=1, fmt=".2f", square=True)
    plt.show()


def stationary_test(df):
    ts = df['sales'].iloc[:15000]
    result = adfuller(ts)

    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))
