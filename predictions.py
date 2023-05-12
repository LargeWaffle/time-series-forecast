import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, ElasticNet, SGDRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, PredictionErrorDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.neural_network import MLPRegressor


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


def apply_pca(x_train, x_val, feature_names):
    pca = PCA(n_components=0.95)
    x_train = pca.fit_transform(x_train)
    x_val = pca.transform(x_val)

    n_pcs = pca.components_.shape[0]

    # get the most important feature on EACH component
    most_important = [np.abs(pca.components_[i]).argmax() for i in range(n_pcs)]
    # get the names
    most_names = [feature_names[most_important[i]] for i in range(n_pcs)]
    feature_names = ['PC{}_{}'.format(i + 1, most_names[i]) for i in range(n_pcs)]

    return x_train, x_val, feature_names


def show_feat_imp(feature_val, feature_names):
    plt.figure(figsize=(12, 6))

    (pd.Series(feature_val, index=feature_names)
     .sort_values(ascending=True)
     .plot(kind='barh'))

    plt.show()


def resume_training(model, x_val, y_val):
    y_pred = model.predict(x_val)
    y_pred[y_pred < 0] = 0

    print('MAE: {:.2f}'.format(mean_absolute_error(y_val, y_pred)))
    print('MSE: {:.2f}'.format(mean_squared_error(y_val, y_pred)))
    print('R2: {:.2f}'.format(r2_score(y_val, y_pred)))

    plot_predictions(x_val, y_val, y_pred)


def lreg(train_df, show_ft_ip=False, pca=False):
    x = train_df.drop(['id', 'store_nbr', 'sales', 'dcoilwtico'], axis=1)
    y = train_df['sales']

    feature_names = list(x.columns)

    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)

    if pca:
        x_train, x_val, feature_names = apply_pca(x_train, x_val, feature_names)

    model = LinearRegression()
    model.fit(x_train, y_train)

    resume_training(model, x_val, y_val)

    if show_ft_ip:
        show_feat_imp(model.coef_, feature_names)


def elastic(train_df, show_ft_ip=False, pca=False):
    x = train_df.drop(['id', 'store_nbr', 'sales', 'dcoilwtico'], axis=1)
    y = train_df['sales']

    feature_names = list(x.columns)

    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)

    if pca:
        x_train, x_val, feature_names = apply_pca(x_train, x_val, feature_names)

    model = ElasticNet()
    model.fit(x_train, y_train)

    resume_training(model, x_val, y_val)

    if show_ft_ip:
        show_feat_imp(model.coef_, feature_names)


def xgb_regressor(train_df, show_ft_ip=False, pca=False):
    x = train_df.drop(['id', 'store_nbr', 'sales', 'dcoilwtico'], axis=1)
    y = train_df['sales']

    feature_names = list(x.columns)

    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)

    if pca:
        x_train, x_val, feature_names = apply_pca(x_train, x_val, feature_names)

    ft_attr = 'gain'
    model = xgb.XGBRegressor(n_estimators=25, importance_type=ft_attr, eval_metric='rmse', early_stopping_rounds=5)
    model = model.fit(x_train, y_train, eval_set=[(x_val, y_val)])
    model.get_booster().feature_names = feature_names

    resume_training(model, x_val, y_val)

    if show_ft_ip:
        show_feat_imp(model.feature_importances_, feature_names)


def mlp(train_df, pca=False):
    x = train_df.drop(['id', 'store_nbr', 'sales', 'dcoilwtico'], axis=1)
    y = train_df['sales']

    scaler = MinMaxScaler()
    x = scaler.fit_transform(x)

    if pca:
        pca = PCA(n_components=0.95)
        x = pca.fit_transform(x)

    model = MLPRegressor(max_iter=150, verbose=True, early_stopping=True, validation_fraction=0.2)
    model = model.fit(x, y)

    x_val = x[:750]
    y_val = y[:750]

    resume_training(model, x_val, y_val)
