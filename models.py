import xgboost as xgb
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from base import BaseModel


class LRegModel(BaseModel):
    def __init__(self, show_fip=True, pca=False, scaler=StandardScaler()):
        super().__init__(show_fip, pca, scaler)
        self.model = LinearRegression()


class ElasticModel(BaseModel):
    def __init__(self, show_fip=True, pca=False, scaler=StandardScaler()):
        super().__init__(show_fip, pca, scaler)
        self.model = ElasticNet()


class XGBModel(BaseModel):
    def __init__(self, show_fip=True, pca=False, scaler=MinMaxScaler(), ft_attr='gain', nb_estimators=100):
        super().__init__(show_fip, pca, scaler)

        self.model = xgb.XGBRegressor(n_estimators=nb_estimators, importance_type=ft_attr, eval_metric='rmse',
                                      early_stopping_rounds=5)

    def train(self, x_train, y_train, x_val, y_val, ft_type="tree"):
        self.model.fit(x_train, y_train, eval_set=[(x_val, y_val)])
        self.model.get_booster().feature_names = self.feature_names
        self.ft_values = self.assign_ftip(ft_type)


class LGBMModel(BaseModel):
    def __init__(self, show_fip=True, pca=False, scaler=MinMaxScaler(), ft_attr='gain', nb_estimators=100):
        super().__init__(show_fip, pca, scaler)

        self.model = LGBMRegressor(n_estimators=nb_estimators, importance_type=ft_attr, eval_metric='rmse',
                                   early_stopping_rounds=5)

    def train(self, x_train, y_train, x_val, y_val, ft_type="tree"):
        self.model.fit(x_train, y_train, eval_set=[(x_val, y_val)], feature_name=self.feature_names)
        self.ft_values = self.assign_ftip(ft_type)


class RandomForestModel(BaseModel):
    def __init__(self, show_fip=True, pca=False, scaler=StandardScaler(), nb_estimators=100):
        super().__init__(show_fip, pca, scaler)

        self.model = RandomForestRegressor(n_estimators=nb_estimators, verbose=2)

    def train(self, x_train, y_train, x_val, y_val, ft_type="tree"):
        self.model.fit(x_train, y_train)
        self.ft_values = self.assign_ftip(ft_type)


class KNNModel(BaseModel):
    def __init__(self, show_fip=False, pca=False, scaler=StandardScaler(), nb_neighbours=5):
        super().__init__(False, pca, scaler)

        self.model = KNeighborsRegressor(n_neighbors=nb_neighbours)


class MLPModel(BaseModel):
    def __init__(self, show_fip=False, pca=False, scaler=MinMaxScaler(), maxiter=150, val_frac=0.2):
        super().__init__(False, pca, scaler)
        self.model = MLPRegressor(max_iter=maxiter, verbose=True, early_stopping=True, validation_fraction=val_frac)
