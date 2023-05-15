import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


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


class BaseModel:
    def __init__(self, show_fip=True, pca=False, scaler=None):
        self.pca = pca
        self.show_fip = show_fip

        self.model = None
        self.ft_values = []
        self.feature_names = None

        self.scaler = scaler
        self.nb_train_samples = 0
        self.nb_val_samples = 0

    def apply_pca(self, x_train, x_val):
        pca = PCA(n_components=0.95)
        x_train = pca.fit_transform(x_train)
        x_val = pca.transform(x_val)

        n_pcs = pca.components_.shape[0]

        # get the most important feature on EACH component
        most_important = [np.abs(pca.components_[i]).argmax() for i in range(n_pcs)]
        # get the names
        most_names = [self.feature_names[most_important[i]] for i in range(n_pcs)]
        self.feature_names = ['PC{}_{}'.format(i + 1, most_names[i]) for i in range(n_pcs)]

        return x_train, x_val

    def process_data(self, train_df, val_df, drop_cols):
        x_train = train_df.drop(drop_cols, axis=1)
        y_train = train_df['sales']

        x_val = val_df.drop(drop_cols, axis=1)
        y_val = val_df['sales']

        self.nb_train_samples = len(x_train)
        self.nb_val_samples = len(x_val)
        self.feature_names = list(x_train.columns)

        x_train = self.scaler.fit_transform(x_train)
        x_val = self.scaler.transform(x_val)

        if self.pca:
            x_train, x_val = self.apply_pca(x_train, x_val)

        return x_train, x_val, y_train, y_val

    def assign_ftip(self, ft_type):
        if self.show_fip:
            if ft_type == "linear":
                return self.model.coef_
            elif ft_type == "tree":
                return self.model.feature_importances_
            else:
                return []

    def train(self, x_train, y_train, x_val, y_val, ft_type="linear"):
        self.model.fit(x_train, y_train)
        self.ft_values = self.assign_ftip(ft_type)

    def predict(self, x_val, neg_to_zero=True):
        y_pred = self.model.predict(x_val)

        if neg_to_zero:
            y_pred[y_pred < 0] = 0

        return y_pred

    def show_feature_importance(self, ft_val):
        plt.figure(figsize=(12, 6))

        (pd.Series(ft_val, index=self.feature_names)
         .sort_values(ascending=True)
         .plot(kind='barh'))

        plt.show()

    def resume_training(self, y_val, y_pred):

        print("\nRegression metrics")
        print('MAE: {:.2f}'.format(mean_absolute_error(y_val, y_pred)))
        print('MSE: {:.2f}'.format(mean_squared_error(y_val, y_pred)))
        print('R2: {:.2f}'.format(r2_score(y_val, y_pred)))

        print("\nPlotting truth vs predictions")
        plot_predictions(self.nb_val_samples, y_val, y_pred)

        if self.show_fip:
            print("Plotting feature importance")
            self.show_feature_importance(self.ft_values)
