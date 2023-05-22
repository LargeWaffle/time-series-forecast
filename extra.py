import matplotlib.pyplot as plt
import pandas as pd


def show_feature_importance(self, ft_val):
    plt.figure(figsize=(12, 6))

    (pd.Series(ft_val, index=self.feature_names)
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


def read_energy(data_path):
    train_df = pd.read_parquet(data_path + '/est_hourly.parquet')
    return train_df
