import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder


def read_energy(data_path):
    train_df = pd.read_parquet(data_path + '/est_hourly.parquet')
    return train_df
