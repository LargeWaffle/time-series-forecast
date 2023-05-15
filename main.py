from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

from features import read_sales
from base import ModelManager
from models import LRegModel, ElasticModel, XGBModel, LGBMModel, RandomForestModel, MLPModel

DATAPATH = "data"


def create_submission(sub_df, model):
    scaler = MinMaxScaler()
    sub_df = scaler.fit_transform(sub_df)
    pca = PCA(n_components=0.95)
    sub_df = pca.fit_transform(sub_df)

    test_sales = model.predict(sub_df)

    sub_df['sales'] = test_sales
    sub_df.to_csv('/kaggle/working/submission.csv', index=False)


if __name__ == '__main__':
    train_data, _, _ = read_sales(DATAPATH)

    # train-test split for time series
    train_size = int(len(train_data) * 0.65)
    val_size = len(train_data) - train_size
    train_df, val_df = train_data[:train_size], train_data[train_size:]

    drop_cols = ['id', 'store_nbr', 'sales', 'dcoilwtico']

    mm = ModelManager(XGBModel)
    mm.run_experiment(train_df, val_df, drop_cols, scaler=MinMaxScaler)

    print("\nEnd of program")
