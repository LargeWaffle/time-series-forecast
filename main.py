from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
from features import read_sales
from models import LRegModel, ElasticModel, XGBModel, LGBMModel, RandomForestModel, KNNModel, MLPModel

DATAPATH = "data"


def create_submission(model, test_df):
    test_sales = model.predict(test_df)

    submission_df = pd.read_csv(DATAPATH + '/store-sales/sample_submission.csv')
    submission_df['sales'] = test_sales
    submission_df.to_csv('/kaggle/working/submission.csv', index=False)

    print("Submission saved!")


if __name__ == '__main__':
    train_data, test_data = read_sales(DATAPATH)

    # train-test split for time series
    train_size = int(len(train_data) * 0.65)
    val_size = len(train_data) - train_size
    train_df, val_df = train_data[:train_size], train_data[train_size:]

    drop_cols = ['id', 'sales', 'dcoilwtico']

    model = XGBModel(show_fip=False, pca=False, scaler=MinMaxScaler(), nb_estimators=200)

    x_train, x_val, y_train, y_val = model.process_data(train_df, val_df, drop_cols)
    model.train(x_train, y_train, x_val, y_val)

    y_pred = model.predict(x_val)
    model.resume_training(y_val, y_pred)

    create_submission(model, test_data)

    print("\nEnd of program")
