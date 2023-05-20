import pandas as pd

from models import *
from salesft import read_sales

DATAPATH = "data/store-sales"


def create_submission(model, test_df):
    test_sales = model.predict(test_df)

    submission_df = pd.read_csv(DATAPATH + '/sample_submission.csv')
    submission_df['sales'] = test_sales
    submission_df.to_csv('submission.csv', index=False)

    print("Submission saved!")


if __name__ == '__main__':
    train_data, test_data = read_sales(DATAPATH)

    # train-test split for time series
    train_size = int(len(train_data) * 0.75)
    val_size = len(train_data) - train_size
    train_df, val_df = train_data[:train_size], train_data[train_size:]

    drop_cols = ['id', 'sales', 'transferred', 'locale_name']

    forecast_model = XGBModel(show_fip=True, use_pca=False, nb_estimators=200)

    x_train, x_val, y_train, y_val = forecast_model.process_data(train_df, val_df, drop_cols)
    forecast_model.train(x_train, y_train, x_val, y_val)

    y_pred = forecast_model.evaluate(x_val)
    forecast_model.resume_training(y_val, y_pred)

    create_submission(forecast_model, test_data)

    print("\nEnd of program")
