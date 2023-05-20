from darts.dataprocessing.transformers import Scaler
from darts.metrics import mae, mse, rmse, rmsle, r2_score

from single_model.salesft import read_sales

DATAPATH = "data/store-sales"

scaler = Scaler()

train_data, test_data = read_sales(DATAPATH)

# train-test split for time series
train_size = int(len(train_data) * 0.75)
val_size = len(train_data) - train_size
train_df, val_df = train_data[:train_size], train_data[train_size:]

drop_cols = ['id', 'sales', 'transferred', 'locale_name']

print("MAE = {:.2f}%".format(mae(series_air_scaled, pred)))
print("MSE = {:.2f}%".format(mse(series_air_scaled, pred)))
print("RMSE = {:.2f}%".format(rmse(series_air_scaled, pred)))
print("RMSLE = {:.2f}%".format(rmsle(series_air_scaled, pred)))
print("R2 = {:.2f}%".format(r2_score(series_air_scaled, pred)))
