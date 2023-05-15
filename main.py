from features import read_sales
from predictions import lreg, elastic, xgb_regressor, mlp, lgbm, rf

DATAPATH = "data"

if __name__ == '__main__':
    train_data, _, _ = read_sales(DATAPATH)

    # train-test split for time series
    train_size = int(len(train_data) * 0.65)
    val_size = len(train_data) - train_size
    train_df, val_df = train_data[:train_size], train_data[train_size:]

    # stationary_test(train_df)
    xgb_regressor(train_data, show_ft_ip=True, pca=False)

    print("\nEnd of program")
