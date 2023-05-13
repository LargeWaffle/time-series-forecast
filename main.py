from features import read_sales
from predictions import lreg, elastic, xgb_regressor, mlp, lgbm, rf

DATAPATH = "data"

if __name__ == '__main__':
    train_df, _, _ = read_sales(DATAPATH)

    # train-test split for time series
    train_size = int(len(train_df) * 0.67)
    val_size = len(train_df) - train_size
    train, val = train_df[:train_size], train_df[train_size:]

    # stationary_test(train_df)
    rf(train_df, show_ft_ip=True, pca=False)

    print("\nEnd of program")
