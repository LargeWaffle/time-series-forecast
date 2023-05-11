from features import read_sales
from predictions import xgb_regressor

DATAPATH = "data"

if __name__ == '__main__':
    train_df, _, _ = read_sales(DATAPATH)

    # stationary_test(train_df)
    xgb_regressor(train_df, show_ft_ip=True, pca=False)

    print("\nEnd of program")
