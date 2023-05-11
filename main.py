from features import read_sales
from predictions import xgb_regressor

DATAPATH = "data"

if __name__ == '__main__':
    train_df, _, _ = read_sales(DATAPATH)

    # stationary_test(train_df)
    xgb_regressor(train_df)

    print("\nEnd of program")
