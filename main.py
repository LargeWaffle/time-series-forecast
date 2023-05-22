import pandas as pd

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

from salesft import read_sales

DATAPATH = "data/store-sales"


def create_submission(models_dict, test_df, drop_cols):
    submission_df = pd.read_csv(DATAPATH + '/sample_submission.csv')

    for fam, model in models_dict.items():
        current_df = test_df[fam]
        fam_df = current_df.drop(drop_cols, axis=1, errors='ignore')

        sales_values = model.predict(fam_df)
        sales_values[sales_values < 0] = 0

        current_df['sales'] = sales_values
        current_df = current_df[['id', 'sales']]

        submission_df = submission_df.merge(current_df, on='id', how='left')
        submission_df['sales'] = submission_df['sales_y'].fillna(submission_df['sales_x'])
        submission_df = submission_df.drop(['sales_x', 'sales_y'], axis=1)

    submission_df.to_csv('submission.csv', index=False)

    print("Submission saved!")


if __name__ == '__main__':
    split_size = 0.8
    drop_cols = ['id', 'sales']

    train_data, test_data = read_sales(DATAPATH, StandardScaler(), False)

    family_models = {}
    for family, data_df in train_data.items():
        family_model = xgb.XGBRegressor(n_estimators=200, importance_type='gain', eval_metric='rmse',
                                        early_stopping_rounds=20, verbosity=1)

        train_size = int(len(data_df) * split_size)
        val_size = len(data_df) - train_size
        train_df, val_df = data_df[:train_size], data_df[train_size:]

        x_train, y_train = train_df.drop(drop_cols, axis=1), train_df['sales']
        x_val, y_val = val_df.drop(drop_cols, axis=1), val_df['sales']

        family_model.fit(x_train, y_train, eval_set=[(x_val, y_val)])

        print(f"\nEvaluating family {family}")
        y_pred = family_model.predict(x_val)
        y_pred[y_pred < 0] = 0

        print("\nRegression metrics")
        print('MAE: {:.2f}'.format(mean_absolute_error(y_val, y_pred)))
        print('MSE: {:.2f}'.format(mean_squared_error(y_val, y_pred, squared=True)))
        print('RMSE: {:.2f}'.format(mean_squared_error(y_val, y_pred, squared=False)))
        print('R2: {:.2f}'.format(r2_score(y_val, y_pred)))

        family_models[family] = family_model

    create_submission(family_models, test_data, drop_cols)

    print("\nEnd of program")
