import xgboost as xgb
import lightgbm as lgbm
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from statistics import mean

from salesft import read_sales
from extra import create_submission, create_frag_submission, show_feature_importance, show_metrics, plot_predictions

DATAPATH = "data/store-sales"

split_size = 0.7
drop_cols = ['id', 'sales']

ft_infos = {
    'ft_scaler': MinMaxScaler(),
    'separator': 'family',
    'fragment': True,
    'dropped': drop_cols
}

train_data, test_data, feature_names = read_sales(DATAPATH, ft_infos)

train_size = int(len(train_data) * split_size)
model = xgb.XGBRegressor(n_estimators=350, importance_type='gain', eval_metric='rmse',
                         early_stopping_rounds=20, verbosity=1)

if ft_infos['fragment']:

    metrics_dict = {"mae": [], "mse": [], "rmse": [], "r2": []}
    family_models = {}

    for family, data_df in train_data.items():
        train_size = int(len(data_df) * split_size)

        train_df, val_df = data_df[:train_size], data_df[train_size:]

        x_train, y_train = train_df.drop(drop_cols, axis=1), train_df['sales']
        x_val, y_val = val_df.drop(drop_cols, axis=1), val_df['sales']

        fit_model = model.fit(x_train, y_train, eval_set=[(x_val, y_val)])

        print(f"\nEvaluating model")
        y_pred = fit_model.predict(x_val)
        y_pred[y_pred < 0] = 0

        show_metrics(y_val, y_pred, metrics_dict)

        family_models[family] = fit_model

    print(f"\nTraining summary")
    print("Average MAE : {:.2f}".format(mean(metrics_dict['mae'])))
    print("Average MSE : {:.2f}".format(mean(metrics_dict['mse'])))
    print("Average RMSE : {:.2f}".format(mean(metrics_dict['rmse'])))
    print("Average R2 : {:.2f}".format(mean(metrics_dict['r2'])))

    create_frag_submission(family_models, test_data, drop_cols, DATAPATH)

else:
    train_df, val_df = train_data[:train_size], train_data[train_size:]

    x_train, y_train = train_df.drop(drop_cols, axis=1), train_df['sales']
    x_val, y_val = val_df.drop(drop_cols, axis=1), val_df['sales']

    model = model.fit(x_train, y_train, eval_set=[(x_val, y_val)])

    print(f"\nEvaluating model")
    y_pred = model.predict(x_val)
    y_pred[y_pred < 0] = 0

    show_metrics(y_val, y_pred, None)

    plot_predictions(len(x_val), y_val, y_pred)

    if model is xgb.XGBRegressor or lgbm.LGBMRegressor or RandomForestRegressor:
        features_val = model.feature_importances_
    else:
        features_val = model.coef_

    show_feature_importance(feature_names, features_val)

    create_submission(model, test_data, drop_cols, DATAPATH)

print("\nEnd of program")
