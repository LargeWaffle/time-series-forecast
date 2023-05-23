import xgboost as xgb
import lightgbm as lgbm
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from salesft import read_sales
from training import train_single_model, train_model_group
from extra import create_submission, create_frag_submission, show_feature_importance

DATAPATH = "data/store-sales"

split_size = 0.8
drop_cols = ['id', 'sales']

ft_infos = {
    'ft_scaler': StandardScaler(),
    'use_pca': False,
    'separator': 'family',
    'fragment': False,
    'dropped': drop_cols
}

train_data, test_data, feature_names = read_sales(DATAPATH, ft_infos)
train_size = int(len(train_data) * split_size)

model = xgb.XGBRegressor(n_estimators=350, importance_type='gain', eval_metric='rmse',
                         early_stopping_rounds=20, verbosity=1)

if ft_infos['fragment']:
    model_group = train_model_group(model, train_data, train_size, drop_cols)

    create_frag_submission(model_group, test_data, drop_cols, DATAPATH)
else:
    model = train_single_model(model, train_data, train_size, drop_cols)

    if model is xgb.XGBRegressor or lgbm.LGBMRegressor or RandomForestRegressor:
        features_val = model.feature_importances_
    else:
        features_val = model.coef_

    show_feature_importance(feature_names, features_val)

    create_submission(model, test_data, drop_cols, DATAPATH)


print("\nEnd of program")
