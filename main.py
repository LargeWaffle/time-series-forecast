import xgboost as xgb
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from statistics import mean

from salesft import read_sales
from extra import create_submission, create_frag_submission, format_data
from datastudy import show_feature_importance, show_metrics, plot_predictions

if __name__ == '__main__':
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

    model = xgb.XGBRegressor(n_estimators=350, importance_type='gain', eval_metric='rmse',
                             early_stopping_rounds=20, verbosity=1)

    if ft_infos['fragment']:

        metrics_dict = {"mae": [], "mse": [], "rmse": [], "r2": []}
        family_models = {}

        for family, data_df in train_data.items():
            x_train, x_val, y_train, y_val = format_data(data_df, split_size, drop_cols)

            fit_model = model.fit(x_train, y_train, eval_set=[(x_val, y_val)])

            print(f"\nEvaluating family model {family}")
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
        x_train, x_val, y_train, y_val = format_data(train_data, split_size, drop_cols)

        model = model.fit(x_train, y_train, eval_set=[(x_val, y_val)])

        print(f"\nEvaluating model")
        y_pred = model.predict(x_val)
        y_pred[y_pred < 0] = 0

        show_metrics(y_val, y_pred, None)
        plot_predictions(len(x_val), y_val, y_pred)

        features_val = model.feature_importances_
        show_feature_importance(feature_names, features_val)

        create_submission(model, test_data, drop_cols, DATAPATH)

    print("\nEnd of program")
