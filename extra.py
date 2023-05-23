import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def create_submission(model, test_df, drop_cols, datapath):
    sales_df = test_df.drop(drop_cols, axis=1, errors='ignore')

    test_sales = model.predict(sales_df)

    submission_df = pd.read_csv(datapath + '/sample_submission.csv')
    submission_df['sales'] = test_sales
    submission_df.to_csv('submission.csv', index=False)

    print("Submission saved!")


def create_frag_submission(models_dict, test_df, drop_cols, datapath):
    submission_df = pd.read_csv(datapath + '/sample_submission.csv')

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


def show_metrics(y_val, y_pred):
    print("\nRegression metrics")
    print('MAE: {:.2f}'.format(mean_absolute_error(y_val, y_pred)))
    print('MSE: {:.2f}'.format(mean_squared_error(y_val, y_pred, squared=True)))
    print('RMSE: {:.2f}'.format(mean_squared_error(y_val, y_pred, squared=False)))
    print('R2: {:.2f}'.format(r2_score(y_val, y_pred)))


def show_feature_importance(feature_names, ft_val):
    plt.figure(figsize=(12, 6))

    (pd.Series(ft_val, index=feature_names)
     .sort_values(ascending=True)
     .plot(kind='barh'))

    plt.show()


def plot_predictions(nb_samples, y_val, y_pred):
    sp_list = list(range(0, nb_samples))
    plt.figure(figsize=(10, 6))

    plt.plot(sp_list, y_val, label='Expected', alpha=0.5)
    plt.plot(sp_list, y_pred, label='Predicted', alpha=0.5)
    plt.legend(loc="upper right")
    plt.show()

    plt.plot(sp_list, abs(y_val - y_pred), label='Difference')
    plt.legend(loc="upper right")
    plt.show()


def read_energy(data_path):
    train_df = pd.read_parquet(data_path + '/est_hourly.parquet')
    return train_df
