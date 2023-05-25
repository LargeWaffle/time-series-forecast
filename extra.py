import pandas as pd


def format_data(data_df, split_size, drop_cols):
    train_size = int(len(data_df) * split_size)

    train_df, val_df = data_df[:train_size], data_df[train_size:]

    x_train, y_train = train_df.drop(drop_cols, axis=1), train_df['sales']
    x_val, y_val = val_df.drop(drop_cols, axis=1), val_df['sales']

    return x_train, x_val, y_train, y_val


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


def read_energy(data_path):
    train_df = pd.read_parquet(data_path + '/est_hourly.parquet')
    return train_df
