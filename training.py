
from extra import plot_predictions, show_metrics


def make_splits(train_data, train_size, drop_cols):
    train_df, val_df = train_data[:train_size], train_data[train_size:]

    x_train, y_train = train_df.drop(drop_cols, axis=1), train_df['sales']
    x_val, y_val = val_df.drop(drop_cols, axis=1), val_df['sales']

    return x_train, x_val, y_train, y_val


def evaluate(model, x_val, y_val, show=False):
    print(f"\nEvaluating model")
    y_pred = model.predict(x_val)
    y_pred[y_pred < 0] = 0

    show_metrics(y_val, y_pred)

    if show:
        plot_predictions(len(x_val), y_val, y_pred)


def train_single_model(model, train_data, train_size, drop_cols):
    x_train, x_val, y_train, y_val = make_splits(train_data, train_size, drop_cols)

    single_model = model.fit(x_train, y_train, eval_set=[(x_val, y_val)])

    evaluate(single_model, x_val, y_val, show=True)

    return single_model


def train_model_group(model, train_data, train_size, drop_cols):

    family_models = {}
    for family, data_df in train_data.items():

        x_train, x_val, y_train, y_val = make_splits(data_df, train_size, drop_cols)
        family_model = model.fit(x_train, y_train, eval_set=[(x_val, y_val)])

        evaluate(family_model, x_val, y_val)

        family_models[family] = family_model

    return family_models
