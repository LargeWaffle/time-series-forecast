from extra import plot_predictions, show_metrics


def train_model(model, train_data, train_size, drop_cols, show):
    train_df, val_df = train_data[:train_size], train_data[train_size:]

    x_train, y_train = train_df.drop(drop_cols, axis=1), train_df['sales']
    x_val, y_val = val_df.drop(drop_cols, axis=1), val_df['sales']

    single_model = model.fit(x_train, y_train, eval_set=[(x_val, y_val)])

    print(f"\nEvaluating model")
    y_pred = model.predict(x_val)
    y_pred[y_pred < 0] = 0

    show_metrics(y_val, y_pred)

    if show:
        plot_predictions(len(x_val), y_val, y_pred)

    return single_model
