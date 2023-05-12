from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from statsmodels.tsa.stattools import adfuller


def stationary_test(df):
    ts = df['sales'].iloc[:15000]
    result = adfuller(ts)

    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))


def create_submission(sub_df, model):
    scaler = MinMaxScaler()
    sub_df = scaler.fit_transform(sub_df)
    pca = PCA(n_components=0.95)
    sub_df = pca.fit_transform(sub_df)

    test_sales = model.predict(sub_df)

    sub_df['sales'] = test_sales
    sub_df.to_csv('/kaggle/working/submission.csv', index=False)
