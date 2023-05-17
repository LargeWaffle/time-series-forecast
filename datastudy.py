import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
import seaborn as sns


def corr_matrix(df):
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=False, linewidth=.5, vmin=-1, vmax=1, fmt=".2f", square=True)
    plt.show()


def stationary_test(df):
    ts = df['sales'].iloc[:15000]
    result = adfuller(ts)

    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))
