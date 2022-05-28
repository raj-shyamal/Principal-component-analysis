import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


stocks = ['GOOG', 'TSLA', 'MSFT', 'FB', 'AMZN', 'AAPL',
          'GS', 'NVDA', 'VLO', 'V', 'PEAK', 'PYPL', 'QCOM']

ohlc = {}


def download_data():

    for stock in stocks:

        ticker = yf.Ticker(stock)
        ohlc[stock] = ticker.history(period='2y', interval='1h')['Close']

    return pd.DataFrame(ohlc)


if __name__ == '__main__':

    df = download_data()
    df_returns = pd.DataFrame()

    for stock in stocks:
        df_returns[stock] = np.log(df[stock]).diff()

    df_returns.dropna(axis=1, how='all', inplace=True)
    df_returns.dropna(axis=0, how='any', inplace=True)

    # print(df_returns.shape)

    X = df_returns.to_numpy()
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    model = PCA()
    Z = model.fit_transform(X)

    print(X.shape)
    print(Z.shape)

    # plt.plot(model.explained_variance_ratio_)

    # plt.plot(model.explained_variance_ratio_[:10])

    cumulative_variance = np.cumsum(model.explained_variance_ratio_)
    # plt.plot(cumulative_variance)

    plt.plot(Z[:, 0])
    plt.show()
