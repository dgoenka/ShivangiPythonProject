from binance import Client
import pandas as pd
import numpy as np
client = Client()

def getdailydata(symbol):
    frame = pd.DataFrame(client.get_historical_klines(symbol,'1d', '3 years ago UTC'))
    frame = frame[[0,4]]
    frame.columns = ['Timestamp', symbol]
    frame = frame.set_index('Timestamp')
    frame = frame.astype(float)
    return frame

symbols = ['BTCUSDT','ETHUSDT','BNBUSDT','ADAUSDT']

prices = []

for symbol in symbols:
    prices.append(getdailydata(symbol))

df = pd.concat(prices, axis=1)

from pypfopt import EfficientFrontier, risk_models, expected_returns, plotting

mu = expected_returns.mean_historical_return(df, frequency=365)

S = risk_models.sample_cov(df, frequency=365)

# sharp
ef_sharpe = EfficientFrontier(mu, S)

weights_sharpe = ef_sharpe.max_sharpe()
print('Weights Sharp:\n')
print(weights_sharpe)

print('\n\nPerformance Sharp:\n')
performance = ef_sharpe.portfolio_performance(verbose=True)
print(performance)

# volatile
ef_volatile = EfficientFrontier(mu, S)

weights_volatile = ef_volatile.min_volatility()
print('Weights Min Volatile:\n')
print(weights_volatile)

print('\n\nPerformance Min Volatile:\n')
performance_volatile = ef_volatile.portfolio_performance(verbose=True)
print(performance_volatile)