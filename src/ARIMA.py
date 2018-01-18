#https://logics-of-blue.com/python-time-series-analysis/
#上記の実装バージョン

import numpy as np
import pandas as pd
from scipy import stats

from matplotlib import pylab as plt
import seaborn as sns
%matplotlib inline

from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15,6

import statsmodels.api as sm

import platform
print(platform.python_version())

dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m')
data = pd.read_csv('AirPassengers.csv', index_col='Month', date_parser=dateparse, dtype='float')
data.head()

# 日付形式にする
#ts = data['Traffic']
ts = data['#Passengers']

diff = ts - ts.shift()
diff = diff.dropna()

# 差分系列への自動ARMA推定関数の実行
resDiff = sm.tsa.arma_order_select_ic(diff, ic='aic', trend='nc')
resDiff

#resDiffの結果から、 P-3, q=2が最善となったので、それをモデル化
from statsmodels.tsa.arima_model import ARIMA
ARIMA_3_1_2 = sm.tsa.ARIMA(ts, order=(3, 1, 2)).fit(dist=False)
pred_ARIMA = ARIMA_3_1_2.predict('1960-01-01', '1961-12-01')

# SARIMAモデルを「決め打ち」で推定する
SARIMA_3_1_2_111 = sm.tsa.SARIMAX(ts, order=(3,1,2), seasonal_order=(1,1,1,12)).fit()
# print(SARIMA_3_1_2_111.summary())

#SARIMAの予測結果
pred_SARIMA = SARIMA_3_1_2_111.predict('1960-01-01', '1961-12-01')
pred_SARIMA
plt.plot(ts,"g", label="data")
plt.plot(pred_ARIMA, "--r", label="ARIMA_Prediction" )
plt.plot(pred_SARIMA, "--b", label="SARIMA_Prefdiction")
plt.legend()
plt.savefig("AirPassengers_ARIMA.png")







# SARIMA残差のチェック
# residSARIMA = SARIMA_3_1_1_111.resid
# fig = plt.figure(figsize=(12,8))
# ax1 = fig.add_subplot(211)
# fig = sm.graphics.tsa.plot_acf(residSARIMA.values.squeeze(), lags=40, ax=ax1)
# ax2 = fig.add_subplot(212)
# fig = sm.graphics.tsa.plot_pacf(residSARIMA, lags=40, ax=ax2)


# # 自己相関を求める
# ts_acf = sm.tsa.stattools.acf(ts.dropna(), nlags=907)
# ts_acf
# plt.plot(ts_acf)
#
# # 偏自己相関
# ts_pacf = sm.tsa.stattools.pacf(ts.dropna(), nlags=800, method='ols')
# ts_pacf
# plt.plot(ts_pacf)
