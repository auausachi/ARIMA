#https://logics-of-blue.com/python-time-series-analysis/
#上記を参考に実装

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

# 日付形式で読み込む（dtype=floatで読み込まないと、あとでARIMAモデル推定時にエラーとなる）
dateparse = lambda dates: pd.datetime.strptime(dates, '%Y/%m/%d')
data = pd.read_csv('Daily.csv', index_col='timestamp', date_parser=dateparse, dtype='float')
dateparse = lambda dates: pd.datetime.strptime(dates, '%Y/%m/%d')
data.tail()

# 日付形式にする
ts = data['Traffic']
ts
# プロット
#plt.plot(ts)

diff = ts - ts.shift()
diff = diff.dropna()

difff = diff - diff.shift()
difff = difff.dropna()

# 差分系列への自動ARMA推定関数の実行
resDiff = sm.tsa.arma_order_select_ic(difff, ic='aic', trend='nc')
resDiff

# P-3, q=1 が最善となったので、それをモデル化
# from statsmodels.tsa.arima_model import ARIMA
# ARIMA_3_1_1 = ARIMA(ts, order=(3, 1, 1)).fit()
# ARIMA_3_1_1.params


###### 総当たりで、AICが最小となるSARIMAの次数を探す
def Param_Arima(max_p, max_q, max_d) :
    pattern = max_p*(max_q + 1)*(max_d + 1)
    modelSelection = pd.DataFrame(index=range(pattern), columns=["model", "aic"])
    num=0
    for p in range(1, max_p + 1):
     for d in range(0, max_d + 1):
        for q in range(0, max_q + 1):
            arima = sm.tsa.ARIMA(ts, order=(p,d,q)).fit(dist=False)
            modelSelection.iloc[num]["model"] = "order=(" + str(p) + ","+ str(d) + ","+ str(q) + ")"
            modelSelection.iloc[num]["aic"] = arima.aic
            num = num + 1
 # モデルごとの結果確認
    print(modelSelection)
 # AIC最小モデル
    print(modelSelection[modelSelection.aic == min(modelSelection.aic)])
###########


# Param_Arima(2,2,2)

arima = sm.tsa.ARIMA(ts, order=(3,2,2)).fit(dist=False)













 # pred = ARIMA_3_1_1.predict('2015-11-01', '2017-12-30')
pred = arima.predict('2015-07-01', '2017-12-30')
#
plt.plot(ts,"g", label='data')
plt.plot(pred,"r", label='predicted')
plt.legend(loc='best')
plt.savefig("Daily_Arima.png")

residARIMA = arima.resid
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(residARIMA.values.squeeze(), lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(residARIMA, lags=40, ax=ax2)
plt.savefig("ACF_PACF_DailyARIMA_322")


# SARIMAモデルを「決め打ち」で推定する
# import statsmodels.api as sm
# sm.version.version

#seasonal_orderの値が不明。。元は12
#SARIMA_3_1_1_111 = sm.tsa.SARIMAX(ts, order=(3, 1, 2), seasonal_order=(1, 1, 1, 30)).fit()
#print(SARIMA_3_1_2_111.summary())
# SARIMA_3_1_2_111 = sm.tsa.SARIMAX(ts, order=(3,1,2), seasonal_order=(1,1,1,12)).fit()
# print(SARIMA_3_1_2_111.summary())

# residSARIMA = SARIMA_3_1_1_111.resid
# fig = plt.figure(figsize=(12,8))
# ax1 = fig.add_subplot(211)
# fig = sm.graphics.tsa.plot_acf(residSARIMA.values.squeeze(), lags=40, ax=ax1)
# ax2 = fig.add_subplot(212)
# fig = sm.graphics.tsa.plot_pacf(residSARIMA, lags=40, ax=ax2)


# pred = SARIMA_3_1_2_111.predict('1960-01-01', '1961-12-01')
# plt.plot(ts)
# plt.plot(pred, "r")


# 予測 Wrong number of items passedが。。
#pred = SARIMA_3_1_1_111.predict('2015-07-07', '2017-12-30')
# get_prediction(start='2015-07-07', end='2017-12-10', dynamic='2017-11-10')
#print(pred)

# 残差のチェック
# SARIMAじゃないので、周期性が残ってしまっている。。。
# resid = ARIMA_3_1_1.resid
# fig = plt.figure(figsize=(12,8))
# ax1 = fig.add_subplot(211)
# fig = sm.graphics.tsa.plot_acf(resid.values.squeeze(), lags=100, ax=ax1)
# ax2 = fig.add_subplot(212)
# fig = sm.graphics.tsa.plot_pacf(resid, lags=100, ax=ax2)



# # 自己相関を求める
# ts_acf = sm.tsa.stattools.acf(ts.dropna(), nlags=907)
# ts_acf
# plt.plot(ts_acf)
#
# # 偏自己相関
# ts_pacf = sm.tsa.stattools.pacf(ts.dropna(), nlags=800, method='ols')
# ts_pacf
# plt.plot(ts_pacf)
