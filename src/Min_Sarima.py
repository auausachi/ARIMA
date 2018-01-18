#https://logics-of-blue.com/python-time-series-analysis/
#上記を参考に実装
#trafficデータは、構成変更前の11/20までのものを利用（11/20かは要確認

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
dateparse = lambda dates: pd.datetime.strptime(dates, '%Y/%m/%d %H:%M')
data = pd.read_csv('30minData.csv', index_col='time', date_parser=dateparse, dtype='float')

# 日付形式にする
ts = data['traffic']
ts
ts["2017-11-30 12:00:00"]
ts = ts.dropna()
ts_log = np.log(ts)
# 傾向(trend)、季節性(seasonal)、残差(residual)に分解してモデル化する。
from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(ts_log, freq=30)
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

# オリジナルの時系列データプロット
plt.subplot(411)
plt.plot(ts_log, label='Original')
plt.legend(loc='best')

# trend のプロット
plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='best')

# seasonal のプロット
plt.subplot(413)
plt.plot(seasonal,label='Seasonality')
plt.legend(loc='best')

# residual のプロット
plt.subplot(414)
plt.plot(residual, label='Residuals')
plt.legend(loc='best')
plt.tight_layout()



###### 総当たりで、AICが最小となるSARIMAの次数を探す
def Param_Sarima(max_p, max_q, max_d, max_sp , max_sq, max_sd) :
    pattern = max_p*(max_q + 1)*(max_d + 1)*(max_sp + 1)*(max_sq + 1)*(max_sd + 1)
    modelSelection = pd.DataFrame(index=range(pattern), columns=["model", "aic"])
    num=0
    for p in range(1, max_p + 1):
     for d in range(0, max_d + 1):
        for q in range(0, max_q + 1):
            for sp in range(0, max_sp + 1):
                for sd in range(0, max_sd + 1):
                    for sq in range(0, max_sq + 1):
                        sarima = sm.tsa.SARIMAX(
                            ts, order=(p,d,q),
                            seasonal_order=(sp,sd,sq,7),
                            enforce_stationarity = False,
                            enforce_invertibility = False
                        ).fit()
                        modelSelection.iloc[num]["model"] = "order=(" + str(p) + ","+ str(d) + ","+ str(q) + "), season=("+ str(sp) + ","+ str(sd) + "," + str(sq) + ")"
                        modelSelection.iloc[num]["aic"] = sarima.aic
                        num = num + 1
 # モデルごとの結果確認
    print(modelSelection)

 # AIC最小モデル
    print(modelSelection[modelSelection.aic == min(modelSelection.aic)])
###########

###### 残差、自己相関、変自己相関のチェック
def acf_check(residSARIMA) :
    # 残差のチェック
    fig = plt.figure(figsize=(12,8))

    # 自己相関
    ax1 = fig.add_subplot(211)
    fig = sm.graphics.tsa.plot_acf(residSARIMA, lags=4, ax=ax1)

    # 偏自己相関
    ax2 = fig.add_subplot(212)
    fig = sm.graphics.tsa.plot_pacf(residSARIMA, lags=4, ax=ax2)
###### 残差、自己相関、変自己相関のチェック


# Param_Sarima(1,1,1,3,3,2)

p=2
d=0
q=3
sp=1
sd=1
sq=2

sarima = sm.tsa.SARIMAX(
    ts, order=(p,d,q),
    seasonal_order=(sp,sd,sq,48), #24h*2=48
    enforce_stationarity = False,
    enforce_invertibility = False
).fit()
# 結果確認
print(sarima.summary())

#残差確認
# acf_check(sarima.resid)

# 予測
ts_pred = sarima.predict('2017-11-30 11:30:00', '2017-12-30 12:00:00')

# ts_pred = sarima.predict(50,5000)


# 実データと予測結果の図示
plt.plot(ts,"g", label='data')
plt.plot(ts_pred,"r", label='predicted')
plt.legend(loc='best')
plt.savefig("Min_Sarima.png")
