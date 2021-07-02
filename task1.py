import matplotlib.pyplot as plt
import numpy as np
import calendar
import datetime
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, acf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from scipy import stats
import pandas as pd
import warnings

warnings.filterwarnings("ignore")


def compress_data(data, n):
    compressed = [data[0]]
    for i in range(1, len(data)):
        if i % n == 0:
            compressed.append(data[i])
        else:
            compressed[len(compressed) - 1] += data[i]
    return np.array(compressed)


def getDates(size):
    dates = [datetime.date(2017, 1, 1)]
    for i in range(1, size):
        last = dates[len(dates) - 1]
        dates.append(datetime.date(
            last.year + int(last.month / 12), (i % 12) + 1, 1))
    return dates


def getData():
    with open("task1.txt") as fin:
        data = list(map(float, fin.read().strip().split()))
        return np.array(data)


def subtask1():
    print("Start subtask #1")
    data = getData()
    dates = getDates(len(data))
    plt.figure(figsize=(10, 4))
    plt.xticks(rotation=90)
    plt.plot(list(map(lambda x: x.strftime("%m.%y"), dates)), data)
    plt.ylabel("Объем экспорта тыс. долл")
    plt.title("график временного ряда")
    plt.show()
    print("End subtask #1")


def getMyCoefACF(x: np.ndarray, tau: int):
    size = len(x)
    a = np.array(x[tau:])
    b = np.array(x[:size - tau])
    return ((a * b).mean() - a.mean() * b.mean()) / (np.std(a) * np.std(b))


def ACF(x: np.ndarray, tau_start, tau_end):
    coefs = []
    for tau in range(tau_start, tau_end + 1):
        coefs.append(getMyCoefACF(x, tau))
    return coefs


def confidence_interval1(N, acf_size):
    return [[1.96 / np.sqrt(N)] * acf_size, [-1.96 / np.sqrt(N)] * acf_size]


def confidence_interval2(coefs, tau_start, tau_end, N):
    confidence_interval = []
    for k in range(tau_start, tau_end + 1):
        sum = 0
        for i in range(k - 1):
            sum += coefs[i] * coefs[i]
        confidence_interval.append(1.96 * np.sqrt((1 + 2 * sum) / N))
    return [confidence_interval, [e * -1 for e in confidence_interval]]


def subtask2():
    print("Start subtask #2")
    data = getData()
    for d, tau_start, tau_end in list([[data, 1, 16]]):
        lags = list(range(tau_start, tau_end + 1))
        acf_res = ACF(d, tau_start, tau_end)
        plt.bar(lags, acf_res, width=0.2, color="black")
        plt.xticks(lags, lags)
        N = len(d)
        conf1 = confidence_interval1(N, len(acf_res))
        conf2 = confidence_interval2(acf_res, tau_start, tau_end, N)
        plt.plot(lags, conf1[0], linewidth=1, color="green")
        plt.plot(lags, conf1[1], linewidth=1, color="green")
        plt.plot(lags, conf2[0], linewidth=1, color="blue")
        plt.plot(lags, conf2[1], linewidth=1, color="blue")
        plt.plot([tau_start, tau_end], [0.0, 0.0], linewidth=1, color="black")
        plt.ylabel("ACF")
        plt.xlabel("Лаг")
        plt.title("Коррелограмма")
        plt.show()
        print("Lags: " + " ".join(list(map(str, range(tau_start, tau_end + 1)))))
        print("ACF: " + " ".join(["{:.2f}".format(c) for c in acf_res]))

        plot_acf(d)
        plt.bar(lags, acf_res, width=0.2, color="black")
        plt.show()
    print("End subtask #2")
    # print("statsmodel ACF: " + " ".join(["{:.2f}".format(c) for c in acf(data, nlags=16, fft=True)[1:]]))


def subtask3():
    print("Start subtask #3")
    data = getData()
    m = len(data) // 2
    test = adfuller(data)
    print(test)
    print("Проведём тест Дики-Фуллера")
    print("adf: ", "{:.2f}".format(test[0]))
    print("p-value: ", "{:.2f}".format(test[1]))
    print("Critical values: ")
    print("\t1%:  ", "{:.2f}".format(test[4]["1%"]))
    print("\t5%:  ", "{:.2f}".format(test[4]["5%"]))
    print("\t10%: ", "{:.2f}".format(test[4]["10%"]))
    if test[0] > test[4]["5%"]:
        print("Eсть единичные корни, ряд не стационарен")
    else:
        print("Единичных корней нет, ряд стационарен")
    print("End subtask #3")


def moving_average(y_t, w, k):
    sum_w = sum(w)
    average = []
    n = len(y_t)
    for t in range(k - 1, n):
        x = 0
        for i in range(k):
            x += w[k - 1 - i] * y_t[t - i]
        average.append(x / sum_w)

    return average


def my_decompose(data, df, period, model):
    result = seasonal_decompose(df, model=model, period=period, extrapolate_trend='freq')
    trend = result.trend
    season = result.seasonal

    mae = ((data - (trend + season)) ** 2).mean()
    return result, mae


def subtask4():
    print("Start subtask #4")
    data = getData()
    dates = getDates(len(data))

    df = pd.DataFrame({"data": data}, index=pd.DatetimeIndex(dates))
    for model in ["additive", "multiplicative"]:
        print(f"Model = {model}")
        for period in range(2, 17):
            decompose, mae = my_decompose(data, df, period, model)
            trend = decompose.trend
            season = decompose.seasonal
            print("period: ", period)
            print("\t% данных, объяс. моделью:  ", 1 - mae / data.var())
            print("\tАбсолютная средняя ошибка: ", np.abs(data - (trend)).mean())

    decompose, mae = my_decompose(data, df, 2, "additive")
    decompose.plot()
    plt.show()
    plt.figure(figsize=(10, 4))
    plt.xticks(rotation=90)
    plt.plot(list(map(lambda x: x.strftime("%m.%y"), dates)), data)
    plt.plot(list(map(lambda x: x.strftime("%m.%y"), dates)), decompose.trend + decompose.seasonal)
    plt.title(f"Исходные данные, тренд, тренд + сезонность, period = {2}")
    plt.show()
    print("End subtask #4")


def subtask5():
    print("Start subtask #5")
    data = getData()
    dates = getDates(len(data))
    df = pd.DataFrame({"data": data}, index=pd.DatetimeIndex(dates))

    decompose = seasonal_decompose(df, model='additive', period=2, extrapolate_trend='freq')

    plt.figure(figsize=(10, 4))
    plt.xticks(rotation=90)
    plt.plot(list(map(lambda x: x.strftime("%m.%y"), dates)), data, label="Исходные данные")
    plt.plot(list(map(lambda x: x.strftime("%m.%y"), dates)), decompose.trend, label="Тренд")
    plt.title(f"Исходные данные, тренд, period = {2}")
    plt.legend()
    plt.show()

    # Средняя ошибка аппроксимации
    print("Period = ", 2)
    print("Средняя абсолютная ошибка тенденции T:   ",
          np.sqrt((np.square((data - decompose.trend) / len(data))).sum()) / data.mean())
    # линейный коэффициент корреляции
    T = decompose.trend
    print("Линейный коэффициент корреляции: ",
          (((data - data.mean()) / data.std()) * ((T - T.mean()) / T.std())).sum() / (len(data) - 1))

    # F критерий
    alpha = 0.05
    F = T.var() / data.var()
    df1 = len(data) - 1
    df2 = len(T) - 1
    p_value = stats.f.cdf(F, df1, df2)
    print("F критерий Фишера")
    print(f"\tВероятность превышения значения статистики: {p_value}")
    if alpha < p_value:
        print("\tНет достаточных оснований для отклонения нулевой гипотезы")
    else:
        print("\tЕсть достаточные основания для отклонения нулевой гипотезы")

    # t критерий стьюдента
    _, p_value = stats.ttest_ind(T.dropna(), data, equal_var=True)
    print("t критерий Стьюдента")
    print(f"\tВероятность превышения значения статистики: {p_value}")
    if alpha < p_value:
        print("\tНет достаточных оснований для отклонения нулевой гипотезы")
    else:
        print("\tЕсть достаточные основания для отклонения нулевой гипотезы")
    print("End subtask #5")


def subtask6():
    print("Start subtask #6")
    data = getData()
    dates = getDates(len(data))
    df = pd.DataFrame({"data": data}, index=pd.DatetimeIndex(dates))

    decompose = seasonal_decompose(df, model='additive', period=2, extrapolate_trend='freq')

    trend = decompose.trend
    season = decompose.seasonal

    mae = ((data - (trend + season)) ** 2).mean()
    print(f"MAE = {mae}")
    print("MAE составляет ", "{:.2f}".format((mae / data.var()) * 100), "% дисперсии")
    print("Модель объясняет ", "{:.2f}".format((1 - mae / data.var()) * 100), "% данных")

    plt.figure(figsize=(10, 4))
    plt.xticks(rotation=90)
    plt.plot(df.index, df.values, label="Исходные данные")
    plt.plot(df.index, trend + season, label="Тренд + сезонность")
    plt.legend()
    plt.show()

    print("End subtask #6")


def subtask7():
    print("Start subtask #7")
    data = getData()
    dates = pd.date_range('2017-01-01', periods=36, freq='MS')
    df = pd.DataFrame({"data": data}, index=pd.DatetimeIndex(dates))

    decompose = seasonal_decompose(df,
                                   model='additive',
                                   extrapolate_trend='freq')

    df_reconstructed = pd.concat(
        [decompose.seasonal,
         decompose.trend,
         decompose.resid,
         decompose.trend + decompose.resid,
         decompose.observed], axis=1)

    df_reconstructed.columns = ['seasonal', 'trend', 'remainders', 'seasonal_adj', 'actual_values']
    df_reconstructed.dropna(inplace=True)

    df_forecast = df_reconstructed.iloc[-4:, :]
    df_forecast = df_forecast.set_index(df_forecast.index.shift(4))
    df_forecast = df_forecast.drop('actual_values', axis=1)
    df_forecast[['trend', 'remainders', 'seasonal_adj']] = np.nan

    df_forecast['trend'] = df_reconstructed.loc[df_reconstructed.index[-1], 'trend']
    df_forecast['remainders'] = df_reconstructed.loc[df_reconstructed.index[-1], 'remainders']
    df_forecast['seasonal_adj'] = df_forecast['trend'] + df_forecast['remainders']
    df_forecast['forecast'] = df_forecast['seasonal_adj'] + df_forecast['seasonal']
    pd.set_option('display.max_columns', None)
    print(df_forecast.head(n=3))

    plt.rcParams.update({'figure.figsize': (20, 7)})
    df_reconstructed['actual_values'].plot()
    df_forecast['forecast'].plot()
    plt.show()

    residuals = df_reconstructed['actual_values'] - (df_reconstructed['seasonal'] + df_reconstructed['trend'])
    resid_std = residuals.std()

    df_forecast['h'] = range(1, 5)
    df_forecast['std_h'] = resid_std * np.sqrt(df_forecast['h'])

    df_forecast['lower_adj'] = (df_forecast['seasonal_adj'] - 1.96 * df_forecast['std_h'])
    df_forecast['upper_adj'] = (df_forecast['seasonal_adj'] + 1.96 * df_forecast['std_h'])
    df_forecast['lower'] = df_forecast['lower_adj'] + df_forecast['seasonal']
    df_forecast['upper'] = df_forecast['upper_adj'] + df_forecast['seasonal']
    df_forecast[['h', 'std_h', 'lower', 'upper']]
    print(df_forecast[['h', 'std_h', 'lower', 'upper']].head(n=3))
    df_reconstructed['actual_values'].plot(label="Исходные данные")
    df_forecast['forecast'].plot(label="Прогноз")
    plt.fill_between(df_forecast.index, df_forecast['lower'], df_forecast['upper'], color='k', alpha=.2)
    plt.legend()
    plt.show()

    print("End subtask #7")


def diff(df):
    diff_ = df - df.shift(1)
    diff_.dropna(inplace=True)
    return diff_


def do_adfuller(data):
    test = adfuller(data)
    print(test)
    print("Проведём тест Дики-Фуллера")
    print("adf: ", "{:.2f}".format(test[0]))
    print("p-value: ", "{:.2f}".format(test[1]))
    print("Critical values: ")
    print("\t1%:  ", "{:.2f}".format(test[4]["1%"]))
    print("\t5%:  ", "{:.2f}".format(test[4]["5%"]))
    print("\t10%: ", "{:.2f}".format(test[4]["10%"]))
    if test[0] > test[4]["5%"]:
        print("Eсть единичные корни, ряд не стационарен")
    else:
        print("Единичных корней нет, ряд стационарен")


def subtask8_9():
    print("Start subtask #8")
    data = getData()
    dates = pd.date_range('2017-01-01', periods=36, freq='MS')
    df = pd.DataFrame({"data": data}, index=pd.DatetimeIndex(dates))

    decompose = seasonal_decompose(df,
                                   model="additive",
                                   extrapolate_trend="freq")
    do_adfuller(df)
    diff_df = diff(diff(df))
    do_adfuller(diff_df)
    diff_df.plot(label="дифференцированный ряд")
    plt.title("Дифференцированный ряд")
    plt.legend()
    plt.show()

    plot_acf(diff_df, lags=15)
    plot_pacf(diff_df, lags=15)
    plt.legend()
    plt.show()

    # считаем коэффициенты минимизируя AIC
    import itertools

    p = q = range(0, 3)
    pdq = list(itertools.product(p, q))
    for i in range(len(pdq)):
        pdq[i] = (pdq[i][0], 0, pdq[i][1])
    combs = {}
    aics = []

    for combination in pdq:
        try:
            model_aic = ARIMA(diff_df, order=combination)
            model_aic = model_aic.fit()
            combs.update({model_aic.aic: combination})
            aics.append(model_aic.aic)
        except:
            continue
    best_aic = min(aics)
    model_aic = ARIMA(diff_df, order=combs[best_aic])
    model_aic = model_aic.fit()
    model_aic.resid.plot(kind='kde')
    plt.title("Ошибка с критерием AIC")
    plt.show()

    print("############################################")
    print("Оптимальные коэффициенты с AIC:  ", combs[best_aic])
    print("Ошибка полученного решения с AIC:", (abs(model_aic.resid)).sum() / len(data))

    diff_df.plot()
    model_aic.fittedvalues.plot()
    plt.title("Модель с критерием AIC")
    plt.show()

    # считаем коэффициенты минимизируя BIC
    combs = {}
    bics = []

    for combination in pdq:
        try:
            model_bic = ARIMA(diff_df, order=combination)
            model_bic = model_bic.fit()
            combs.update({model_bic.bic: combination})
            bics.append(model_bic.bic)
        except:
            continue
    best_bic = min(bics)
    model_bic = ARIMA(diff_df, order=combs[best_bic])
    model_bic = model_bic.fit()
    model_bic.resid.plot(kind='kde')
    plt.title("Ошибка с критерием BIC")
    plt.show()

    print("############################################")
    print("Оптимальные коэффициенты c BIC:  ", combs[best_bic])
    print("Ошибка полученного решения c BIC:", (abs(model_bic.resid)).sum() / len(data))

    diff_df.plot()
    model_bic.fittedvalues.plot()
    plt.title("Модель с критерием BIC")
    plt.show()

    plt.figure(figsize=(20, 4))
    diff_df.plot()
    model_aic.fittedvalues.plot()
    model_bic.fittedvalues.plot()
    plt.title("Сравнение моделей AIC и BIC")
    plt.show()
    print("End subtask #8")


subtask1()
subtask2()
subtask3()
subtask4()
subtask5()
subtask6()
subtask7()
subtask8_9()
