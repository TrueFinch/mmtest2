import matplotlib.pyplot as plt
import numpy as np
import calendar
import datetime
from statsmodels.tsa.stattools import adfuller, acf
from statsmodels.graphics.tsaplots import plot_acf


def getData():
    with open("task1.txt") as fin:
        data = list(map(float, fin.read().strip().split()))
        return np.array(data)


def subtask1():
    print("Start subtask #1")
    data = getData()
    dates = [datetime.date(2017, 1, 1)]
    for i in range(1, len(data)):
        last = dates[len(dates) - 1]
        dates.append(datetime.date(
            last.year + int(last.month / 12), (i % 12) + 1, 1))
    plt.figure(figsize=(18, 4))
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
    tau_start = 1
    tau_end = 16
    lags = list(range(tau_start, tau_end + 1))
    acf_res = ACF(data, tau_start, tau_end)
    plt.bar(lags, acf_res, width=0.2, color="black")
    plt.xticks(lags, lags)
    N = len(data)
    conf1 = confidence_interval1(N, len(acf_res))
    conf2 = confidence_interval2(acf_res, tau_start, tau_end, N)
    plt.plot(lags, conf1[0], linewidth=1, color="green")
    plt.plot(lags, conf1[1], linewidth=1, color="green")
    plt.plot(lags, conf2[0], linewidth=1, color="blue")
    plt.plot(lags, conf2[1], linewidth=1, color="blue")
    plt.plot([1, 16], [0.0, 0.0], linewidth=1, color="black")
    plt.ylabel("ACF")
    plt.xlabel("Лаг")
    plt.title("Коррелограмма")
    plt.show()
    print("Lags: " + " ".join(list(map(str, range(tau_start, tau_end + 1)))))
    print("ACF: " + " ".join(["{:.2f}".format(c) for c in acf_res]))

    plot_acf(data)
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
    if test[0] > test[4]["10%"]:
        print("Eсть единичные корни, ряд не стационарен")
    else:
        print("Единичных корней нет, ряд стационарен")
    print("End subtask #3")

def subtask4():
    pass


# subtask1()
# subtask2()
subtask3()

