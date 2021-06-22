import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.stattools import adfuller


def getData():
    with open("task1.txt") as fin:
        data = list(map(float, fin.read().strip().split()))
        return np.array(data)


def subtask1():
    print("Start subtask #1")
    plt.plot(getData())
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


def subtask2():
    print("Start subtask #2")
    data = getData()
    tau_start = 1
    tau_end = len(data) - 2
    lags = list(range(tau_start, tau_end + 1))
    acf_res = ACF(data, tau_start, tau_end)
    plt.bar(lags, acf_res,
            width=0.2, color="black")
    plt.ylabel("ACF")
    plt.xlabel("Лаг")
    plt.title("Коррелограмма")
    plt.show()
    print("Lags: " + " ".join(list(map(str, range(tau_start, tau_end + 1)))))
    print("ACF: " + " ".join(["{:.4f}".format(c) for c in acf_res]))
    print("End subtask #2")


def subtask3():
    print("Start subtask #3")
    data = getData()
    m = len(data) // 2
    print("Дисперсии: ", "{:.4f}".format(data[:m].var()), " ", "{:.4f}".format(data[m:].var()))
    print("Матожидание: ", "{:.4f}".format(data[:m].mean()), " ", "{:.4f}".format(data[m:].mean()))
    test = adfuller(data)
    print("Проведём тест Дики-Фуллера")
    print("p-value: ", test[1])
    print("Critical values: ", test[4])
    if test[0] > test[4]["5%"]:
        print("Eсть единичные корни, ряд не стационарен")
    else:
        print("Единичных корней нет, ряд стационарен")
    print("End subtask #3")
    pass


subtask1()
subtask2()
subtask3()
