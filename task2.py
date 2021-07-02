import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import pca

with open("data.txt", "r") as file_in:
    global_labels = file_in.readline().strip().split()
    global_data = file_in.read()
    global_data = global_data.strip().split('\n')
    for i in range(len(global_data)):
        global_data[i] = list(map(float, global_data[i].strip().split()))
    global_data = pd.DataFrame(global_data)


def build_scattering_plots(data, labels):
    N = len(data)
    for i in range(N):
        for j in range(i, N):
            if i == j:
                continue
            plt.title(f"x{i + 1} и x{j + 1}")
            plt.scatter(data[i], data[j])
            plt.xlabel(labels[i])
            plt.ylabel(labels[j])
            plt.show()


def build_biplot(data):
    model = pca.pca(n_components=0.95)
    model.fit_transform(data.to_numpy())
    model.biplot(n_feat=5)
    plt.show()


def calculate_corr_coefs(data):
    N = len(data)
    result = [[0.0 for _ in range(N)] for _ in range(N)]
    for i in range(N):
        for j in range(N):
            result[i][j] = np.corrcoef(data[i], data[j])[0, 1]
    return result


def PrintCoeffs(coeffs, labels):
    n = len(coeffs)
    for i in range(n):
        for j in range(i, n):
            if i == j:
                continue
            print()


def subtask1():
    print("Start subtask #1")
    build_scattering_plots(global_data.to_numpy().transpose(), global_labels)
    build_biplot(global_data)
    corr_coefs = calculate_corr_coefs(global_data.to_numpy().transpose())
    N = 5
    for i in range(N):
        for j in range(i, N):
            if i == j:
                continue
            print(f"Коэффициент корреляции x{i + 1} и x{j + 1}: {corr_coefs[j][i]}")
    print("End subtask #1")


def subtask2():
    print("Start subtask #2")
    centred_data = global_data - global_data.mean()
    print(centred_data)
    norm_data = centred_data / centred_data.std()
    print(norm_data)
    print("End subtask #2")


def convert_to_center_norm(data):
    new_data = data - data.mean()
    new_data = new_data / new_data.std()
    return new_data


def subtask3():
    print("Start subtask #3")
    center_norm_data = convert_to_center_norm(global_data)
    cov_m = center_norm_data.cov()
    print(cov_m)
    print("End subtask #3")


def get_cov_matrix(data):
    center_norm_data = convert_to_center_norm(global_data)
    cov_m = center_norm_data.cov()
    return cov_m


def subtask4_5():
    print("Start subtask #4")
    cov_matrix = get_cov_matrix(global_data)
    e_values, e_vectors = np.linalg.eig(cov_matrix.to_numpy())
    indexes = np.argsort(-e_values)
    print("Собственные значения: ", e_values)
    e_vectors = e_vectors[indexes]
    print("Собственные векторы:  ", e_vectors)
    print("End subtask #4")


# subtask1()
# subtask2()
# subtask3()
subtask4_5()
