from scipy.cluster.hierarchy import linkage, dendrogram
from matplotlib import pyplot as plt

plt.rcParams["figure.figsize"] = (17, 8)

data = [
    [44.5, 84.98, 20.4, 3.2],
    [32.5, 30.58, 71.4, 8.5],
    [33.9, 38.42, 78.7, 9.2],
    [38.8, 60.34, 12.1, 3.3],
    [34.4, 60.22, 10.9, 3.2],
    [43.6, 60.79, 20.4, 5.4],
    [41.0, 29.82, 79.7, 8.3],
    [36.4, 70.57, 17.3, 5.4],
    [17.9, 34.51, 69.7, 7.1],
    [32.1, 64.73, 24.5, 6.0],
    [38.1, 36.63, 76.2, 8.6],
    [41.5, 32.84, 44.4, 5.7],
    [55.0, 62.64, 11.3, 3.5],
    [36.7, 34.07, 79.2, 6.7],
    [15.8, 39.27, 57.0, 6.7],
    [40.9, 28.46, 54.8, 7.3],
    [49.4, 30.27, 72.1, 8.5],
    [38.1, 69.04, 13.4, 3.3],
    [27.6, 25.42, 79.9, 10.2],
    [33.2, 53.13, 11.2, 3.4]
]
countries = [
    "Россия", "Австралия", "Австрия", "Азербайджан",
    "Армения", "Беларусь", "Бельгия", "Болгария",
    "Великобритания", "Венгрия", "Германия", "Греция",
    "Грузия", "Дания", "Ирландия", "Испания",
    "Италия", "Казахстан", "Канада", "Киргизия"
]


def build_dendogram(clusterd_date, labels):
    plt.figure()
    dendrogram(clusterd_date, orientation='right', labels=labels, distance_sort='descending', show_leaf_counts=True)
    plt.show()


def euclidean_nearest_neighbour_clustering(data, labels):
    clusterd_date = linkage(data, 'single', metric='euclidean')
    build_dendogram(clusterd_date, labels)


def subtask1():
    print("Start subtask #1")
    euclidean_nearest_neighbour_clustering(data, countries)
    print("End subtask #1")


def euclidean_further_neighbour_clustering(data, labels):
    clusterd_date = linkage(data, 'complete', metric='euclidean')
    build_dendogram(clusterd_date, labels)


def subtask2():
    print("Start subtask #2")
    euclidean_further_neighbour_clustering(data, countries)
    print("End subtask #2")


def mahalanobis_nearest_neighbour_clustering(data, labels):
    clusterd_date = linkage(data, 'single', metric='mahalanobis')
    build_dendogram(clusterd_date, labels)


def subtask3():
    print("Start subtask #3")
    mahalanobis_nearest_neighbour_clustering(data, countries)
    print("End subtask #3")


subtask1()
subtask2()
subtask3()
