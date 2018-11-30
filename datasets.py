import numpy
from pandas import read_csv
from scipy import mean
from scipy.stats import sem


def read_set(filename):
    with open(filename) as file:
        data = read_csv(file, skiprows=3)
    return data


def split_views(data_set):
    views = numpy.split(data_set, [9], axis=1)
    return views[0], views[1]


def get_labels(data_set):
    iterator = data_set.iterrows()
    labels = []
    for r in iterator:
        labels.append(r[0])
    return labels


def describe_data_set(data_set):
    shape = data_set.shape
    print('Number of rows: ', shape[0])
    print('Number of variables: ', shape[1])

    print('Colunas: ')
    for c in data_set.columns:
        print(c)

    print('Media de cada coluna: ')
    for c in data_set.columns:
        print(mean(data_set[c]))

    print('Desvio padrao de cada coluna:')
    for c in data_set.columns:
        print(sem(data_set[c]))


def normalize_data_set(data_set):
    for column in data_set.columns:
        #data_set[column] = ((data_set[column] - mean(data_set[column])) /
        #                    sem(data_set[column]))
        data_set[column] = ((data_set[column] - numpy.min(data_set[column])) /
                            (numpy.max(data_set[column]) -
                             numpy.min(data_set[column])))
    return data_set
