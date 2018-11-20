import numpy
from pandas import read_csv


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
