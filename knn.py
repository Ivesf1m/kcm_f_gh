import numpy
from math import sqrt


def knn(training_set, classes, xk, k):
    distances = {}
    iterator = training_set.iterrows()
    count = 0
    xk_att = xk[1]
    for r in iterator:
        row = r[1]
        d = (xk_att - row)
        d = numpy.dot(d, d)
        distances[count] = d
        count += 1
    distances = sorted(distances.items(), key=lambda x: x[1])

    class_counts = {}
    for c in classes:
        class_counts[c] = 0.0

    count = 0
    while count < k:
        i = training_set.index[distances[count][0]]
        if distances[count][1] == 0:
            class_counts[i] += 1.0 / sqrt(distances[count][1] + 1e-5)
        else:
            class_counts[i] += 1.0 / sqrt(distances[count][1])
        count += 1
    print(class_counts)
    return max(class_counts, key=lambda x: x[1])



