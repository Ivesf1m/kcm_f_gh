import numpy
from math import sqrt
from sklearn.model_selection import StratifiedKFold


def knn(training_set, train_fold, classes, xk, k):
    distances = {}
    count = 0

    for r in train_fold:
        row = training_set.iloc[r]
        d = (xk - row)
        d = numpy.dot(d, d)
        distances[count] = d
        count += 1
    distances = sorted(distances.items(), key=lambda x: x[1])

    class_counts = {}
    for c in classes:
        class_counts[c] = 0.0

    count = 0
    for t in distances:
        i = training_set.index[t[0]]
        if t[1] == 0.0:
            class_counts[i] += 1.0 / 1e-5
        else:
            class_counts[i] += 1.0 / sqrt(t[1])
        count += 1
        if count == k:
            break

    # print(class_counts)
    return max(class_counts, key=lambda key: class_counts[key])


def validate_k(training_set, classes, labels):
    ks = [2 * i + 1 for i in range(1, 10)]

    rates = []
    for k in ks:
        mean_rate = 0.0

        kf = StratifiedKFold(n_splits=10, shuffle=True)
        folds = kf.split(training_set, labels)
        for train, test in folds:
            error_rate = 0.0
            for i in test:
                xi = training_set.iloc[i]
                predicted_class = knn(training_set, train, classes, xi, k)
                if predicted_class != xi.name:
                    error_rate += 1.0
            error_rate /= len(test)
            mean_rate += error_rate
        mean_rate /= 10 # numero de folds
        rates.append(mean_rate)
    print(rates)
    m_index = numpy.argmin(rates)
    return 2 * (m_index + 1) + 1
