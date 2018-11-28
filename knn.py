import numpy
from math import sqrt
from sklearn.model_selection import StratifiedKFold


def knn(training_set, distance_matrix, train_fold, classes, xk_index, k):
    distances = {}
    count = 0

    distances = distance_matrix[xk_index, :]
    min_indices = numpy.argpartition(distances, k)
    min_indices = numpy.delete(min_indices, 0) # o proprio xk
    min_indices = min_indices[:k]

    class_counts = {}
    class_probs = {}
    for c in classes:
        class_counts[c] = 0.0
        class_probs[c] = 0.0

    count = 0
    for t in min_indices:
        i = training_set.index[t]
        if distance_matrix[xk_index][t] == 0.0:
            class_counts[i] += 1.0 / 1e-5
        else:
            class_counts[i] += 1.0 / distance_matrix[xk_index][t]
        class_probs[i] += 1.0
        count += 1
        if count == k:
            break

    # calculando a probabilidade para cada classe como k_i / k
    class_probs = [ki / k for ki in class_probs.values()]

    # print(class_counts)
    return max(class_counts, key=lambda key: class_counts[key]), class_probs


def calculate_distance_matrix(data_set):
    number_rows = data_set.shape[0]
    distance_matrix = numpy.zeros((number_rows, number_rows))

    for i in range(0, number_rows):
        xi = data_set.iloc[i]
        for j in range(i + 1, number_rows):
            xj = data_set.iloc[j]
            d = numpy.subtract(xi, xj)
            dist = sqrt(numpy.dot(d, d))
            distance_matrix[i][j] = distance_matrix[j][i] = dist

    return distance_matrix


def validate_k(training_set, classes, labels):
    ks = [2 * i + 1 for i in range(1, 10)]
    fold_count = 0
    distance_matrix = calculate_distance_matrix(training_set)
    print('matriz de distancias calculada')

    rates = []
    for k in ks:
        mean_rate = 0.0
        print('k = ', k)
        kf = StratifiedKFold(n_splits=10, shuffle=True)
        folds = kf.split(training_set, labels)
        for train, test in folds:
            print('fold ', fold_count)
            fold_count += 1
            error_rate = 0.0
            for i in test:
                xi = training_set.iloc[i]
                predicted_class, class_probs = \
                    knn(training_set, distance_matrix, train, classes, i, k)
                if predicted_class != xi.name:
                    error_rate += 1.0
            error_rate /= len(test)
            mean_rate += error_rate
        mean_rate /= 10 # numero de folds
        rates.append(mean_rate)
    print(rates)
    m_index = numpy.argmin(rates)
    return 2 * (m_index + 1) + 1
