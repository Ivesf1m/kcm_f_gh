import numpy


def mle(data_set, classes):
    # maximum likelihood estimation
    data_shape = data_set.shape
    class_count = numpy.full(len(classes), 0)
    class_mean = numpy.full((len(classes), data_shape[1]), 0.0)
    class_variance = numpy.full((len(classes), data_shape[1]), 0.0)

    iterator = data_set.iterrows()
    for r in iterator:
        row = r[1]
        for i in range(0, len(classes)):
            if r[0] == classes[i]:
                class_count[i] += 1
                for j in range(0, data_shape[1]):
                    class_mean[i][j] += row[j]

    for i in range(0, len(classes)):
        for j in range(0, data_shape[1]):
            class_mean[i][j] /= class_count[i]

    iterator = data_set.iterrows()
    for r in iterator:
        row = r[1]
        for i in range(0, len(classes)):
            if r[0] == classes[i]:
                for j in range(0, data_shape[1]):
                    class_variance[i][j] += (row[j] - class_mean[i][j]) ** 2

    for i in range(0, len(classes)):
        for j in range(0, data_shape[1]):
            class_variance[i][j] /= class_count[i]
            if class_variance[i][j] == 0:
                class_variance[i][j] = 1e-5

    class_count = class_count / data_shape[0]

    return class_count, class_mean, class_variance


def bayes_probability(xk, means, cov_matrices):
    max_prob = 0.0

def train_bayesian_classifier(data_set, classes):
    probs, means, variances = mle(data_set, classes)
    inv_cov_matrices = []
    for i in range(0, len(classes)):
        d = numpy.linalg.inv(numpy.diag(variances[i]))
        inv_cov_matrices.append(d)
    print(inv_cov_matrices)
