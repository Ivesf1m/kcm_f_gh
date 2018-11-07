import numpy
from math import exp, pi


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
            if class_variance[i][j] == 0.0:
                class_variance[i][j] = 1e-5
                print(i, j)

    class_count = class_count / data_shape[0]

    return class_count, class_mean, class_variance


def prior_probability(num_variables, xk, mean, inv_cov_matrix):
    prob = (2 * pi) ** (-num_variables / 2)
    det = numpy.linalg.det(inv_cov_matrix)
    prob *= det ** -0.5
    diff = numpy.matrix(xk - mean)
    prob *= exp(-0.5 * diff * inv_cov_matrix * numpy.transpose(diff))
    return prob


def bayes_probability(num_variables, xk, class_probs, means, inv_cov_matrices):
    num_classes = len(class_probs)
    priors = numpy.zeros(num_classes)
    for i in range(0, num_classes):
        priors[i] = prior_probability(num_variables, xk, means[i], inv_cov_matrices[i])
    evidence = numpy.sum(numpy.multiply(priors, class_probs))
    probs = [priors[i] * class_probs[i] / evidence for i in range(0, num_classes)]
    return probs


def train_bayesian_classifier(data_set, classes):
    probs, means, variances = mle(data_set, classes)
    inv_cov_matrices = []
    variances = [numpy.diag(v) for v in variances]
    for i in range(0, len(classes)):
        d = numpy.linalg.inv(variances[i])
        inv_cov_matrices.append(d)
    return probs, means, inv_cov_matrices


def test_bayesian_classifier(test_set, classes, class_probs, means,
                             inv_cov_matrices):
    num_variables = test_set.shape[1]
    num_classes = len(classes)
    iterator = test_set.iterrows()
    confusion_matrix = numpy.zeros((num_classes, num_classes))

    print(means)
    print(inv_cov_matrices[2])

    for row in iterator:
        real_class = -1
        for i in range(0, num_classes):
            if classes[i] == row[0]:
                real_class = i
        xk = row[1]
        xk_probs = bayes_probability(num_variables, xk, class_probs, means,
                                     inv_cov_matrices)
        new_class = numpy.argmax(xk_probs)
        confusion_matrix[real_class, new_class] += 1
    print(confusion_matrix)
