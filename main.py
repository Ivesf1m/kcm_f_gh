import numpy
import datasets
from knn import knn, validate_k
from bayesian_classifier import train_bayesian_classifier,\
    test_bayesian_classifier, bayes_probability
from kcm_f_gh import kcm_f_gh
from scipy import mean
from scipy.stats import sem, t, friedmanchisquare
from scikit_posthocs import posthoc_nemenyi_friedman
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.model_selection import StratifiedKFold


def confidence_interval(data, size=30):
    confidence = 0.95  # nivel de confianca
    n = 30
    m = mean(data)
    std_err = sem(data)
    error = std_err * t.ppf((1 + confidence) / 2, n - 1)
    inferior = m - error
    superior = m + error
    print('taxa de acerto media: ', m)
    print('[', inferior, ', ', superior, ']')


def test_kcm_f_gh(data_set, class_names, labels):
    num_clusters = len(class_names)
    num_rows = data_set.shape[0]
    best_ari = -2.0
    best_clusters = []
    best_parameters = []
    predicted_labels = numpy.full(num_rows, 'class')
    for i in range(0, 1):
        clusters, hyper_parameters = kcm_f_gh(data_set, num_clusters)
        for j in range(0, num_clusters):
            for k in clusters[j]:
                predicted_labels[k] = class_names[j]
        ari = adjusted_rand_score(labels, predicted_labels)
        if ari > best_ari:
            best_ari = ari
            best_clusters = clusters
            best_parameters = hyper_parameters

    # melhor resultado
    print(best_clusters)
    print(best_parameters)
    print(best_ari)


def bayesian_classifier(data_set, classes, labels):
    rates = numpy.zeros(30)
    for i in range(0, 30):
        kf = StratifiedKFold(n_splits=10, shuffle=True)
        folds = kf.split(data_set, labels)
        mean_rate = 0.0
        for train, test in folds:
            class_probs, means, inv_cov_matrices = train_bayesian_classifier(
                data_set, classes, train)
            rate = test_bayesian_classifier(data_set, classes, test,
                                            class_probs, means, inv_cov_matrices)
            mean_rate += rate
        mean_rate /= 10 # numero de folds
        rates[i] = mean_rate

    confidence_interval(rates)
    return rates


def test_knn(data_set, classes, labels, k):
    rates = numpy.zeros(30)
    for i in range(0, 30):
        kf = StratifiedKFold(n_splits=10, shuffle=True)
        folds = kf.split(data_set, labels)
        mean_rate = 0.0
        for train, test in folds:
            rate = 0.0
            for x in test:
                xk = data_set.iloc[x]
                predicted_class = knn(data_set, train, classes, xk, k)
                if xk.name == predicted_class:
                    rate += 1.0
            rate /= len(test)
            mean_rate += rate
        mean_rate /= 10 # numero de folds
        rates[i] = mean_rate

    confidence_interval(rates)
    return rates

def max_rule(data_set, view1, view2, classes, labels, ks):
    L = 3
    num_classes = len(classes)
    num_variables1 = data_set.shape[1]
    num_variables2 = view1.shape[1]
    num_variables3 = view2.shape[1]
    rates = numpy.zeros(30)

    for i in range(0, 30):
        kf = StratifiedKFold(n_splits=10, shuffle=True)
        folds = kf.split(data_set, labels)
        mean_rate = 0.0
        for train, test in folds:
            rate = 0.0

            class_probs1, means1, inv_cov_matrices1 = train_bayesian_classifier(
                data_set, classes, train)

            class_probs2, means2, inv_cov_matrices2 = train_bayesian_classifier(
                view1, classes, train)

            class_probs3, means3, inv_cov_matrices3 = train_bayesian_classifier(
                view2, classes, train)

            for x in test:
                x1 = data_set.iloc[x]
                x2 = view1.iloc[x]
                x3 = view2.iloc[x]

                probs1 = bayes_probability(num_variables1, x1, class_probs1,
                                           means1, inv_cov_matrices1)

                probs2 = bayes_probability(num_variables2, x2, class_probs2,
                                           means2, inv_cov_matrices2)

                probs3 = bayes_probability(num_variables3, x3, class_probs3,
                                           means3, inv_cov_matrices3)

                class_votes = numpy.zeros(num_classes)
                for j in range(0, num_classes):
                    class_votes[j] = (1 - L) * class_probs1[j] + L * max(
                        probs1[j], probs2[j], probs3[j])
                predicted_class = numpy.argmax(class_votes)
                if predicted_class == x1.name:
                    rate += 1.0
            rate /= len(test)
            mean_rate += rate
        mean_rate /= 10 # numero de folds
        rates[i] = mean_rate

    confidence_interval(rates)
    return rates

def compare_classifiers(data_set, view1, view2, classes, labels, ks):

    # taxas para o classificador bayesiano
    bayesian_rates_data_set = bayesian_classifier(data_set, classes, labels)
    bayesian_rates_view1 = bayesian_classifier(view1, classes, labels)
    bayesian_rates_view2 = bayesian_classifier(view2, classes, labels)

    # taxas para os knns
    knn_rates_data_set = test_knn(data_set, classes, labels, ks[0])
    knn_rates_view1 = test_knn(view1, classes, labels, ks[1])
    knn_rates_view2 = test_knn(view2, classes, labels, ks[2])

    # taxas para o classificador combinado
    max_rule_rates = max_rule(data_set, view1, view2, classes, labels, ks)

    statistic, pvalue = friedmanchisquare(bayesian_rates_data_set,
                                          bayesian_rates_view1,
                                          bayesian_rates_view2,
                                          knn_rates_data_set,
                                          knn_rates_view1, knn_rates_view2,
                                          max_rule_rates)

    if pvalue < 0.05:
        # rejeitando a hipotese de que nao existe diferenca entre
        # os classificadores
        rate_matrix = numpy.stack(bayesian_rates_data_set, bayesian_rates_view1,
                                  bayesian_rates_view2, knn_rates_data_set,
                                  knn_rates_view1, knn_rates_view2, max_rule_rates)
        posthoc_nemenyi_friedman(rate_matrix)


data_set_classes = ['BRICKFACE', 'SKY', 'FOLIAGE', 'CEMENT', 'WINDOW', 'PATH', 'GRASS']
training_set = datasets.read_set('segmentation_mod.data')
training_set = datasets.normalize_data_set(training_set)
training_shape_view, training_rgb_view = datasets.split_views(training_set)
data_set_labels = datasets.get_labels(training_set)

# removendo as colunas 3, 4 e 5 do training set original e da shape view
# pois todas tem variabilidade muito baixa (var -> 0)
training_set = training_set.drop(columns=['REGION-PIXEL-COUNT',
                                          'SHORT-LINE-DENSITY-5',
                                          'SHORT-LINE-DENSITY-2'])
training_shape_view = training_shape_view.drop(columns=['REGION-PIXEL-COUNT',
                                                        'SHORT-LINE-DENSITY-5',
                                                        'SHORT-LINE-DENSITY-2'])

test_kcm_f_gh(training_rgb_view, data_set_classes, data_set_labels)
# bayesian_classifier(training_set, data_set_classes, data_set_labels)
# test_knn(training_set, data_set_classes, data_set_labels, 3)

# best_k = validate_k(training_set, classes, labels)
# print(best_k)
