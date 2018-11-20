import numpy
from datasets import read_set, split_views, get_labels
from knn import knn, validate_k
from bayesian_classifier import train_bayesian_classifier, test_bayesian_classifier
from kcm_f_gh import kcm_f_gh
# from scipy.stats import friedmanchisquare
# from scikit_posthocs import posthoc_nemenyi_friedman
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.model_selection import StratifiedKFold


def test_kcm_f_gh(data_set, class_names, labels):
    num_clusters = len(class_names)
    num_rows = data_set.shape[0]
    best_ari = -2.0
    best_clusters = []
    best_parameters = []
    predicted_labels = numpy.full(num_rows, 'class')
    for i in range(0, 100):
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
    for i in range(0, 2):
        kf = StratifiedKFold(n_splits=10, shuffle=True)
        folds = kf.split(data_set, labels)
        for train, test in folds:
            class_probs, means, inv_cov_matrices = train_bayesian_classifier(
                data_set, classes, train)
            rate = test_bayesian_classifier(data_set, classes, test,
                                            class_probs, means, inv_cov_matrices)
            print(rate)
        print(' ')


classes = ['BRICKFACE', 'SKY', 'FOLIAGE', 'CEMENT', 'WINDOW', 'PATH', 'GRASS']
training_set = read_set('segmentation_mod.data')
training_shape_view, training_rgb_view = split_views(training_set)
labels = get_labels(training_set)

# removendo as colunas 3, 4 e 5 do training set original e da shape view
# pois todas tem variabilidade muito baixa (var -> 0)
training_set = training_set.drop(columns=['REGION-PIXEL-COUNT',
                                          'SHORT-LINE-DENSITY-5',
                                          'SHORT-LINE-DENSITY-2'])
training_shape_view = training_shape_view.drop(columns=['REGION-PIXEL-COUNT',
                                                        'SHORT-LINE-DENSITY-5',
                                                        'SHORT-LINE-DENSITY-2'])

# train_kcm_f_gh(training_shape_view, len(classes))
bayesian_classifier(training_set, classes, labels)
# iterator = training_shape_view.iterrows()
# c = knn.knn(training_shape_view, classes, next(iterator), 3)
# best_k = validate_k(training_set, classes, labels)
# print(best_k)
