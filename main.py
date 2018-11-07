from datasets import read_set, split_views
# import knn
from bayesian_classifier import train_bayesian_classifier, test_bayesian_classifier
# from kcm_f_gh import train_kcm_f_gh

classes = ['BRICKFACE', 'SKY', 'FOLIAGE', 'CEMENT', 'WINDOW', 'PATH', 'GRASS']
training_set = read_set('segmentation.data')
training_shape_view, training_rgb_view = split_views(training_set)

# removendo as colunas 3, 4 e 5 do training set original e da shape view
# pois todas tem variabilidade muito baixa (var -> 0)
training_set = training_set.drop(columns=['REGION-PIXEL-COUNT',
                                          'SHORT-LINE-DENSITY-5',
                                          'SHORT-LINE-DENSITY-2'])
training_shape_view = training_shape_view.drop(columns=['REGION-PIXEL-COUNT',
                                          'SHORT-LINE-DENSITY-5',
                                          'SHORT-LINE-DENSITY-2'])

# train_kcm_f_gh(training_shape_view, len(classes))
class_probs, means, inv_cov_matrices = train_bayesian_classifier(
    training_shape_view, classes)
test_bayesian_classifier(training_shape_view, classes, class_probs, means,
                         inv_cov_matrices)
# iterator = training_shape_view.iterrows()
# c = knn.knn(training_shape_view, classes, next(iterator), 3)
# print(c)
