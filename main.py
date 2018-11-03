from datasets import read_set, split_views
# import knn
from bayesian_classifier import train_bayesian_classifier
from kcm_f_gh import kcm_f_gh

classes = ['BRICKFACE', 'SKY', 'FOLIAGE', 'CEMENT', 'WINDOW', 'PATH', 'GRASS']
training_set = read_set('segmentation.data')
training_shape_view, training_rgb_view = split_views(training_set)

# removendo a terceira coluna do training set original e da shape view
# pois sua variancia = 0
training_set = training_set.drop(columns=['REGION-PIXEL-COUNT'])
training_shape_view = training_shape_view.drop(columns=['REGION-PIXEL-COUNT'])

kcm_f_gh(training_shape_view, len(classes))
# train_bayesian_classifier(training_shape_view, classes)
# iterator = training_shape_view.iterrows()
# c = knn.knn(training_shape_view, classes, next(iterator), 3)
# print(c)
