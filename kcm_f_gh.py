from math import exp, inf, sqrt
import numpy


def squared_l2_distance(xi, xj):
    dist = numpy.dot(xi - xj, xi - xj)
    return dist


def calculate_initial_variance(data_set):
    # estimando a variancia inicial a partir dos quantis do data set
    number_rows = data_set.shape[0]
    distances = numpy.zeros((number_rows, number_rows))
    for i in range(0, number_rows):
        xi = data_set.iloc[i]
        for j in range(i + 1, number_rows):
            xj = data_set.iloc[j]
            distances[i][j] = distances[j][i] = squared_l2_distance(xi, xj)
    quantile01 = numpy.quantile(distances, 0.1)
    quantile09 = numpy.quantile(distances, 0.9)
    return (quantile01 + quantile09) / 2.0


def calculate_distance_matrix(data_set):
    number_rows = data_set.shape[0]
    number_variables = data_set.shape[1]
    distance_matrix = numpy.zeros((number_rows, number_rows, number_variables))
    for i in range(0, number_rows):
        xi = data_set.iloc[i]
        for j in range(i + 1, number_rows):
            xj = data_set.iloc[j]
            distance_matrix[i][j] = distance_matrix[j][i] =\
                numpy.power(xi - xj, 2)
    return distance_matrix


def gaussian_kernel(distance, hyper_parameters):
    # essa funcao representa a equacao 9 do artigo
    acc = numpy.sum(numpy.multiply(hyper_parameters, distance))
    return exp((-0.5) * acc)


def calculate_prototype_distance(xk, prototype, hyper_parameters):
    sum1 = 2.0 * gaussian_kernel(numpy.power(xk - prototype, 2),
                                 hyper_parameters)
    return 1 - sum1


def calculate_distances(data_set, num_clusters, clusters, cluster_sizes, dists, distance_matrix, hyper_parameters):
    # essa funcao representa a equacao 21 do artigo
    print('calculating distances')
    number_rows = data_set.shape[0]

    gaussian_kernel_matrix = numpy.zeros((number_rows, number_rows))
    for i in range(0, number_rows):
        for j in range(i + 1, number_rows):
            gaussian_kernel_matrix[i][j] = gaussian_kernel_matrix[j][i] = \
                gaussian_kernel(dists[i][j], hyper_parameters)

    intra_cluster_distances = numpy.zeros(num_clusters)
    for i in range(0, num_clusters):
        for r in range(0, cluster_sizes[i]):
            ri = clusters[i][r]
            for s in range(r+1, cluster_sizes[i]):
                si = clusters[i][s]
                intra_cluster_distances[i] += 2.0 * gaussian_kernel_matrix[ri][si]
        if cluster_sizes[i] != 0:
            intra_cluster_distances[i] /= (cluster_sizes[i] ** 2)

    for i in range(0, number_rows):
        for j in range(0, num_clusters):
            sum1 = 0.0

            if cluster_sizes[j] == 0:
                continue

            for k in range(0, cluster_sizes[j]):
                ki = clusters[j][k]
                sum1 += gaussian_kernel_matrix[i][ki]
            sum1 *= 2.0 / cluster_sizes[j]
            distance_matrix[j][i] = 1 - sum1 + intra_cluster_distances[j]


def assign_initial_clusters(data_set, prototypes, num_clusters, hyper_parameters):
    # Primeira atribuicao de clusters, utilizando a estimativa inicial de hiper
    # parametros. Os clusters estao vazios aqui.
    number_rows = data_set.shape[0]
    clusters = []
    cluster_sizes = numpy.zeros(num_clusters, dtype=int)
    distance_matrix = numpy.zeros((num_clusters, data_set.shape[0]))

    for i in range(0, num_clusters):
        clusters.append([])

    for i in range(0, number_rows):
        min_dist = inf
        cluster_index = -1
        xi = data_set.iloc[i]
        for j in range(0, num_clusters):
            pj = prototypes.iloc[j]
            distance_matrix[j][i] = calculate_prototype_distance(
                xi, pj, hyper_parameters)
            if min_dist > distance_matrix[j][i]:
                min_dist = distance_matrix[j][i]
                cluster_index = j
        clusters[cluster_index].append(i)
        cluster_sizes[cluster_index] += 1
    return clusters, cluster_sizes


def update_hyper_parameters(data_set, clusters, num_clusters, cluster_sizes, dists, hyper_parameters, gamma):
    # essa funcao representa a equacao 24 do artigo (com a errata)
    print('updating hyperparameters')
    intra_cluster = numpy.zeros(num_clusters)
    number_rows = data_set.shape[0]
    number_variables = data_set.shape[1]
    exponent = 1.0 / number_variables
    prod = gamma ** exponent

    # construindo uma matriz com os valores do kernel gaussiano aplicado a cada
    # par de exemplos do conjunto e multiplicadas pela diferenca em cada variavel
    lookup_matrix = numpy.zeros((number_rows, number_rows, number_variables))
    for i in range(0, number_rows):
        for j in range(i + 1, number_rows):
            g = gaussian_kernel(dists[i][j], hyper_parameters)
            for k in range(0, number_variables):
                lookup_matrix[i][j][k] = lookup_matrix[j][i][k] =\
                    g * dists[i][j][k]

    # numerador
    for h in range(0, number_variables):
        for i in range(0, num_clusters):
            if cluster_sizes[i] == 0:
                continue

            for r in range(0, cluster_sizes[i]):
                kr = clusters[i][r]
                for s in range(r + 1, cluster_sizes[i]):
                    ks = clusters[i][s]
                    intra_cluster[i] += 2.0 * lookup_matrix[kr][ks][h]

    for i in range(0, num_clusters):
        if cluster_sizes[i] != 0:
            intra_cluster[i] /= cluster_sizes[i]
    prod *= numpy.prod(intra_cluster) ** exponent

    # denominador
    cluster_sums = numpy.zeros(number_variables)
    for j in range(0, number_variables):
        for i in range(0, num_clusters):
            if cluster_sizes[i] == 0:
                continue

            sum_i = 0.0
            for k in range(0, cluster_sizes[i]):
                kk = clusters[i][k]
                for l in range(k + 1, cluster_sizes[i]):
                    kl = clusters[i][l]
                    sum_i += 2.0 * lookup_matrix[kk][kl][j]
            cluster_sums[j] += sum_i / cluster_sizes[i]

    for i in range(0, number_variables):
        hyper_parameters[i] = prod / cluster_sums[i]


def update_clusters(data_set, clusters, num_clusters, cluster_sizes, dists, hyper_parameters, gamma):
    number_rows = data_set.shape[0]
    print(cluster_sizes)
    print(hyper_parameters)

    test = True
    num_iterations = 0

    while test and num_iterations < 10:
        # agora que temos os clusters iniciais, podemos calcular novas distancias
        distance_matrix = numpy.zeros((num_clusters, number_rows))
        calculate_distances(data_set, num_clusters, clusters, cluster_sizes, dists,
                            distance_matrix, hyper_parameters)

        # estimando os novos hiper-parametros
        update_hyper_parameters(data_set, clusters, num_clusters,
                                cluster_sizes, dists, hyper_parameters, gamma)
        print(hyper_parameters)

        # etapa de alocacao
        test = False
        to_remove = [[] for i in range(0, num_clusters)]
        to_add = [[] for i in range(0, num_clusters)]
        for i in range(0, num_clusters):
            for j in range(0, cluster_sizes[i]):
                xj = clusters[i][j]
                xj_distances = distance_matrix[:, xj]
                ci = numpy.argmin(xj_distances)
                if ci != i:
                    to_remove[i].append(xj)
                    to_add[ci].append(xj)
                    test = True

        for i in range(0, num_clusters):
            for x in to_remove[i]:
                clusters[i].remove(x)
                cluster_sizes[i] -= 1

            for x in to_add[i]:
                clusters[i].append(x)
                cluster_sizes[i] += 1

        print('new cluster sizes')
        print(cluster_sizes)
        num_iterations += 1
    return clusters, hyper_parameters


def kcm_f_gh(data_set, num_clusters, dists):
    # initialization step
    # inv_variance = 1.0 / calculate_initial_variance(data_set)
    inv_variance = 0.5335807721826547 # calculado com a linha anterior
    prototypes = data_set.sample(n=num_clusters, replace=False)
    num_variables = data_set.shape[1]
    hyper_parameters = numpy.full(num_variables, inv_variance)
    gamma = inv_variance ** num_variables
    clusters, cluster_sizes = assign_initial_clusters(
        data_set, prototypes, num_clusters, hyper_parameters)
    clusters, hyper_parameters = update_clusters(
        data_set, clusters, num_clusters, cluster_sizes, dists,
        hyper_parameters, gamma)
    print('clusters')
    print(clusters)
    print('hyper parameters')
    print(hyper_parameters)
    return clusters, hyper_parameters

