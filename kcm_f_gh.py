from math import exp, inf, sqrt
import numpy


def l2_distance(xi, xj):
    dist = 0.0
    for i in range(len(xi)):
        dist += (xi[i] - xj[i]) ** 2
    return sqrt(dist)


def calculate_initial_variance(data_set):
    # estimando a variancia inicial a partir dos quantis do data set
    number_rows = data_set.shape[0]
    distances = numpy.zeros((number_rows, number_rows))
    for i in range(0, number_rows):
        xi = data_set.iloc[i]
        for j in range(i + 1, number_rows):
            xj = data_set.iloc[j]
            distances[i][j] = distances[j][i] = l2_distance(xi, xj)
    quantile01 = numpy.quantile(distances, 0.1)
    quantile09 = numpy.quantile(distances, 0.9)
    return (quantile01 + quantile09) / 2.0


def gaussian_kernel(xl, xk, hyper_parameters):
    # essa funcao representa a equacao 9 do artigo
    acc = 0.0
    for j in range(len(hyper_parameters)):
        acc += hyper_parameters[j] * ((xl.iloc[0, j] - xk.iloc[0, j]) ** 2)
    return exp((-0.5) * acc)


def calculate_prototype_distance(xk, prototype, hyper_parameters):
    sum1 = 2.0 * gaussian_kernel(xk, prototype, hyper_parameters)
    return 1 - sum1


def calculate_distances(data_set, num_clusters, clusters, cluster_sizes, distance_matrix, hyper_parameters):
    # essa funcao representa a equacao 21 do artigo
    print('calculating distances')
    number_rows = data_set.shape[0]

    intra_cluster_distances = numpy.full(num_clusters, 0.0)
    for i in range(0, num_clusters):
        for r in range(0, cluster_sizes[i]):
            ri = clusters[i][r]
            xr = data_set[ri:ri+1]
            for s in range(r+1, cluster_sizes[i]):
                si = clusters[i][s]
                xs = data_set[si:si+1]
                intra_cluster_distances[i] += gaussian_kernel(xr, xs, hyper_parameters)
        intra_cluster_distances[i] /= cluster_sizes[i] ** 2

    for i in range(0, number_rows):
        xi = data_set[i:i+1]
        for j in range(0, num_clusters):
            sum1 = 0.0
            for k in range(0, cluster_sizes[j]):
                ki = clusters[j][k]
                xk = data_set[ki:ki+1]
                sum1 += gaussian_kernel(xi, xk, hyper_parameters)
            sum1 *= 2.0 / cluster_sizes[j]
            distance_matrix[j][i] = 1 - sum1 + intra_cluster_distances[j]


def assign_initial_clusters(data_set, prototypes, num_clusters, hyper_parameters):
    # Primeira atribuicao de clusters, utilizando a estimativa inicial de hiper
    # parametros. Os clusters estao vazios aqui.
    number_rows = data_set.shape[0]
    clusters = []
    cluster_sizes = numpy.full(num_clusters, 0)
    distance_matrix = numpy.zeros((num_clusters, data_set.shape[0]))

    for i in range(0, num_clusters):
        clusters.append([])

    for i in range(0, number_rows):
        min_dist = inf
        cluster_index = -1
        xi = data_set[i:i+1]
        for j in range(0, num_clusters):
            pj = prototypes[j:j+1]
            distance_matrix[j][i] = calculate_prototype_distance(xi, pj, hyper_parameters)
            if min_dist > distance_matrix[j][i]:
                min_dist = distance_matrix[j][i]
                cluster_index = j
        clusters[cluster_index].append(i)
        cluster_sizes[cluster_index] += 1
    return clusters, cluster_sizes


def update_hyper_parameters(data_set, clusters, num_clusters, cluster_sizes, hyper_parameters, gamma):
    # essa funcao representa a equacao 24 do artigo (com a errata)
    print('updating hyperparameters')
    intra_cluster = numpy.full(num_clusters, 0.0)
    number_variables = data_set.shape[1]
    exponent = 1.0 / number_variables
    prod = gamma ** exponent

    # numerador
    for h in range(0, number_variables):
        for i in range(0, num_clusters):
            for r in range(0, cluster_sizes[i]):
                ri = clusters[i][r]
                xr = data_set[ri:ri + 1]
                xrh = xr.iloc[0, h]
                for s in range(r + 1, cluster_sizes[i]):
                    si = clusters[i][s]
                    xs = data_set[si:si + 1]
                    xsh = xs.iloc[0, h]
                    intra_cluster[i] += gaussian_kernel(xr, xs, hyper_parameters) * ((xrh - xsh) ** 2)
            intra_cluster[i] /= cluster_sizes[i]
    prod *= numpy.prod(intra_cluster) ** exponent

    # denominador
    csum = numpy.zeros(number_variables)
    for j in range(0, number_variables):
        for i in range(0, num_clusters):
            for k in range(0, cluster_sizes[i]):
                indexk = clusters[i][k]
                ek = data_set[indexk:indexk + 1]
                ekj = ek.iloc[0, j]
                for l in range(k + 1, cluster_sizes[i]):
                    indexl = clusters[i][l]
                    el = data_set[indexl:indexl + 1]
                    elj = el.iloc[0, j]
                    csum[j] += gaussian_kernel(ek, el, hyper_parameters) * ((ekj - elj) ** 2)

    for i in range(0, number_variables):
        hyper_parameters[i] = prod / csum[i]


def update_clusters(data_set, clusters, num_clusters, cluster_sizes, hyper_parameters, gamma):
    number_rows = data_set.shape[0]
    print(cluster_sizes)
    print(hyper_parameters)

    test = True

    while test:
        # agora que os temos os clusters iniciais, podemos calcular novas distancias
        distance_matrix = numpy.zeros((num_clusters, number_rows))
        calculate_distances(data_set, num_clusters, clusters, cluster_sizes, distance_matrix, hyper_parameters)

        # estimando os novos hiper-parametros
        update_hyper_parameters(data_set, clusters, num_clusters, cluster_sizes, hyper_parameters, gamma)
        print('updated hyper parameters')
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

        print(to_remove)
        print(to_add)

        for i in range(0, num_clusters):
            for x in to_remove[i]:
                clusters[i].remove(x)
                cluster_sizes[i] -= 1

            for x in to_add[i]:
                clusters[i].append(x)
                cluster_sizes[i] += 1

        print('new cluster sizes')
        print(cluster_sizes)


def train_kcm_f_gh(data_set, num_clusters):
    # initialization step
    inv_variance = 1.0 / calculate_initial_variance(data_set)
    prototypes = data_set.sample(n=num_clusters)
    hyper_parameters = numpy.full(data_set.shape[1], inv_variance)
    gamma = inv_variance ** num_clusters
    clusters, cluster_sizes = assign_initial_clusters(data_set, prototypes, num_clusters, hyper_parameters)
    update_clusters(data_set, clusters, num_clusters, cluster_sizes, hyper_parameters, gamma)
    print('clusters')
    print(clusters)
    print('hyper parameters')
    print(hyper_parameters)

