# -*- coding: utf-8 -*-

import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial.distance import pdist, squareform


def cknneighbors_graph(X, n_neighbors=5, delta=1.0, metric='euclidean',
                       t='inf', include_self=False, is_sparse=False):
    if n_neighbors < 1 or n_neighbors > X.shape[0]-1:
        raise Exception("Invalid number of neighbors")

    if metric == 'precomputed':
        dmatrix = X
    else:
        dist = pdist(X, metric=metric)
        dmatrix = squareform(dist)
    sorted_dmatrix = np.sort(dmatrix)
    nnei_dists = sorted_dmatrix[:, [n_neighbors]]
    eps_matrix = delta * np.sqrt(nnei_dists.dot(nnei_dists.T))
    adjacency = dmatrix < eps_matrix

    if not include_self:
        adjacency[np.eye(adjacency.shape[0], dtype=bool)] = False

    if t == 'inf':
        neigh = adjacency*1
    else:
        dmatrix[adjacency] = np.exp(np.power(dmatrix[adjacency], 2)/t)
        dmatrix[np.invert(adjacency)] = 0.
        neigh = dmatrix

    if is_sparse:
        return csr_matrix(neigh)
    else:
        return neigh

def main():
    np.random.seed(1)
    data = np.random.randn(20, 2)
    result = cknneighbors_graph(data, n_neighbors=7, delta=1.0)
    print(type(result))
    print(result)

if __name__ == '__main__':
    main()
