# -*- coding: utf-8 -*-

import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial.distance import pdist, squareform


def cknneighbors_graph(X, n_neighbors=5, delta=1.0, metric='euclidean',
                       t='inf', include_self=True, is_sparse=False):
    if n_neighbors < 1 or n_neighbors > X.shape[0]-1:
        raise Exception("Invalid number of neighbors")

    dist = pdist(X, metric=metric) if metric != 'precomputed' else X
    dmatrix = squareform(dist)
    sorted_dmatrix = np.sort(dmatrix)
    nnei_dists = sorted_dmatrix[:, [n_neighbors]]
    eps_matrix = delta * np.sqrt(nnei_dists.dot(nnei_dists.T))
    adjacency = dmatrix < eps_matrix

    if not include_self:
        adjacency[np.eye(adjacency.shape[0], dtype=int)] = False

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
    data = np.arange(40).reshape(20, -1)
    result = cknneighbors_graph(data, n_neighbors=3, delta=0.5)
    print(type(result))
    print(result)

if __name__ == '__main__':
    main()
