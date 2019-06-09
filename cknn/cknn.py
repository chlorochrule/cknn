# -*- coding: utf-8 -*-

import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial.distance import pdist, squareform


def cknneighbors_graph(X, n_neighbors, delta=1.0, metric='euclidean', t='inf',
                       include_self=False, is_sparse=True, 
                       return_instance=False):

    cknn = CkNearestNeighbors(n_neighbors=n_neighbors, delta=delta,
                              metric=metric, t=t, include_self=include_self,
                              is_sparse=is_sparse)
    cknn.cknneighbors_graph(X)

    if return_instance:
        return cknn
    else:
        return cknn.ckng


class CkNearestNeighbors(object):
    def __init__(self, n_neighbors=5, delta=1.0, metric='euclidean', t='inf', 
                 include_self=False, is_sparse=True):
        self.n_neighbors = n_neighbors
        self.delta = delta
        self.metric = metric
        self.t = t
        self.include_self = include_self
        self.is_sparse = is_sparse
        self.ckng = None

    def cknneighbors_graph(self, X):
        n_neighbors = self.n_neighbors
        delta = self.delta
        metric = self.metric
        t = self.t
        include_self = self.include_self
        is_sparse = self.is_sparse

        n_samples = X.shape[0]

        if n_neighbors < 1 or n_neighbors > n_samples-1:
            raise ValueError("`n_neighbors` must be "
                             "in the range 1 to number of samples")
        if len(X.shape) != 2:
            raise ValueError("`X` must be 2D matrix")
        if n_samples < 2:
            raise ValueError("At least 2 data points are required")

        if metric == 'precomputed':
            if X.shape[0] != X.shape[1]:
                raise ValueError("`X` must be square matrix")
            dmatrix = X
        else:
            dmatrix = squareform(pdist(X, metric=metric))

        darray_n_nbrs = np.partition(dmatrix, n_neighbors)[:, [n_neighbors]]
        ratio_matrix = dmatrix / np.sqrt(darray_n_nbrs.dot(darray_n_nbrs.T))
        diag_ptr = np.arange(n_samples)

        if isinstance(delta, (int, float)):
            ValueError("Invalid argument type. "
                       "Type of `delta` must be float or int")
        adjacency = csr_matrix(ratio_matrix < delta)

        if include_self:
            adjacency[diag_ptr, diag_ptr] = True
        else:
            adjacency[diag_ptr, diag_ptr] = False

        if t == 'inf':
            neigh = adjacency.astype(np.float)
        else:
            mask = adjacency.nonzero()
            weights = np.exp(-np.power(dmatrix[mask], 2)/t)
            dmatrix[:] = 0.
            dmatrix[mask] = weights
            neigh = csr_matrix(dmatrix)

        if is_sparse:
            self.ckng = neigh
        else:
            self.ckng = neigh.toarray()

        return self.ckng


def main():
    np.random.seed(1)
    data = np.random.randn(20, 2)
    result = cknneighbors_graph(data, n_neighbors=7, delta=1.0,
                                metric='euclidean', t='inf',
                                include_self=True, is_sparse=False, 
                                return_instance=False)
    print(type(result))
    print(result)

if __name__ == '__main__':
    main()
