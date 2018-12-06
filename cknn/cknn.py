# -*- coding: utf-8 -*-

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph
from networkx import has_path
import numba


@numba.jit
def concat_cluster(prev_dmatrix, n_components, labels, prev_indices=None):
    if prev_dmatrix.shape[0] != prev_dmatrix.shape[1]:
        raise ValueError("`prev_dmatrix` must be square matrix")

    next_dmatrix = np.empty((n_components, n_components), dtype=np.float)
    if prev_indices:
        next_indices = np.empty((n_components, n_components, 2), dtype=np.int)
    arange = np.arange(n_components)
    for i in arange:
        for j in arange:
            if i == j:
                next_dmatrix[i][j] = np.inf
            else:
                cropped_dmatrix = prev_dmatrix[labels == i][:, labels == j]
                d_argmin = np.argmin(cropped_dmatrix)
                next_dmatrix[i][j] = cropped_dmatrix.flatten()[d_argmin]
                if prev_indices:
                    cropped_indices = prev_indices[labels == i][:, labels == j]
                    next_indices[i][j] = cropped_indices.reshape(-1, 2)[d_argmin]

    if prev_indices:
        return next_dmatrix, next_indices
    else:
        return next_dmatrix

@numba.jit
def cut_rng(rng, k, critical_conn):
    rng_conn = rng + critical_conn
    r_rng, c_rng = rng.nonzero()
    conn_row, conn_col = critical_conn.nonzero()
    for i_row, n_conn in enumerate(np.bincount(conn_row)):
        if n_conn > 0:
            r_rng_part = r_rng[r_rng==i_row]
            c_rng_part = c_rng[r_rng==i_row]
            ind_indices = np.argsort(-rng[r_rng_part, c_rng_part])
            indices = np.concatenate((r_rng_part, c_rng_part)).reshape(2, -1)
            indices = indices[ind_indices]
            for i, j in indices:
                rng_conn[i, j] = 0
                if has_path(rng_conn, i, j) or has_path(rng_conn, j, i):
                    break
                else:
                    rng_conn[i, j] = 1
            else:
                raise Exception("Difficult to find connected graph")

    return rng_conn

@numba.jit
def connect_rng(dmatrix, radius, minimize=False, same_nbrs=False, verbose=0):
    rng = radius_neighbors_graph(dmatrix, radius, metric='precomputed')
    if minimize:
        rng_orig = rng
        rng_minimize = lil_matrix(rng.shape)
        nd = dmatrix.shape[0]
        indtile = np.tile(np.arange(nd), (nd, 1))
        indices = np.concatenate((indtile, indtile.T)).reshape(2, nd, nd)
    n_components, labels = connected_components(rng)
    cropped_dmatrix = dmatrix
    while n_components != 1:
        if minimize:
            cropped_dmatrix, indices = \
                concat_cluster(cropped_dmatrix, n_components, labels,
                               prev_indices=indices)
        else:
            cropped_dmatrix = concat_cluster(cropped_dmatrix, n_components,
                                             labels)
        argmin_dmatrix = np.argmin(cropped_dmatrix, axis=1)
        r_ptr = np.arange(cropped_dmatrix.shape[0])
        if minimize:
            rng = lil_matrix((n_components, n_components))
            rng[r_ptr, argmin_dmatrix] = 1.
            rng = csr_matrix(rng)
            r_ind, c_ind = indices[:, r_ptr, argmin_dmatrix]
            rng_minimize[r_ind, c_ind] = 1.
        else:
            radius = np.max(cropped_dmatrix[r_ptr, argmin_dmatrix])
            rng = radius_neighbors_graph(cropped_dmatrix, radius,
                                         metric='precomputed')
        n_components, labels = connected_components(rng)

    if verbose == 1:
        print("Log: radius={}".format(radius))

    if minimize:
        if minimize:
            return cut_rng(rng_orig, radius, rng_minimize)
        else:
            return rng_orig + rng_minimize
    else:
        return radius_neighbors_graph(dmatrix, radius, metric='precomputed')

@numba.jit
def cknneighbors_graph(X, n_neighbors, neighbors='delta', delta=None,
                       k=None, metric='euclidean', t='inf',
                       include_self=False, is_sparse=True, directed=False,
                       connected=False, conn_type='nature', verbose=0):

    if n_neighbors < 1 or n_neighbors > X.shape[0]-1:
        raise ValueError("Invalid argument `n_neighbors={}`"
                         .format(n_neighbors))
    if len(X.shape) != 2:
        raise ValueError("`X` must be 2d matrix")
    if X.shape[0] < 2:
        raise ValueError("At least 2 data points are required")

    if metric == 'precomputed':
        if X.shape[0] != X.shape[1]:
            raise ValueError("`X` must be square matrix")
        dmatrix = X
    else:
        dist = pdist(X, metric=metric)
        dmatrix = squareform(dist)

    darray_n_nbrs = np.partition(dmatrix, n_neighbors)
    ratio_matrix = dmatrix / np.sqrt(darray_n_nbrs.dot(darray_n_nbrs.T))
    cr_ptr = np.arange(X.shape[0])
    ratio_matrix[cr_ptr, cr_ptr] = 0

    if neighbors == 'delta':
        if not delta:
            ValueError("Invalid argument `delta={}`, or not passed delta"
                       .format(k))
        if connected:
            if conn_type == 'nature':
                adjacency = connect_rng(ratio_matrix, delta, verbose=verbose)
            elif conn_type == 'force':
                adjacency = connect_rng(ratio_matrix, delta, minimize=True,
                                        verbose=verbose)
        else:
            adjacency = radius_neighbors_graph(ratio_matrix, delta,
                                               metric='precomputed')
    elif neighbors == 'k':
        if not k:
            ValueError("Invalid argument `k={}`, or not passed k"
                       .format(k))
        if connected:
            argsorted_ratio_matrix = np.argsort(ratio_matrix)
            if conn_type == 'nature':
                adjacency = connect_rng(argsorted_ratio_matrix, k,
                                        verbose=verbose)
            elif conn_type == 'force':
                adjacency = connect_rng(argsorted_ratio_matrix, k,
                                        minimize=True, same_nbrs=True,
                                        verbose=verbose)
        else:
            adjacency = kneighbors_graph(ratio_matrix, k, metric='precomputed')
        if not directed:
            adjacency = lil_matrix(adjacency)
            adjacency = adjacency + adjacency.T
            adjacency.data[:] = 1.
            adjacency = csr_matrix(adjacency)
    else:
        raise ValueError("Invalid argument `neighbors={}`".format(neighbors))

    if include_self:
        adjacency[cr_ptr, cr_ptr] = 1.
    else:
        adjacency[cr_ptr, cr_ptr] = 0.

    if t == 'inf':
        neigh = adjacency
    else:
        mask = adjacency.nonzero()
        dmatrix[mask] = np.exp(np.power(dmatrix[mask], 2)/t)
        dmatrix[np.invert(mask)] = 0.
        neigh = csr_matrix(dmatrix)

    if is_sparse:
        return neigh
    else:
        return neigh.toarray()

def main():
    np.random.seed(1)
    data = np.random.randn(20, 2)
    result = cknneighbors_graph(data, n_neighbors=7, delta=1.0)
    print(type(result))
    print(result)

if __name__ == '__main__':
    main()
