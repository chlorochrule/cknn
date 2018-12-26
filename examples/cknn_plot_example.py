# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import pdist, squareform
from sklearn.datasets import make_s_curve
from sklearn.neighbors import kneighbors_graph

from cknn import cknneighbors_graph, connect_rng

def make_kng(data, n_neighbors):
    k = n_neighbors
    dist = pdist(data)
    dmatrix = squareform(dist)
    argsorted_dmatrix = np.argsort(dmatrix)
    order_dmatrix = np.empty(argsorted_dmatrix.shape)
    n_samples = data.shape[0]
    rind = np.tile(np.arange(n_samples), (n_samples, 1)).T.flatten()
    cind = argsorted_dmatrix.flatten()
    order_dmatrix[rind, cind] = np.tile(np.arange(n_samples), n_samples)
    kng = connect_rng(order_dmatrix, k)
    k = kng.nonzero()[0].shape[0] // n_samples
    return kng, k

def load_datasets(datasets, n_samples, params=None, seed=None):
    np.random.seed(seed)
    if datasets == 'ball':
        if params is None:
            length = 5.0
            r_ab = 0.7
            dt = 0.18
            n_classes = 2
            t_dist = 'uniform'
            t_sigma = 0.64
            r_dist = 'uniform'
            r_sigma = 0.05
        else:
            length = params['length']
            r_ab = params['r_ab']
            dt = params['dt']
            n_classes = params['n_classes']
            t_dist = params['t_dist']
            t_sigma = params['t_sigma']
            r_dist = params['r_dist']
            r_sigma = params['r_sigma']

        if isinstance(n_samples, tuple):
            n_classes = len(n_samples)
            label = np.concatenate([np.full((n_samples[i],), i, dtype=np.int) for i in range(n_classes)])
            n_samples = sum(n_samples)
        else:
            n_samples_per_class = n_samples // n_classes
            n_samples = n_samples_per_class * n_classes
            label = np.tile(np.arange(n_classes), (n_samples_per_class, 1)).T.flatten()
        if r_dist == 'uniform':
            r_noise = (np.random.rand(n_samples) - 0.5) * r_sigma
        elif r_dist == 'normal':
            r_noise = np.random.randn(n_samples) * r_sigma
        else:
            raise ValueError("unexpected params r_dist={}".format(r_dist))
        r = length / 2 + r_noise
        yrad = np.random.rand(n_samples) * np.pi
        x = r * np.cos(yrad)
        y = np.zeros(x.shape, dtype=np.float)
        z = r * np.sin(yrad)
        if t_dist == 'uniform':
            t_noise = (np.random.rand(n_samples) - 0.5) * t_sigma / n_classes
        elif t_dist == 'normal':
            t_noise = np.random.randn(n_samples) * t_sigma / n_classes
        else:
            raise ValueError("unexpected params t_dist={}".format(t_dist))
        xrad = 2 * np.pi * (label/n_classes + x*dt + t_noise)
        y = r_ab * z * np.cos(xrad)
        z = r_ab * z * np.sin(xrad)
        data = np.concatenate((x.reshape(-1, 1),
                               y.reshape(-1, 1),
                               z.reshape(-1, 1)), axis=1)

    else:
        raise ValueError("unexpected keyword datasets={}".format(datasets))

    return data, label

def connect_points(ax, data, ng, color=None):
    source, target = ng.nonzero()
    source, target = source[source < target], target[source < target]
    for s, t in zip(source, target):
        if color is not None:
            if color[s] != color[t]:
                ax.plot(*data[[s,t],:].T, color='r')
            else:
                ax.plot(*data[[s,t],:].T, color='b')
        else:
            ax.plot(*data[[s,t],:].T, color='g')

def plot_graph(data, graph, color=None):
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(*data.T, c=color)
    connect_points(ax, data, graph, color=color)
    plt.show()
    plt.close()

def main():
    data_params = {
        'length': 5.0,
        'r_ab': 0.7,
        'dt': 0.18,
        'n_classes': 2,
        't_dist': 'uniform',
        't_sigma': 0.7,
        'r_dist': 'uniform',
        'r_sigma': 0.05,
    }
    n_neighbors = 3
    data, color = load_datasets(datasets='ball', n_samples=(100, 700), seed=0,
                                params=data_params)
    # ckng = cknneighbors_graph(data, n_neighbors=10, neighbors='delta',
    #                           delta=0.0, directed=False, connected=True,
    #                           conn_type='nature', return_instance=True)
    # n, labels = connected_components(ckng.ckng)
    # print(ckng.delta)
    # print(n)
    # plot_graph(data, ckng.ckng, color=color)

    ckng = cknneighbors_graph(data, n_neighbors=n_neighbors, neighbors='k', k=1,
                              directed=False, connected=True,
                              return_instance=True, verbose=1)
    n, labels = connected_components(ckng.ckng)
    n_ckng = n
    print(ckng.k)
    plot_graph(data, ckng.ckng, color=color)

    kng, k = make_kng(data, n_neighbors=n_ckng)
    n, labels = connected_components(kng)
    print(k)
    plot_graph(data, ckng.ckng, color=color)

    # ckng = cknneighbors_graph(data, n_neighbors=5, neighbors='k', k=4,
    #                           directed=True, connected=True, conn_type='force')
    # n, labels = connected_components(ckng)
    # print(n)
    # plot_graph(data, ckng, color=color)
    #
    # ckng = cknneighbors_graph(data, n_neighbors=3, neighbors='k', k=3,
    #                           directed=True, connected=False, conn_type='force')
    # n, labels = connected_components(ckng)
    # print(n)
    # plot_graph(data, ckng, color=color)

if __name__ == '__main__':
    main()
