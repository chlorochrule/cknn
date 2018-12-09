# -*- coding: utf-8 -*-

from sklearn.datasets import make_s_curve
from scipy.sparse.csgraph import connected_components
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

from cknn import cknneighbors_graph


def connect_points(ax, data, ng):
    source, target = ng.nonzero()
    source, target = source[source < target], target[source < target]
    for s, t in zip(source, target):
        ax.plot(*data[[s,t],:].T, color='r')

def plot_graph(data, graph, color=None):
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(*data.T, c=color)
    connect_points(ax, data, graph)
    plt.show()
    plt.close()

def main():
    data, color = make_s_curve(300, random_state=0)
    ckng = cknneighbors_graph(data, n_neighbors=3, neighbors='delta',
                              delta=0.0, directed=False, connected=True)
    n, labels = connected_components(ckng)
    print(n)
    plot_graph(data, ckng, color=color)

    ckng = cknneighbors_graph(data, n_neighbors=3, neighbors='k', k=2,
                              directed=False, connected=True)
    n, labels = connected_components(ckng)
    print(n)
    plot_graph(data, ckng, color=color)

    ckng = cknneighbors_graph(data, n_neighbors=3, neighbors='k', k=4,
                              directed=True, connected=True, conn_type='force')
    n, labels = connected_components(ckng)
    print(n)
    plot_graph(data, ckng, color=color)

    ckng = cknneighbors_graph(data, n_neighbors=3, neighbors='k', k=3,
                              directed=True, connected=False, conn_type='force')
    n, labels = connected_components(ckng)
    print(n)
    plot_graph(data, ckng, color=color)

if __name__ == '__main__':
    main()

