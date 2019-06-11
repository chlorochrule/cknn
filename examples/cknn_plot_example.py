# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_swiss_roll
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph

from cknn import cknneighbors_graph

sns.set()


def connect_points(ax, data, graph):
    source, target = graph.nonzero()
    source, target = source[source < target], target[source < target]
    for s, t in zip(source, target):
        ax.plot(*data[[s,t], :].T, color='g')

def plot_graph(data, graph, title=None):
    fig = plt.figure()
    ax = Axes3D(fig)
    if title is not None:
        ax.set_title(title)
    ax.scatter(*data.T)
    connect_points(ax, data, graph)
    plt.show()
    plt.close()

def main():
    data, _ = make_swiss_roll(random_state=1)

    n_knn = 3
    kng = kneighbors_graph(data, n_neighbors=n_knn)
    title = 'KNN Graph where n_neighbors={}'.format(n_knn)
    plot_graph(data, kng, title=title)

    n_knn = 4
    kng = kneighbors_graph(data, n_neighbors=n_knn)
    title = 'KNN Graph where n_neighbors={}'.format(n_knn)
    plot_graph(data, kng)
    
    radius = 6.5
    rng = radius_neighbors_graph(data, radius=radius)
    title = 'RN Graph where radius={}'.format(radius)
    plot_graph(data, rng, title)

    n_neighbors = 5
    delta = 0.95
    ckng = cknneighbors_graph(data, n_neighbors=5, delta=0.95)
    title = 'CKNN Graph where n_neighbors={}, delta={}'\
        .format(n_neighbors, delta)
    plot_graph(data, ckng, title)
    

if __name__ == '__main__':
    main()
