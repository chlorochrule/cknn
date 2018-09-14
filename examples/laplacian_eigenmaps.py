# -*- coding: utf-8 -*-

import numpy as np
from sklearn.datasets import load_digits, make_swiss_roll, make_s_curve
from sklearn.manifold import SpectralEmbedding
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import offsetbox
import seaborn as sns
from cknn import cknneighbors_graph


def laplacian_eigenmaps(data, n_neighbors=5):
    model_cknn = SpectralEmbedding(n_components=2, affinity='precomputed')
    model_normal = SpectralEmbedding(n_components=2, n_neighbors=n_neighbors)
    
    neighbors = cknneighbors_graph(data, n_neighbors=n_neighbors, delta=0.2, 
                                   metric='euclidean', t='inf',
                                   include_self=True)
    y_cknn = model_cknn.fit_transform(neighbors.toarray())
    y_normal = model_normal.fit_transform(data)

    return y_cknn, y_normal


def plot3d(data, color):
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=color, 
                cmap=plt.cm.Spectral)
    plt.show()
    plt.clf()


def plot2d_spectral(data, color):
    plt.scatter(data[:, 0], data[:, 1], c=color, cmap=plt.cm.Spectral)
    plt.show()
    plt.clf()


def plot2d_label(X, title=None):
    digits = load_digits()
    y = digits.target
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(digits.target[i]),
                 color=plt.cm.Set1(y[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})

    if hasattr(offsetbox, 'AnnotationBbox'):
        # only print thumbnails with matplotlib > 1.0
        shown_images = np.array([[1., 1.]])  # just something big
        for i in range(digits.data.shape[0]):
            dist = np.sum((X[i] - shown_images) ** 2, 1)
            if np.min(dist) < 4e-3:
                # don't show points that are too close
                continue
            shown_images = np.r_[shown_images, [X[i]]]
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(digits.images[i], cmap=plt.cm.gray_r),
                X[i])
            ax.add_artist(imagebox)
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)
    plt.show()


def main():
    # data, color = make_s_curve(n_samples=1000, noise=0.0)
    # data, color = make_swiss_roll(n_samples=1000, noise=0.5)
    data, _ = load_digits(return_X_y=True)

    y_cknn, y_normal = laplacian_eigenmaps(data, n_neighbors=10)
    
    # plot3d(data, color)
    # plot2d_spectral(y_cknn, color)
    # plot2d_spectral(y_normal, color)
    plot2d_label(y_cknn)
    plot2d_label(y_normal)


if __name__ == '__main__':
    main()
