# -*- coding: utf-8 -*-

import numpy as np
from sklearn.datasets import load_digits
from sklearn.manifold import SpectralEmbedding
import matplotlib.pyplot as plt
from matplotlib import offsetbox
import seaborn as sns

from cknn import cknneighbors_graph

sns.set()


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
        shown_images = np.array([[1., 1.]])
        for i in range(digits.data.shape[0]):
            dist = np.sum((X[i] - shown_images) ** 2, 1)
            if np.min(dist) < 4e-3:
                continue
            shown_images = np.r_[shown_images, [X[i]]]
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(digits.images[i], cmap=plt.cm.gray_r),
                X[i])
            ax.add_artist(imagebox)
    plt.xticks([])
    plt.yticks([])
    if title is not None:
        plt.title(title)
    plt.show()


def main():
    data, _ = load_digits(return_X_y=True)
    n_neighbors = 10

    model_normal = SpectralEmbedding(n_components=2, n_neighbors=n_neighbors)
    y_normal = model_normal.fit_transform(data)
    plot2d_label(y_normal)

    ckng = cknneighbors_graph(data, n_neighbors=n_neighbors, delta=1.5)
    model_cknn = SpectralEmbedding(n_components=2, affinity='precomputed')
    y_cknn = model_cknn.fit_transform(ckng.toarray())
    plot2d_label(y_cknn)


if __name__ == '__main__':
    main()
