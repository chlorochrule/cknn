# -*- coding: utf-8 -*-

import unittest
from numpy.testing import assert_almost_equal

import numpy as np
from scipy.spatial.distance import pdist, squareform
import math

from cknn import cknneighbors_graph

def calc_cknn(X, d, d_k, delta=1.0, t='inf'):
    cknn = np.empty((4, 4))
    for i, d_k_i in enumerate(d_k):
        for j, d_k_j in enumerate(d_k):
            if i == j:
                cknn[i, j] = 0.0
            elif d[i, j] < delta*(d_k[i]*d_k[j]) ** 0.5:
                cknn[i, j] = 1.0 if t=='inf' else math.exp(-d[i, j]**2/t)
                cknn[j, i] = 1.0 if t=='inf' else math.exp(-d[i, j]**2/t)
            else:
                cknn[i, j] = 0.0
                cknn[j, i] = 0.0
    return cknn


class TestCknn(unittest.TestCase):
    """Test class of cknn module"""

    def setUp(self):
        self.X = np.array([[0, 0], [1, 0], [2, 1], [0, 0.5]])

    def test_result_when_default_params(self):
        d = squareform(pdist(self.X))
        d_k = np.array([d[0, 1], d[1, 3], d[2, 3], d[3, 1]])
        expected = calc_cknn(self.X, d, d_k)
        result = cknneighbors_graph(self.X, 2).toarray()
        assert_almost_equal(expected, result)

    def test_result_when_customized_params(self):
        t=2
        delta=1.2
        d = squareform(pdist(self.X))
        d_k = np.array([d[0, 3], d[1, 0], d[2, 1], d[3, 0]])
        expected = calc_cknn(self.X, d, d_k, delta=delta, t=t)
        for i in range(4):
            expected[i, i] = 1.0
        result = cknneighbors_graph(self.X, 1, delta=delta, metric='euclidean', 
                                    t=t, include_self=True, 
                                    is_sparse=False, return_instance=True)
        result = result.ckng
        assert_almost_equal(expected, result)


if __name__ == '__main__':
    unittest.main()
