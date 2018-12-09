# -*- coding: utf-8 -*-
"""
The :mod:`cknn` module implements the Continuous k-Nearest Neighbors[1].

Reference
---------
.. [1] T. Berry and T. Sauer, “Consistent man-ifold representation for
       topological dataanalysis,” 2016.
"""

from .cknn import cknneighbors_graph, connect_rng

__all__ = ['cknneighbors_graph', 'connect_rng']
