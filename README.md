# Continuous k-Nearest Neighbors in Python

**Note:** This package supports Python 3.6 or newer.

This is a Python implementation of Continuous k-Nearest Neighbors(CkNN)
proposed in the paper 'Consistent Manifold Representation for Topological Data 
Analysis' (https://arxiv.org/pdf/1606.02353.pdf)

[![license](http://img.shields.io/badge/license-MIT-blue.svg?style=flat)](https://github.com/chlorochrule/cknn/blob/master/LICENSE)
[![Build Status](https://travis-ci.org/chlorochrule/cknn.svg?branch=master)](https://travis-ci.org/chlorochrule/cknn)

## Installation

This package only depends on [numpy](http://www.numpy.org/) and 
[scipy](https://www.scipy.org/). The package can be installed via `pip`:

```
$ pip install git+https://github.com/chlorochrule/cknn
```

## Usage

`X` is a data matrix. A simple example is like:

```python
from cknn import cknneighbors_graph

ckng = cknneighbors_graph(X, n_neighbors=5, delta=1.0)
```

## License

MIT
