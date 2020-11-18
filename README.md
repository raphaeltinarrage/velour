# velour

Python package for topological inference from point clouds with persistent homology.
Based on the [`gudhi`](https://gudhi.inria.fr/python/latest/)  library.

## Methods

The package `velour` gathers implementations of our methods for topological inference. It allows the use of:
- **DTM-filtrations:** a family of filtrations for persistent homology, that can be applied even when the input point cloud contains anomalous points. Notebook demo [here](https://github.com/raphaeltinarrage/DTM-Filtrations/blob/master/Demo.ipynb) and mathematical explanation [here](https://arxiv.org/abs/1811.04757).
- **Lifted sets and lifted filtrations:** allows to estimate the homology of an abstract manifold from a finite sample of an immersion of it. Notebook demo [here](https://github.com/raphaeltinarrage/ImmersedManifolds/blob/master/Demo.ipynb) and mathematical explanation [here](https://arxiv.org/abs/1912.03033).
- **Persistent Stiefel-Whitney classes:** allows to estimate the first Stiefel-Whitney class of a vector bundle from a finite sample of it. Notebook demo [here](https://github.com/raphaeltinarrage/PersistentCharacteristicClasses/blob/master/Demo.ipynb) and mathematical explanation [here](https://arxiv.org/abs/2005.12543).

## Structure

The package is divided into three modules:
- `persistent` gathers tools for handling filtrations of simplicial complexes (simplex trees).
- `geometry` contains the implementation of various geometric quantities used by `persistent`.
- `datasets` consists in various utilities for sampling datasets (from $\mathbb{R}^2$ to $\mathbb{R}^{12}$) and plotting them.

## Setup

It can be installed from PyPI via
```
pip install velour
```
Current release: 2020.11.18

## Documentation

Not yet! But feel free to contact me anytime.

Raphaël Tinarrage - https://raphaeltinarrage.github.io/
