[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity)
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18599472.svg)](https://doi.org/10.5281/zenodo.18599472)
[![JOSS paper](https://joss.theoj.org/papers/10.21105/joss.09688/status.svg)](https://joss.theoj.org/papers/10.21105/joss.09688)
[![Documentation Status](https://app.readthedocs.org/projects/pyfive/badge/?version=latest)](https://pyfive.readthedocs.io/en/latest/?badge=latest)
[![Test](https://github.com/NCAS-CMS/pyfive/actions/workflows/pytest.yml/badge.svg)](https://github.com/NCAS-CMS/pyfive/actions/workflows/pytest.yml)
[![codecov](https://codecov.io/gh/NCAS-CMS/pyfive/graph/badge.svg?token=3In5JuzeGK)](https://codecov.io/gh/NCAS-CMS/pyfive)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/pyfive/badges/version.svg)](https://anaconda.org/conda-forge/pyfive)

![pyfive-logo](https://raw.githubusercontent.com/NCAS-CMS/pyfive/main/doc/figures/Pyfive-logo.png)

[Latest doc builds on RTD](https://app.readthedocs.org/projects/pyfive/builds/)

pyfive : A pure Python HDF5 file reader
=======================================

``pyfive`` is an open source library for reading HDF5 files written in Python,
using NumPy and numcodecs without the HDF5 C library. The package is still in development and not all
features of HDF5 files are supported.

``pyfive`` aims to support the same API as [`h5py`](https://github.com/h5py/h5py) for reading files.
Cases where a file uses a feature that is supported by ``h5py`` but not ``pyfive`` are considered bugs
and should be reported in our [Issues](https://github.com/NCAS-CMS/pyfive/issues).
Writing HDF5 output is not a goal of ``pyfive`` and portions of the API which apply only to writing will not be implemented.

Dependencies
============

``pyfive`` is tested against Python versions 3.11 to 3.14.
It may also work with other Python versions.

Runtime dependencies are ``numpy``, ``numcodecs``, ``fsspec`` and
``typing-extensions``. Fletcher32 checksum verification uses numcodecs' native
implementation; pyfive itself does not require the HDF5 C library.

Install
=======

pyfive can be installed using ``pip`` using the command::

    pip install pyfive

``conda`` packages are also available from conda-forge::

    conda install -c conda-forge pyfive

To install from source in your home directory use::

    pip install --user ./pyfive

The library can also be imported directly from the source directory.


Development
===========

git
---

You can check out the latest ``pyfive`` souces with the command::

    git clone https://github.com/NCAS-CMS/pyfive.git

testing
-------

``pyfive`` comes with a test suite in the ``tests`` directory.
These tests can be exercised using the ``pytest`` command from the root directory (requires installation of the ``pytest`` package).

Conda-forge feedstock
=====================

Package repository [conda-forge feedstock](https://github.com/conda-forge/pyfive-feedstock)

Codecov
=======

Test coverage assessement is done using [codecov](https://app.codecov.io/gh/NCAS-CMS/pyfive/)

Documentation
=============

Build locally with Sphinx:

    $ sphinx-build -Ea doc doc/build
