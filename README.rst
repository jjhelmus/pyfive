pyfive : A pure Python HDF5 file reader
=======================================

|Travis|_

.. |Travis| image:: https://api.travis-ci.org/jjhelmus/pyfive.png?branch=master
.. _Travis: https://travis-ci.org/jjhelmus/pyfive

pyfive is an open source library for reading HDF5 files written using
pure Python (no C extensions). The package is still in development and not all
features of HDF5 files are supported. For a more mature Python library for
read and writing HDF5 files, try `h5py`_.

pyfive aims to support the same API as `h5py`_ for reading files. Cases where a
file uses a feature that is supported by `h5py`_ but not pyfive are considered
bug and should be reported on the `issue tracker`_. Writing HDF5 is not a goal
of pyfive and portions of the API which apply only to writing will not be
implemented.

.. _h5py: http://www.h5py.org/
.. _issue tracker: https://github.com/jjhelmus/pyfive/issues

Dependencies
============

pyfive is tested to work with Python 3.8 to 3.13.  It may also work
with other Python versions.

The only dependencies to run the software besides Python is NumPy.

Install
=======

pyfive can be installed using pip using the command::

    pip install pyfive

conda package are also available from `conda-forge`_ which can be installed::

    conda install -c conda-forge pyfive

To install from source in your home directory use::

    python setup.py install --user

The library can also be imported directly from the source directory.

.. _conda-forge: https://conda-forge.github.io/

Development
===========

git
---

You can check out the latest pyfive souces with the command::

    git clone https://github.com/jjhelmus/pyfive.git

testing
-------

pyfive comes with a test suite in the ``tests`` directory.  These tests can be
exercised using the commands ``pytest`` from the root directory assuming the
``pytest`` package is installed.

Related Projects
================

`jsfive`_ is a pure javascript HDF5 file reader based on pyfive.

.. _jsfive: https://github.com/usnistgov/jsfive
