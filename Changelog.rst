Unreleased
----------

* New ``p5check`` command (``pyfive.p5check.check_layout``) which checks local or remote (``https://``, ``s3://``) files for fragmented chunk-index metadata and chunk sizes of one, without reading data.

Version 1.1.1
-------------

**2026-02-23**

This is a bugfix release:

* (main bugfix) extract filename from pyfive S3 Dataset correctly by `Valeriu Predoi <https://github.com/valeriupredoi>`_ in https://github.com/NCAS-CMS/pyfive/pull/197
* add DOI badge in README by `Valeriu Predoi <https://github.com/valeriupredoi>`_ in https://github.com/NCAS-CMS/pyfive/pull/194
* add JOSS paper badge in README by `Valeriu Predoi <https://github.com/valeriupredoi>`_ in https://github.com/NCAS-CMS/pyfive/pull/196

Version 1.1.0
-------------

**2026-02-10**

**Highlights**

* Add support for pre-commit with mypy and Ruff and start adding type hints by `Valeriu Predoi <https://github.com/valeriupredoi>`_ in https://github.com/NCAS-CMS/pyfive/pull/173
* Timing + Optimisations + Minor fixes by `Bryan Lawrence <https://github.com/bnlawrence>`_ in https://github.com/NCAS-CMS/pyfive/pull/191
* Add ``__orthogonal_indexing__`` flag by `David Hassell <https://github.com/davidhassell>`_ in https://github.com/NCAS-CMS/pyfive/issue/171
* Allow negative slices when indexing chunked data by `David Hassell <https://github.com/davidhassell>`_ in https://github.com/NCAS-CMS/pyfive/pull/170

**Other Changes**

* [JOSS REVIEW PR] Harmonize depenedncies from environment.yml and pyproject.toml by `Valeriu Predoi <https://github.com/valeriupredoi>`_ in https://github.com/NCAS-CMS/pyfive/pull/174
* pep639 license by @valeriupredoi in https://github.com/NCAS-CMS/pyfive/pull/180
* add CONTRIBUTING.md file (and contributing guidelines, of course) by `Valeriu Predoi <https://github.com/valeriupredoi>`_ in https://github.com/NCAS-CMS/pyfive/pull/181
* Documentation consistency fixes by `Trevor James Smith <https://github.com/Zeitsperre>`_ in https://github.com/NCAS-CMS/pyfive/pull/175
* Overhaul `.gitignore` file by `Valeriu Predoi <https://github.com/valeriupredoi>`_ in https://github.com/NCAS-CMS/pyfive/pull/179
* add link to Pyfive GitHub repository on doc landing page by `Valeriu Predoi <https://github.com/valeriupredoi>`_ in https://github.com/NCAS-CMS/pyfive/pull/185
* Skip tests if ceda-jasmin s3 storage is offline by `Valeriu Predoi <https://github.com/valeriupredoi>`_ in https://github.com/NCAS-CMS/pyfive/pull/187
* [pre-commit.ci] pre-commit autoupdate by ``@pre-commit-ci[bot]`` in https://github.com/NCAS-CMS/pyfive/pull/188
* Replace python-lzf with python-neo-lzf by `Valeriu Predoi <https://github.com/valeriupredoi>`_ in https://github.com/NCAS-CMS/pyfive/pull/189

Version 1.0.1
-------------

**2025-12-18**

* Set ``__version__`` attribute dynamically in ``__init__.py`` by `Valeriu Predoi <https://github.com/valeriupredoi>`_ in https://github.com/NCAS-CMS/pyfive/pull/152
* pin ``numpy>=2`` by `Valeriu Predoi <https://github.com/valeriupredoi>`_ in https://github.com/NCAS-CMS/pyfive/pull/157
* add a ``pip check`` by `Valeriu Predoi <https://github.com/valeriupredoi>`_ in https://github.com/NCAS-CMS/pyfive/pull/159
* Highlight: Test case for corner case file (buffer too small) and add bookkeeping for Fractal Heaps by `Kai Mühlbauer <https://github.com/kmuehlbauer>`_ in https://github.com/NCAS-CMS/pyfive/pull/160
* update ``setuptools`` pins by `Valeriu Predoi <https://github.com/valeriupredoi>`_ in https://github.com/NCAS-CMS/pyfive/pull/165
* Fix display of ``p5dump`` global attributes by `Ezequiel Cimadevilla <https://github.com/zequihg50>`_ in https://github.com/NCAS-CMS/pyfive/pull/163

Version 1.0.0
-------------

**2025-11-13**

* fix fletcher32, add tests by `Kai Mühlbauer <https://github.com/kmuehlbauer>`_ in https://github.com/NCAS-CMS/pyfive/pull/133
* add lzf decompress filter by `Kai Mühlbauer <https://github.com/kmuehlbauer>`_ in https://github.com/NCAS-CMS/pyfive/pull/136
* introduce new H5 types to replace current type-tuples by `Kai Mühlbauer <https://github.com/kmuehlbauer>`_ in https://github.com/NCAS-CMS/pyfive/pull/122
* mark ``test_hdf5_filters`` as flaky by `Valeriu Predoi <https://github.com/valeriupredoi>`_ in https://github.com/NCAS-CMS/pyfive/pull/141
* add pytest reruns plugin ``pytest-rerunfailures`` and minimal settings by `Valeriu Predoi <https://github.com/valeriupredoi>`_ in https://github.com/NCAS-CMS/pyfive/pull/142
* Optimise when we get access to b-tree by providing lazier view of datasets, access to b-tree location, and new p5dump by `Bryan Lawrence <https://github.com/bnlawrence>`_ in https://github.com/NCAS-CMS/pyfive/pull/138
* Added btree v2 test but skipping it (#137) by `Ezequiel Cimadevilla <https://github.com/zequihg50>`_ in https://github.com/NCAS-CMS/pyfive/pull/143
* ``p5dump`` examples by `David Hassell <https://github.com/davidhassell>`_ in https://github.com/NCAS-CMS/pyfive/pull/147
* Milestone for v1.0.0 release by `Bryan Lawrence <https://github.com/bnlawrence>`_ in https://github.com/NCAS-CMS/pyfive/pull/148
* Added consolidated metadata functionality by `Ezequiel Cimadevilla <https://github.com/zequihg50>`_ in https://github.com/NCAS-CMS/pyfive/pull/145

Version 0.9.0
-------------

**2025-10-17**

* use pytest temporary dir factory to write some of the test hdf5 files and move all fixed hdf5 sample data files to `tests/data` by `Valeriu Predoi <https://github.com/valeriupredoi>`_ in https://github.com/NCAS-CMS/pyfive/pull/117
* Install netcdf4 from conda-forge and pin netcdf4<1.7.3 in pyproject.toml by `Valeriu Predoi <https://github.com/valeriupredoi>`_ in https://github.com/NCAS-CMS/pyfive/pull/124
* Support Python 3.14 by `Valeriu Predoi <https://github.com/valeriupredoi>`_ in https://github.com/NCAS-CMS/pyfive/pull/125
* remove pin on netcdf4 by `Valeriu Predoi <https://github.com/valeriupredoi>`_ in https://github.com/NCAS-CMS/pyfive/pull/126
* all changes above with review from `Kai Mühlbauer <https://github.com/kmuehlbauer>`_ and `David Hassell <https://github.com/davidhassell>`_

Version 0.8.0
-------------

**2025-10-07**

* Support for Opaque datasets by `Bryan Lawrence <https://github.com/bnlawrence>`_ in https://github.com/NCAS-CMS/pyfive/pull/114 with review from `Kai Mühlbauer <https://github.com/kmuehlbauer>`_

Version 0.7.0
-------------

**2025-10-06**


* add joss paper pdf conversion via gha by `Valeriu Predoi <https://github.com/valeriupredoi>`_ in https://github.com/NCAS-CMS/pyfive/pull/97
* fix changelog to include Kai as contributor to v0.6.0 and change Brian L -> Bryan L (typos) by `Valeriu Predoi
  <https://github.com/valeriupredoi>`_ in https://github.com/NCAS-CMS/pyfive/pull/96
* New logo by `Valeriu Predoi <https://github.com/valeriupredoi>`_ in https://github.com/NCAS-CMS/pyfive/pull/98
* fix Enum and Empty attributes by `Kai Mühlbauer <https://github.com/kmuehlbauer>`_ in https://github.com/NCAS-CMS/pyfive/pull/102
* Fix user datatypes (enum, compound) by `Kai Mühlbauer <https://github.com/kmuehlbauer>`_ in https://github.com/NCAS-CMS/pyfive/pull/105
* Added partial support for compact datasets. by `Ezequiel Cimadevilla <https://github.com/zequihg50>`_ in https://github.com/NCAS-CMS/pyfive/pull/107
* fix handling of uninitialized vlen strings by `Kai Mühlbauer <https://github.com/kmuehlbauer>`_ in https://github.com/NCAS-CMS/pyfive/pull/110
* add dataobjects.dtype to DatasetMeta by `Kai Mühlbauer <https://github.com/kmuehlbauer>`_ in https://github.com/NCAS-CMS/pyfive/pull/112

Version 0.6.0
-------------

**2025-09-16**

* Enumeration Support (https://github.com/NCAS-CMS/pyfive/issues/85 by
  `Bryan Lawrence <https://github.com/bnlawrence>`_,
  `Kai Mühlbauer <https://github.com/kmuehlbauer>`_,
  `Brian Maranville <https://github.com/bmaranville>`_))

Version 0.5.1
-------------

**2025-08-21**

* Add a Changelog (https://github.com/NCAS-CMS/pyfive/issues/87 by
  `David Hassell <https://github.com/davidhassell>`_)
* Improved documentation (https://github.com/NCAS-CMS/pyfive/pull/84
  by `Kai Mühlbauer <https://github.com/kmuehlbauer>`_)
* When getting object by address, do not fully instantiate when
  iterating (https://github.com/NCAS-CMS/pyfive/pull/83 by `Kai
  Mühlbauer <https://github.com/kmuehlbauer>`_)
* Add documentation for Pyfive
  (https://github.com/NCAS-CMS/pyfive/pull/81 by `Bryan Lawrence
  <https://github.com/bnlawrence>`_)
* Setup documentation builds on Readthedocs
  (https://github.com/NCAS-CMS/pyfive/pull/80 by `Valeriu Predoi
  <https://github.com/valeriupredoi>`_)

----

Version 0.5.0
-------------

**2025-07-11**

* Add docs basic structure and ancillary files
  (https://github.com/NCAS-CMS/pyfive/pull/79 by `Valeriu Predoi
  <https://github.com/valeriupredoi>`_)
* Test codecov CI
  (https://github.com/NCAS-CMS/pyfive/pull/77 by `Valeriu Predoi
  <https://github.com/valeriupredoi>`_)
* Setup test coverage reporting via codecov reporting
  (https://github.com/NCAS-CMS/pyfive/pull/76 by `Valeriu Predoi
  <https://github.com/valeriupredoi>`_)
* Changes to README to reflect new location and added a PR template
  (https://github.com/NCAS-CMS/pyfive/pull/74 by `Valeriu Predoi
  <https://github.com/valeriupredoi>`_)
* PyPI package build GHA package builder
  (https://github.com/NCAS-CMS/pyfive/pull/73 by `Valeriu Predoi
  <https://github.com/valeriupredoi>`_)
* Switch from master to main in Github Action test and add GHA PyPI
  package builder (https://github.com/NCAS-CMS/pyfive/pull/72 by
  `Valeriu Predoi <https://github.com/valeriupredoi>`_)
* Modernize packaging (https://github.com/NCAS-CMS/pyfive/pull/69 by
  `Valeriu Predoi <https://github.com/valeriupredoi>`_)
* Functionality enhancements to address lazy loading of chunked data,
  variable length strings, and other minor bug fixes
  (https://github.com/NCAS-CMS/pyfive/pull/68 by `Bryan Lawrence
  <https://github.com/bnlawrence>`_)

----

Version 0.4.0
-------------

**2024-10-29**

* Update README and setup for Python 3.8-3.13
  (https://github.com/NCAS-CMS/pyfive/pull/63 by `Jonathan Helmus
  <https://github.com/jjhelmus>`_)
* Remove use of deprecated product function from numpy
  (https://github.com/NCAS-CMS/pyfive/pull/62 by `Jonathan Helmus
  <https://github.com/jjhelmus>`_)
* Add Python 3.11 and 3.12 to test matrix in CI
  (https://github.com/NCAS-CMS/pyfive/pull/57 by `Jonathan Helmus
  <https://github.com/jjhelmus>`_)
* Retrieve maxshape for Dataset
  (https://github.com/NCAS-CMS/pyfive/issues/50 by `Brian Maranville
  <https://github.com/bmaranville>`_)
* Minimal change to avoid a numpy deprecation failure
  (https://github.com/NCAS-CMS/pyfive/pull/55 by `Bryan Lawrence
  <https://github.com/bnlawrence>`_)
* Use name as key for links if creation order is not specified
  (https://github.com/NCAS-CMS/pyfive/pull/54 by `Brian Maranville
  <https://github.com/bmaranville>`_)
* Filter pipeline description v2
  (https://github.com/NCAS-CMS/pyfive/pull/52 by `Brian Maranville
  <https://github.com/bmaranville>`_)
* B-tree v2 for links from LINK_INFO messages
  (https://github.com/NCAS-CMS/pyfive/issues/46 by `Wout De Nolf
  <https://github.com/woutdenolf>`_)
* Fix test errors and warnings, drop support for Python 2.7
  (https://github.com/NCAS-CMS/pyfive/pull/44 by `Jonathan Helmus
  <https://github.com/jjhelmus>`_)
* Adding an implementation of SoftLink
  (https://github.com/NCAS-CMS/pyfive/pull/43 by `Brian Maranville
  <https://github.com/bmaranville>`_)
* Property offset is the same for layout_class 1 and 2
  (https://github.com/NCAS-CMS/pyfive/pull/42 by `Brian Maranville
  <https://github.com/bmaranville>`_)
* Fix reading of superblock version 3
  (https://github.com/NCAS-CMS/pyfive/pull/40 by `Jonathan Helmus
  <https://github.com/jjhelmus>`_)
* Support v3 superblocks, minor maintenance
  (https://github.com/NCAS-CMS/pyfive/pull/39 by `Jonathan Helmus
  <https://github.com/jjhelmus>`_)
* Dataset.read_direct method
  (https://github.com/NCAS-CMS/pyfive/pull/37 by `Jonathan Helmus
  <https://github.com/jjhelmus>`_)
* Dataset.astype and Dataset.value methods
  (https://github.com/NCAS-CMS/pyfive/pull/36 by `Jonathan Helmus
  <https://github.com/jjhelmus>`_)
* Access datasets and groups by path
  (https://github.com/NCAS-CMS/pyfive/issues/15 by `Jonathan Helmus
  <https://github.com/jjhelmus>`_)

----

Version 0.3.0
-------------

**2017-09-29**

* Split low_level module into multiple smaller modules and refactor
  (https://github.com/NCAS-CMS/pyfive/pull/34 by `Jonathan Helmus
  <https://github.com/jjhelmus>`_)
* Unit tests for resizable datasets
  (https://github.com/NCAS-CMS/pyfive/pull/33 by `Jonathan Helmus
  <https://github.com/jjhelmus>`_)
* Add File, Dataset, and Group __repr__ methods
  (https://github.com/NCAS-CMS/pyfive/pull/32 by `Jonathan Helmus
  <https://github.com/jjhelmus>`_)
* Add visit and visititems methods to Group class
  (https://github.com/NCAS-CMS/pyfive/pull/31 by `Jonathan Helmus
  <https://github.com/jjhelmus>`_)
* Support for reading dataset fillvalues
  (https://github.com/NCAS-CMS/pyfive/pull/29 by `Jonathan Helmus
  <https://github.com/jjhelmus>`_)
* Support for reading and verifting fletcher32 checksums
  (https://github.com/NCAS-CMS/pyfive/pull/28 by `Jonathan Helmus
  <https://github.com/jjhelmus>`_)
* Refactor datatype message funcs into class
  (https://github.com/NCAS-CMS/pyfive/pull/27 by `Jonathan Helmus
  <https://github.com/jjhelmus>`_)
* Support for complex attribute datatypes
  (https://github.com/NCAS-CMS/pyfive/pull/26 by `Jonathan Helmus
  <https://github.com/jjhelmus>`_)
* Refactor attribute value retrieval
  (https://github.com/NCAS-CMS/pyfive/pull/25 by `Jonathan Helmus
  <https://github.com/jjhelmus>`_)
* Dataset.ndim attribute (https://github.com/NCAS-CMS/pyfive/pull/24
  by `Jonathan Helmus <https://github.com/jjhelmus>`_)
* Filename attribute set for file-like objects
  (https://github.com/NCAS-CMS/pyfive/pull/23 by `Jonathan Helmus
  <https://github.com/jjhelmus>`_)
* File can be used as a context manager
  (https://github.com/NCAS-CMS/pyfive/pull/22 by `Jonathan Helmus
  <https://github.com/jjhelmus>`_)
* Support for dimension labels and scales
  (https://github.com/NCAS-CMS/pyfive/issues/14 by `Jonathan Helmus
  <https://github.com/jjhelmus>`_)
* Read variable length sequence attributes
  (https://github.com/NCAS-CMS/pyfive/pull/20 by `Jonathan Helmus
  <https://github.com/jjhelmus>`_)
* Add Python 3.6 to travis matrix
  (https://github.com/NCAS-CMS/pyfive/pull/19 by `Jonathan Helmus
  <https://github.com/jjhelmus>`_)
* Support for reading arrayed attributes
  (https://github.com/NCAS-CMS/pyfive/pull/18 by `Jonathan Helmus
  <https://github.com/jjhelmus>`_)
* Support for Reference attribute types
  (https://github.com/NCAS-CMS/pyfive/pull/17 by `Jonathan Helmus
  <https://github.com/jjhelmus>`_)
* Add support for v1 and v2 data objects
  (https://github.com/NCAS-CMS/pyfive/pull/16 by `synaptic
  <https://github.com/synaptic>`_)
* Allow reading from BytesIO objects
  (https://github.com/NCAS-CMS/pyfive/pull/13 by `Jonathan Helmus
  <https://github.com/jjhelmus>`_)

----

Version 0.2.0
-------------

**2016-09-10**

* Add chunks attribute to pyfive.Dataset class
  https://github.com/NCAS-CMS/pyfive/pull/11 by `Jonathan Helmus
  <https://github.com/jjhelmus>`_)
* Reading of file-like objects with the tell method
  (https://github.com/NCAS-CMS/pyfive/issues/5 by `Jonathan Helmus
  <https://github.com/jjhelmus>`_)
* Add support for gzip compression and shuffle filter
  (https://github.com/NCAS-CMS/pyfive/issues/4 by `Jonathan Helmus
  <https://github.com/jjhelmus>`_)

----

Version 0.1.0
-------------

**2016-07-26**

* First release by `Jonathan Helmus <https://github.com/jjhelmus>`_
