"""
pyfive : a pure python HDF5 file reader.
This is the public API exposed by pyfive,
which is a small subset of the H5PY API.
"""

from pyfive.high_level import File, Group, Dataset
from pyfive.h5t import check_enum_dtype, check_string_dtype, check_dtype
from pyfive.h5py import Datatype, Empty
from importlib.metadata import version

__version__ = '0.5.0.dev'

