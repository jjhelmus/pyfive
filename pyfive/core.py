""" Core low-level functions and classes used by multiple pyfive modules. """

from collections import OrderedDict
from math import ceil
import struct


class InvalidHDF5File(Exception):
    """ Exception raised when an invalid HDF5 file is detected. """
    pass


def _padded_size(size, padding_multipe=8):
    """ Return the size of a field padded to be a multiple a give value. """
    return int(ceil(size / padding_multipe) * padding_multipe)


def _structure_size(structure):
    """ Return the size of a structure in bytes. """
    fmt = '<' + ''.join(structure.values())
    return struct.calcsize(fmt)


def _unpack_struct_from_file(structure, fh):
    """ Unpack a structure into an OrderedDict from an open file. """
    size = _structure_size(structure)
    buf = fh.read(size)
    return _unpack_struct_from(structure, buf)


def _unpack_struct_from(structure, buf, offset=0):
    """ Unpack a structure into an OrderedDict from a buffer of bytes. """
    fmt = '<' + ''.join(structure.values())
    values = struct.unpack_from(fmt, buf, offset=offset)
    return OrderedDict(zip(structure.keys(), values))
