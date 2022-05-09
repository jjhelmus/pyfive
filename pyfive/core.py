""" Core low-level functions and classes used by multiple pyfive modules. """

from __future__ import division

from collections import OrderedDict
from math import ceil
import struct


UNDEFINED_ADDRESS = 0xffffffffffffffff


class InvalidHDF5File(Exception):
    """ Exception raised when an invalid HDF5 file is detected. """
    pass


class Reference(object):
    """
    HDF5 Reference.
    """

    def __init__(self, address_of_reference):
        self.address_of_reference = address_of_reference

    def __bool__(self):
        # False for null references (address of 0) True otherwise
        return bool(self.address_of_reference)

    __nonzero__ = __bool__  # Python 2.x requires __nonzero__ for truth value


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


def _unpack_integer(nbytes, buf, offset=0):
    """ Read an integer with an uncommon number of bytes. """
    fmt = "{}s".format(nbytes)
    values = struct.unpack_from(fmt, buf, offset=offset)
    return int.from_bytes(values[0], byteorder="little", signed=False)
