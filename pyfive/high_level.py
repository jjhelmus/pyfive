""" High-level classes for reading HDF5 files.  """

import struct
from collections import OrderedDict

import numpy as np

from . import low_level

# Plans
# Classes
# -------
# Data Objects - read and store Data object header and Messages
# Object header?


# https://www.hdfgroup.org/HDF5/doc/H5.format.html

class HDF5File:
    """
    Class for reading data from HDF5 files.

    """

    def __init__(self, filename):
        """ initalize. """

        self._fh = open(filename, 'rb')
        self.superblock = low_level.SuperBlock(self._fh)
        self.root_group = low_level.RootGroup(self._fh)

        self._fh.seek(self.root_group.address_of_btree)
        self.btree = low_level.BTree(self._fh)

        self._fh.seek(self.root_group.address_of_heap)
        self.heap = low_level.Heap(self._fh)

        self.symboltables = []
        self._dataset_offsets = {}
        for symbol_table_addreess in self.btree. symbol_table_addresses():
            self._fh.seek(symbol_table_addreess)
            table = low_level.SymbolTable(self._fh)
            table.assign_name(self.heap)

            self.symboltables.append(table)
            self._dataset_offsets.update(table.links_and_offsets())

        self.datasets = {k: Dataset(k, v, self._fh) for k, v in
                         self._dataset_offsets.items()}

    def close(self):
        """ Close the file. """
        self._fh.close()


class Dataset(object):
    """
    Class providing access to attribute and data stored in a HDF5 Dataset.

    Parameters
    ----------

    """

    def __init__(self, name, offset, fh):
        """ initalize with fh position at the Data Object Header. """
        self.name = name
        fh.seek(offset)
        dataobjects = low_level.DataObjects(fh)
        self._msg_data = dataobjects.message_data
        self._msgs = dataobjects.messages
        self.fh = fh

    def get_attributes(self):
        """ Return a dictionary of all attributes. """
        attrs = {}
        attr_msgs = [m for m in self._msgs if m['type'] == 12]
        for msg in attr_msgs:
            offset = msg['offset_to_message']
            name, value = low_level.unpack_attribute(self._msg_data, offset)
            attrs[name] = value
        return attrs

    def get_data(self):
        attr_msg = [m for m in self._msgs if m['type'] == 8][0]
        start = attr_msg['offset_to_message']
        size = attr_msg['size']
        version, layout_class, offset, size = struct.unpack_from(
            '<BBQQ', self._msg_data, start)
        print(offset, size)
        self.fh.seek(offset)
        buf = self.fh.read(size)
        dtype='<i4'
        shape = (100, )
        return np.frombuffer(buf, dtype=dtype).reshape(shape)
