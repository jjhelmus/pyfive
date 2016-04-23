""" High-level classes for reading HDF5 files.  """

import struct
from collections import OrderedDict

import numpy as np

from . import low_level

from .low_level import SuperBlock, BTree, Heap, SymbolTable, DataObjects


# Plans
# Classes
# -------
# Data Objects - read and store Data object header and Messages
# Object header?


# https://www.hdfgroup.org/HDF5/doc/H5.format.html


class Group(object):
    """

    Attributes
    ----------
    * Public
    name
    datasets
    groups

    * Private?
    btree
    heap
    symboltables
    _dataobjects
    _fh

    * Remove?
    offset

    """

    def __init__(self, name, offset, fh):

        # read the group data objects
        fh.seek(offset)
        dataobjects = DataObjects(fh)

        # extract the B-tree and local heap address from the Symbol table
        # message
        btree_address, heap_address = dataobjects.get_btree_heap_addresses()

        self.name = name
        self.offset = offset
        self._dataobjects = dataobjects
        self._fh = fh

        self._fh.seek(btree_address)
        self.btree = BTree(self._fh)

        self._fh.seek(heap_address)
        self.heap = Heap(self._fh)

        self.symboltables = []
        self._dataset_offsets = {}
        self._group_offsets = {}
        for symbol_table_addreess in self.btree.symbol_table_addresses():
            self._fh.seek(symbol_table_addreess)
            table = SymbolTable(self._fh)
            table.assign_name(self.heap)

            self.symboltables.append(table)
            self._dataset_offsets.update(table.find_datasets())
            self._group_offsets.update(table.find_groups())

        self.datasets = {k: Dataset(k, v, self._fh) for k, v in
                         self._dataset_offsets.items()}
        self.groups = {k: Group(k, v, self._fh) for k, v in
                       self._group_offsets.items()}


    def get_attributes(self):
        """ Return a dictionary of all attributes. """
        return self._dataobjects.get_attributes()


class HDF5File(Group):
    """
    Class for reading data from HDF5 files.

    """

    def __init__(self, filename):
        """ initalize. """

        fh = open(filename, 'rb')
        self.superblock = SuperBlock(fh)

        # read in the symbol table
        entry = low_level._unpack_struct_from_file(
            low_level.SYMBOL_TABLE_ENTRY, fh)
        assert entry['cache_type'] == 1
        self._entry = entry
        offset = entry['object_header_address']
        name = None
        super(HDF5File, self).__init__(name, offset, fh)

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
        self._dataobjects = DataObjects(fh)

    def get_attributes(self):
        """ Return a dictionary of all attributes. """
        return self._dataobjects.get_attributes()

    def get_data(self):
        return self._dataobjects.get_data()
