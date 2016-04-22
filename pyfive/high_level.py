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

        # read in the symbol table
        entry = low_level._unpack_struct_from_file(
            low_level.SYMBOL_TABLE_ENTRY, self._fh)
        assert entry['cache_type'] == 1
        self.dataobjects = low_level.DataObjects(self._fh)

        msgs = self.dataobjects.find_msg_type(low_level.SYMBOL_TABLE_MSG_TYPE)
        assert len(msgs) == 1
        assert msgs[0]['type'] == low_level.SYMBOL_TABLE_MSG_TYPE
        assert msgs[0]['size'] == 16
        symbol_table_message = low_level._unpack_struct_from(
            low_level.SYMBOL_TABLE_MESSAGE, self.dataobjects.msg_data,
            msgs[0]['offset_to_message'])
        self.address_of_btree = symbol_table_message['btree_address']
        self.address_of_heap = symbol_table_message['heap_address']

        #self.root_group = low_level.RootGroup(self._fh)

        self._fh.seek(self.address_of_btree)
        self.btree = low_level.BTree(self._fh)

        self._fh.seek(self.address_of_heap)
        self.heap = low_level.Heap(self._fh)

        self.symboltables = []
        self._dataset_offsets = {}
        self._group_offsets = {}
        for symbol_table_addreess in self.btree.symbol_table_addresses():
            self._fh.seek(symbol_table_addreess)
            table = low_level.SymbolTable(self._fh)
            table.assign_name(self.heap)

            self.symboltables.append(table)
            self._dataset_offsets.update(table.find_datasets())
            self._group_offsets.update(table.find_groups())

        self.datasets = {k: Dataset(k, v, self._fh) for k, v in
                         self._dataset_offsets.items()}
        self.groups = {k: Group(k, v, self._fh) for k, v in
                       self._group_offsets.items()}


    def get_attributes(self):
        return self.dataobjects.get_attributes()

    def close(self):
        """ Close the file. """
        self._fh.close()


class Group(object):

    def __init__(self, name, offset, fh):

        # read the group data objects
        fh.seek(offset)
        dataobjects = low_level.DataObjects(fh)

        # extract the B-tree and local heap address from the symbol table
        # message
        msgs = dataobjects.find_msg_type(17)
        assert len(msgs) == 1
        assert msgs[0]['size'] == 16
        symbol_table_message = low_level._unpack_struct_from(
            low_level.SYMBOL_TABLE_MESSAGE, dataobjects.msg_data,
            msgs[0]['offset_to_message'])

        self.address_of_btree = symbol_table_message['btree_address']
        self.address_of_heap = symbol_table_message['heap_address']

        self.name = name
        self.offset = offset
        self._dataobjects = dataobjects
        self._fh = fh

        self._fh.seek(self.address_of_btree)
        self.btree = low_level.BTree(self._fh)

        self._fh.seek(self.address_of_heap)
        self.heap = low_level.Heap(self._fh)

        self.symboltables = []
        self._dataset_offsets = {}
        self._group_offsets = {}
        for symbol_table_addreess in self.btree.symbol_table_addresses():
            self._fh.seek(symbol_table_addreess)
            table = low_level.SymbolTable(self._fh)
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
        self._dataobjects = low_level.DataObjects(fh)

    def get_attributes(self):
        """ Return a dictionary of all attributes. """
        return self._dataobjects.get_attributes()

    def get_data(self):
        return self._dataobjects.get_data()
