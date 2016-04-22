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
        self.superblock = self._get_superblock(self._fh)
        self.root_group = self._get_root_group(self._fh)

        self._fh.seek(self.root_group['address_of_btree'])
        self.btree = low_level.BTree(self._fh)

        self._fh.seek(self.root_group['address_of_heap'])
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

        self._fh.seek(800)
        self.dataobjects = low_level.DataObjects(self._fh)

    def _get_superblock(self, fh):
        """ read in the superblock from an open file object. """
        superblock = low_level._unpack_struct_from_file(
            low_level.SUPERBLOCK, fh)

        # check
        assert superblock['format_signature'] == low_level.FORMAT_SIGNATURE
        assert superblock['superblock_version'] == 0
        assert superblock['offset_size'] == 8
        assert superblock['length_size'] == 8
        assert superblock['free_space_address'] == low_level.UNDEFINED_ADDRESS
        assert superblock['driver_information_address'] == low_level.UNDEFINED_ADDRESS
        return superblock

    def _get_root_group(self, fh):
        """ Read in the root group symbol table entry from an open file. """

        # read in the symbol table entry
        entry = low_level._unpack_struct_from_file(
            low_level.SYMBOL_TABLE_ENTRY, fh)
        assert entry['cache_type'] == 1

        # unpack the scratch area B-Tree and Heap addressed
        scratch = entry['scratch']
        address_of_btree, address_of_heap = struct.unpack('<QQ', scratch)
        entry['address_of_btree'] = address_of_btree
        entry['address_of_heap'] = address_of_heap

        # verify that the DataObject messages agree with the cache
        root_data_objects = low_level.DataObjects(fh)
        assert root_data_objects.messages[0]['type'] == 17
        assert root_data_objects.messages[0]['size'] == 16

        symbol_table_message = low_level._unpack_struct_from(
            low_level.SYMBOL_TABLE_MESSAGE, root_data_objects.message_data,
            root_data_objects.messages[0]['offset_to_message'])
        assert symbol_table_message['btree_address'] == address_of_btree
        assert symbol_table_message['heap_address'] == address_of_heap
        entry['symbol_table_message'] = symbol_table_message

        return entry

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
