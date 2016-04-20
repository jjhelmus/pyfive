"""
ppyhdf5 : Pure Python HDF5 reader

"""

import struct
from collections import OrderedDict

import numpy as np


# https://www.hdfgroup.org/HDF5/doc/H5.format.html

class HDF5File:
    """
    Class for reading data from HDF5 files.

    """

    def __init__(self, filename):
        """ initalize. """
        self._fh = open(filename, 'rb')

        # read in the superblock
        self.superblock = _unpack_struct_from_file(SUPERBLOCK, self._fh)

        assert self.superblock['format_signature'] == FORMAT_SIGNATURE
        # current support is only for version 0 superblock and 64-bit
        # addressing
        assert self.superblock['superblock_version'] == 0
        assert self.superblock['offset_size'] == 8
        assert self.superblock['length_size'] == 8

        assert self.superblock['free_space_address'] == UNDEFINED_ADDRESS
        address = self.superblock['driver_information_address']
        assert address == UNDEFINED_ADDRESS

        # BTree
        self.btree = []
        self._fh.seek(136)
        node = _unpack_struct_from_file(B_LINK_NODE_V1, self._fh)
        assert node['signature'] == b'TREE'
        self.btree.append(node)

        # Symbol table starts at 1072
        self._fh.seek(1072)
        node = _unpack_struct_from_file(SYMBOL_TABLE_NODE, self._fh)
        assert node['signature'] == b'SNOD'
        node['entries'] = [
            _unpack_struct_from_file(SYMBOL_TABLE_ENTRY, self._fh) for i in
            range(node['symbols'])]
        self.symbol_table = node

        # foo attribute with value of <float64> 99.5
        self._fh.seek(944)
        attr = _unpack_attribute(self._fh)
        self.foo = attr

        # bar attribute with value of <int64> 42
        self._fh.seek(1408)
        attr = _unpack_attribute(self._fh)
        self.bar = attr

        # HEAP : starts at 680
        self._fh.seek(680)
        local_heap = _unpack_struct_from_file(LOCAL_HEAP, self._fh)
        assert local_heap['signature'] == b'HEAP'
        assert local_heap['version'] == 0
        self.local_heap = local_heap

        # Data Object header
        self._fh.seek(800)
        data_object = _unpack_struct_from_file(OBJECT_HEADER_V1, self._fh)
        assert data_object['version'] == 1
        self.data_object = data_object
        #


    def close(self):
        """ Close the file. """
        self._fh.close()

def _unpack_attribute(fh):
    attr_dict = _unpack_struct_from_file(ATTRIBUTE_MESSAGE_HEADER, fh)

    true_name_size = int(np.ceil(attr_dict['name_size'] / 8.) * 8)
    true_datatype_size = int(np.ceil(attr_dict['datatype_size'] / 8.) * 8)
    true_dataspace_size = int(np.ceil(attr_dict['dataspace_size'] / 8.) * 8)

    attr_dict['name_data'] = fh.read(true_name_size)
    attr_dict['datatype_data'] = datatype = fh.read(true_datatype_size)
    attr_dict['dataspace'] = fh.read(true_dataspace_size)

    datatype = _unpack_struct_from(DATATYPE_MESSAGE,
                                   attr_dict['datatype_data'])
    datatype['version'] = datatype['class_and_version'] >> 4      # first 4 bits
    datatype['class'] = datatype['class_and_version'] & 0x0F  # last 4 bits
    attr_dict['datatype'] = datatype

    if datatype['class'] == 1:  # floating point, assume IEEE.
        # XXX check properties field to check that IEEEE
        if datatype['size'] == 8:
            attr_dict['value'] = struct.unpack('<d', fh.read(8))[0]
        else:
            raise NotImplementedError
    elif datatype['class'] == 0:  # fixed-point
        # XXX assuming signed, need to check Fix-point field properties
        if datatype['size'] == 8:
            attr_dict['value'] = struct.unpack('<q', fh.read(8))[0]
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    return attr_dict

# IV.A.2.d The Datatype Message

DATATYPE_MESSAGE = OrderedDict((
    ('class_and_version', 'B'),
    ('class_bit_field_0', 'B'),
    ('class_bit_field_1', 'B'),
    ('class_bit_field_2', 'B'),
    ('size', 'I'),
))

def _calc_structure_size(structure):
    fmt = '<' + ''.join(structure.values())
    return struct.calcsize(fmt)


def _unpack_struct_from_file(structure, fh):
    size = _calc_structure_size(structure)
    buf = fh.read(size)
    return _unpack_struct_from(structure, buf)


def _unpack_struct_from(structure, buf, offset=0):
    fmt = '<' + ''.join(structure.values())
    values = struct.unpack_from(fmt, buf, offset=offset)
    return OrderedDict(zip(structure.keys(), values))


# HDF5 Structures
# Values for all fields in this document should be treated as unsigned
# integers, unless otherwise noted in the description of a field. Additionally,
# all metadata fields are stored in little-endian byte order.

FORMAT_SIGNATURE = b'\211HDF\r\n\032\n'
UNDEFINED_ADDRESS = struct.unpack('<Q', b'\xff\xff\xff\xff\xff\xff\xff\xff')[0]

# Version 0 SUPERBLOCK
SUPERBLOCK = OrderedDict((
    ('format_signature', '8s'),

    ('superblock_version', 'B'),
    ('free_storage_version', 'B'),
    ('root_group_version', 'B'),
    ('reserved_0', 'B'),

    ('shared_header_version', 'B'),
    ('offset_size', 'B'),            # assume 8
    ('length_size', 'B'),            # assume 8
    ('reserved_1', 'B'),

    ('group_leaf_node_k', 'H'),
    ('group_internal_node_k', 'H'),

    ('file_consistency_flags', 'L'),

    ('base_address', 'Q'),                  # assume 8 byte addressing
    ('free_space_address', 'Q'),            # assume 8 byte addressing
    ('end_of_file_address', 'Q'),           # assume 8 byte addressing
    ('driver_information_address', 'Q'),    # assume 8 byte addressing

    ('root_group_symbol_table', 'L'),

))


B_LINK_NODE_V1 = OrderedDict((
    ('signature', '4s'),

    ('node_type', 'B'),
    ('node_level', 'B'),
    ('entries_used', 'H'),

    ('left_sibling', 'Q'),     # 8 byte addressing
    ('right_sibling', 'Q'),    # 8 byte addressing

    ('key', '8s'),
))


SYMBOL_TABLE_NODE = OrderedDict((
    ('signature', '4s'),
    ('version', 'B'),
    ('reserved_0', 'B'),
    ('symbols', 'H'),
))

SYMBOL_TABLE_ENTRY = OrderedDict((
    ('link_name_offset', 'Q'),     # 8 byte address
    ('object_header_address', 'Q'),
    ('cache_type', 'I'),
    ('reserved', 'I'),
    ('scratch', '16s'),
))

# IV.A.2.m The Attribute Message
ATTRIBUTE_MESSAGE_HEADER = OrderedDict((
    ('version', 'B'),
    ('reserved', 'B'),
    ('name_size', 'H'),
    ('datatype_size', 'H'),
    ('dataspace_size', 'H'),
))


# III.D Disk Format: Level 1D - Local Heaps
LOCAL_HEAP = OrderedDict((
    ('signature', '4s'),
    ('version', 'B'),
    ('reserved', '3s'),
    ('data_segment_size', 'Q'),         # 8 byte size of lengths
    ('offset_to_free_list', 'Q'),       # 8 bytes size of lengths
    ('address_of_data_segment', 'Q'),   # 8 byte addressing
))


# IV.A.1.a Version 1 Data Object Header Prefix
OBJECT_HEADER_V1 = OrderedDict((
    ('version', 'B'),
    ('reserved', 'B'),
    ('total_header_messaged', 'H'),
    ('object_reference_count', 'I'),
    ('object_header_size', 'I'),
))
