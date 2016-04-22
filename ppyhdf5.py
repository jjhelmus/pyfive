"""
ppyhdf5 : Pure Python HDF5 reader

"""

import struct
from collections import OrderedDict

import numpy as np


# Plans
# Classes
# -------
# Data Objects - read and store Data object header and Messages
# B-tree - nodes and leafs?
# Heap
# Symbol table / entry?
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
        self.btree = BTree(self._fh)

        self._fh.seek(self.root_group['address_of_heap'])
        self.heap = Heap(self._fh)

        self.symboltables = []
        self._dataset_offsets = {}
        for symbol_table_addreess in self.btree. symbol_table_addresses():
            self._fh.seek(symbol_table_addreess)
            table = SymbolTable(self._fh)
            table.assign_name(self.heap)

            self.symboltables.append(table)
            self._dataset_offsets.update(table.links_and_offsets())

        self.datasets = {k: Dataset(k, v, self._fh) for k, v in
                         self._dataset_offsets.items()}

        self._fh.seek(800)
        self.dataobjects = DataObjects(self._fh)

    def _get_superblock(self, fh):
        """ read in the superblock from an open file object. """
        superblock = _unpack_struct_from_file(SUPERBLOCK, fh)

        # check
        assert superblock['format_signature'] == FORMAT_SIGNATURE
        assert superblock['superblock_version'] == 0
        assert superblock['offset_size'] == 8
        assert superblock['length_size'] == 8
        assert superblock['free_space_address'] == UNDEFINED_ADDRESS
        assert superblock['driver_information_address'] == UNDEFINED_ADDRESS
        return superblock

    def _get_root_group(self, fh):
        """ Read in the root group symbol table entry from an open file. """

        # read in the symbol table entry
        entry = _unpack_struct_from_file(SYMBOL_TABLE_ENTRY, fh)
        assert entry['cache_type'] == 1

        # unpack the scratch area B-Tree and Heap addressed
        scratch = entry['scratch']
        address_of_btree, address_of_heap = struct.unpack('<QQ', scratch)
        entry['address_of_btree'] = address_of_btree
        entry['address_of_heap'] = address_of_heap

        # verify that the DataObject messages agree with the cache
        root_data_objects = DataObjects(fh)
        assert root_data_objects.messages[0]['type'] == 17
        assert root_data_objects.messages[0]['size'] == 16

        symbol_table_message = _unpack_struct_from(
            SYMBOL_TABLE_MESSAGE, root_data_objects.message_data,
            root_data_objects.messages[0]['offset_to_message'])
        assert symbol_table_message['btree_address'] == address_of_btree
        assert symbol_table_message['heap_address'] == address_of_heap
        entry['symbol_table_message'] = symbol_table_message

        return entry

    def close(self):
        """ Close the file. """
        self._fh.close()


class BTree(object):
    """
    Class for working with HDF5 B-Trees.
    """

    def __init__(self, fh):
        """ initalize. """
        self.btree = []
        node = _unpack_struct_from_file(B_LINK_NODE_V1, fh)
        assert node['signature'] == b'TREE'

        keys = []
        addresses = []
        for i in range(node['entries_used']):
            key = struct.unpack('<Q', fh.read(8))[0]
            address = struct.unpack('<Q', fh.read(8))[0]
            keys.append(key)
            addresses.append(address)
        # N+1 key
        keys.append(struct.unpack('<Q', fh.read(8))[0])
        node['keys'] = keys
        node['addresses'] = addresses


        self.btree.append(node)

    def symbol_table_addresses(self):
        """ Return a list of all symbol table address. """
        all_address = []
        for tree in self.btree:
            all_address.extend(tree['addresses'])
        return all_address

class Dataset(object):
    """
    Class to store and access data from a Dataset.
    """

    def __init__(self, name, offset, fh):
        """ initalize with fh position at the Data Object Header. """
        self.name = name
        fh.seek(offset)
        self.dataobjects = DataObjects(fh)

    def get_attributes(self):
        """ Return a dictionary of all attributes. """
        attrs = {}
        attr_msgs = [m for m in self.dataobjects.messages if m['type'] == 12]
        for attr_msg in attr_msgs:
            start = attr_msg['offset_to_message']
            size = attr_msg['size']
            attr_bytes = self.dataobjects.message_data[start:start+size]
            name, value = _unpack_attribute_from_bytes(attr_bytes)
            attrs[name] = value
        return attrs


class Heap(object):
    """
    Class to store the heap.
    """

    def __init__(self, fh):
        """ initalize. """
        # HEAP : starts at 680
        local_heap = _unpack_struct_from_file(LOCAL_HEAP, fh)
        assert local_heap['signature'] == b'HEAP'
        assert local_heap['version'] == 0
        fh.seek(local_heap['address_of_data_segment'])
        heap_data = fh.read(local_heap['data_segment_size'])
        local_heap['heap_data'] = heap_data
        self.local_heap = local_heap

    def get_object_name(self, offset):
        """ Return the name of the object indicated by the given offset. """
        end = self.local_heap['heap_data'].index(b'\x00', offset)
        return self.local_heap['heap_data'][offset:end]


class SymbolTable(object):
    """
    Class to store Symbol Tables
    """

    def __init__(self, fh):
        node = _unpack_struct_from_file(SYMBOL_TABLE_NODE, fh)
        assert node['signature'] == b'SNOD'
        node['entries'] = [
            _unpack_struct_from_file(SYMBOL_TABLE_ENTRY, fh) for i in
            range(node['symbols'])]
        self.symbol_table = node

    def assign_name(self, heap):
        """ Assign link names to all entries in the symbol table. """
        for entry in self.symbol_table['entries']:
            offset = entry['link_name_offset']
            link_name = heap.get_object_name(offset).decode('utf-8')
            entry['link_name'] = link_name
        return

    def links_and_offsets(self):
        """ Return a dictionary of link name: offsets. """
        return {e['link_name']: e['object_header_address'] for e in
                self.symbol_table['entries']}





class DataObjects(object):
    """
    Class for storing a collection of data objects.
    """

    def __init__(self, fh):
        """ Initalize from open file or file like object. """
        header = _unpack_struct_from_file(OBJECT_HEADER_V1, fh)
        assert header['version'] == 1
        message_data = fh.read(header['object_header_size'])

        offset = 0
        messages = []
        for i in range(header['total_header_messages']):
            info = _unpack_struct_from(
                HEADER_MESSAGE_INFO, message_data, offset)
            info['offset_to_message'] = offset + 8
            if info['type'] == 0x0010:  # object header continuation
                fh_offset, size = struct.unpack_from(
                    '<QQ', message_data, offset + 8)
                fh.seek(fh_offset)
                message_data += fh.read(size)
            messages.append(info)
            offset += 8 + info['size']

        self.messages = messages
        self.message_data = message_data
        self._header = header


def _unpack_attribute_from_bytes(attr_bytes):

    attr_dict = _unpack_struct_from(ATTRIBUTE_MESSAGE_HEADER, attr_bytes)

    true_name_size = int(np.ceil(attr_dict['name_size'] / 8.) * 8)
    true_datatype_size = int(np.ceil(attr_dict['datatype_size'] / 8.) * 8)
    true_dataspace_size = int(np.ceil(attr_dict['dataspace_size'] / 8.) * 8)

    pos = 8
    attr_dict['name_data'] = attr_bytes[pos:pos+true_name_size]
    pos += true_name_size

    attr_dict['datatype_data'] = attr_bytes[pos:pos+true_datatype_size]
    pos += true_datatype_size

    attr_dict['dataspace'] = attr_bytes[pos:pos+true_dataspace_size]
    pos += true_dataspace_size

    datatype = _unpack_struct_from(DATATYPE_MESSAGE,
                                   attr_dict['datatype_data'])
    datatype['version'] = datatype['class_and_version'] >> 4      # first 4 bits
    datatype['class'] = datatype['class_and_version'] & 0x0F  # last 4 bits
    attr_dict['datatype'] = datatype

    if datatype['class'] == 1:  # floating point, assume IEEE.
        # XXX check properties field to check that IEEEE
        if datatype['size'] == 8:
            attr_dict['value'] = struct.unpack_from('<d', attr_bytes, pos)[0]
        else:
            raise NotImplementedError
    elif datatype['class'] == 0:  # fixed-point
        # XXX assuming signed, need to check Fix-point field properties
        if datatype['size'] == 8:
            attr_dict['value'] = struct.unpack_from('<q', attr_bytes, pos)[0]
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    return (attr_dict['name_data'].decode('utf-8').strip('\x00'),
            attr_dict['value'])


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

))


B_LINK_NODE_V1 = OrderedDict((
    ('signature', '4s'),

    ('node_type', 'B'),
    ('node_level', 'B'),
    ('entries_used', 'H'),

    ('left_sibling', 'Q'),     # 8 byte addressing
    ('right_sibling', 'Q'),    # 8 byte addressing
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
    ('total_header_messages', 'H'),
    ('object_reference_count', 'I'),
    ('object_header_size', 'I'),
    ('padding', 'I'),
))


#
HEADER_MESSAGE_INFO = OrderedDict((
    ('type', 'H'),
    ('size', 'H'),
    ('flags', 'B'),
    ('reserved', '3s'),
))


SYMBOL_TABLE_MESSAGE = OrderedDict((
    ('btree_address', 'Q'),     # 8 bytes addressing
    ('heap_address', 'Q'),      # 8 byte addressing
))
