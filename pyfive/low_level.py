""" low-level classes for reading HDF5 files.  """

import struct
from collections import OrderedDict

import numpy as np


class SuperBlock(object):
    """
    Class for represeting the HDF5 Superblock
    """

    def __init__(self, fh):
        superblock = _unpack_struct_from_file(SUPERBLOCK, fh)

        # check
        assert superblock['format_signature'] == FORMAT_SIGNATURE
        assert superblock['superblock_version'] == 0
        assert superblock['offset_size'] == 8
        assert superblock['length_size'] == 8
        assert superblock['free_space_address'] == UNDEFINED_ADDRESS
        assert superblock['driver_information_address'] == UNDEFINED_ADDRESS
        self._contents = superblock



class RootGroup(object):
    """
    Class for representing the HDF5 Root Group.
    """

    def __init__(self, fh):
        # Read in the root group symbol table entry from an open file.
        # read in the symbol table entry
        entry = _unpack_struct_from_file(SYMBOL_TABLE_ENTRY, fh)
        assert entry['cache_type'] == 1

        # unpack the scratch area B-Tree and Heap addressed
        scratch = entry['scratch']
        address_of_btree, address_of_heap = struct.unpack('<QQ', scratch)
        self.address_of_btree = address_of_btree
        self.address_of_heap = address_of_heap
        self._contents = entry

        self.dataobjects = DataObjects(fh)
        self._verify_cache()

    def _verify_cache(self):
        # verify that the DataObject messages agree with the cache
        msgs = self.dataobjects.find_msg_type(SYMBOL_TABLE_MSG_TYPE)
        assert len(msgs) == 1

        assert msgs[0]['type'] == SYMBOL_TABLE_MSG_TYPE
        assert msgs[0]['size'] == 16
        symbol_table_message = _unpack_struct_from(
            SYMBOL_TABLE_MESSAGE, self.dataobjects.msg_data,
            msgs[0]['offset_to_message'])
        assert symbol_table_message['btree_address'] == self.address_of_btree
        assert symbol_table_message['heap_address'] == self.address_of_heap





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
            if info['type'] == OBJECT_CONTINUATION_MSG_TYPE:
                fh_offset, size = struct.unpack_from(
                    '<QQ', message_data, offset + 8)
                fh.seek(fh_offset)
                message_data += fh.read(size)
            messages.append(info)
            offset += 8 + info['size']

        self.msgs = messages
        self.msg_data = message_data
        self._header = header
        self.fh = fh

    def get_attributes(self):
        """ Return a dictionary of all attributes. """
        attrs = {}
        attr_msgs = self.find_msg_type(ATTRIBUTE_MSG_TYPE)
        for msg in attr_msgs:
            offset = msg['offset_to_message']
            name, value = unpack_attribute(self.msg_data, offset)
            attrs[name] = value
        return attrs

    def get_data(self):
        msg = self.find_msg_type(DATA_STORAGE_MSG_TYPE)[0]
        start = msg['offset_to_message']
        version, layout_class, offset, size = struct.unpack_from(
            '<BBQQ', self.msg_data, start)
        self.fh.seek(offset)
        buf = self.fh.read(size)
        dtype='<i4'
        shape = (100, )
        return np.frombuffer(buf, dtype=dtype).reshape(shape)

    def find_msg_type(self, msg_type):
        """ Return a list of all messages of a given type. """
        return [m for m in self.msgs if m['type'] == msg_type]


def unpack_attribute(buf, offset=0):
    """ Return the attribute name and value. """

    attr_dict = _unpack_struct_from(ATTRIBUTE_MESSAGE_HEADER, buf, offset)
    assert attr_dict['version'] == 1
    offset += ATTRIBUTE_MESSAGE_HEADER_SIZE

    # read in the attribute name
    name_size = attr_dict['name_size']
    name = buf[offset:offset+name_size].strip(b'\x00').decode('utf-8')
    offset += _padded_size(name_size)

    # read in the datatype information
    datatype = _unpack_struct_from(DATATYPE_MESSAGE, buf, offset)
    dtype_version = datatype['class_and_version'] >> 4  # first 4 bits
    dtype_class = datatype['class_and_version'] & 0x0F  # last 4 bits
    offset += _padded_size(attr_dict['datatype_size'])

    # read in the dataspace information
    dataspace_size = attr_dict['dataspace_size']
    attr_dict['dataspace'] = buf[offset:offset+dataspace_size]
    offset += _padded_size(dataspace_size)

    if dtype_class == 1:  # floating point, assume IEEE.
        # XXX check properties field to check that IEEEE
        if datatype['size'] == 8:
            value = struct.unpack_from('<d', buf, offset)[0]
        else:
            raise NotImplementedError
    elif dtype_class == 0:  # fixed-point
        # XXX assuming signed, need to check Fix-point field properties
        if datatype['size'] == 8:
            value = struct.unpack_from('<q', buf, offset)[0]
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    return name, value


def _padded_size(size, padding_multipe=8):
    """ Return the size of a field padded to be a multiple a give value. """
    return int(np.ceil(size / padding_multipe) * padding_multipe)


# IV.A.2.d The Datatype Message

DATATYPE_MESSAGE = OrderedDict((
    ('class_and_version', 'B'),
    ('class_bit_field_0', 'B'),
    ('class_bit_field_1', 'B'),
    ('class_bit_field_2', 'B'),
    ('size', 'I'),
))

def _structure_size(structure):
    fmt = '<' + ''.join(structure.values())
    return struct.calcsize(fmt)


def _unpack_struct_from_file(structure, fh):
    size = _structure_size(structure)
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
ATTRIBUTE_MESSAGE_HEADER_SIZE = _structure_size(ATTRIBUTE_MESSAGE_HEADER)


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


# Data Object Message types
# Section IV.A.2.a - IV.A.2.x
NIL_MSG_TYPE = 0x0000
DATASPACE_MSG_TYPE = 0x0001
LINK_INFO_MSG_TYPE = 0x0002
DATATYPE_MSG_TYPE = 0x0003
FILLVALUE_OLD_MSG_TYPE = 0x0004
FILLVALUE_MSG_TYPE = 0x0005
LINK_MSG_TYPE = 0x0006
EXTERNAL_DATA_FILES_MSG_TYPE = 0x0007
DATA_STORAGE_MSG_TYPE = 0x0008
BOGUS_MSG_TYPE = 0x0009
GROUP_INFO_MSG_TYPE = 0x000A
DATA_STORAGE_FILTER_PIPELINE_MSG_TYPE = 0x000B
ATTRIBUTE_MSG_TYPE = 0x000C
OBJECT_COMMENT_MSG_TYPE = 0x000D
OBJECT_MODIFICATION_TIME_OLD_MSG_TYPE = 0x000E
SHARED_MESSAGE_TABLE_MSG_TYPE = 0x000F
OBJECT_CONTINUATION_MSG_TYPE = 0x0010
SYMBOL_TABLE_MSG_TYPE = 0x0011
OBJECT_MODIFICATION_TIME_MSG_TYPE = 0x0012
BTREE_K_VALUE_MSG_TYPE = 0x0013
DRIVER_INFO_MSG_TYPE = 0x0014
ATTRIBUTE_INFO_MSG_TYPE = 0x0015
OBJECT_REFERENCE_COUNT_MSG_TYPE = 0x0016
FILE_SPACE_INFO_MSG_TYPE = 0x0018
