""" Low-level classes for reading HDF5 files.  """

from __future__ import division

from collections import OrderedDict
import struct
import warnings

import numpy as np


class InvalidHDF5File(Exception):
    """ Exception raised when an invalid HDF5 file is detected. """
    pass


class SuperBlock(object):
    """
    HDF5 Superblock.
    """

    def __init__(self, fh, offset):
        """ initalize. """

        fh.seek(offset)
        version_hint = struct.unpack_from('<B', fh.peek(9), 8)[0]
        if version_hint == 0:
            contents = _unpack_struct_from_file(SUPERBLOCK_V0, fh)
        elif version_hint == 2:
            contents = _unpack_struct_from_file(SUPERBLOCK_V2, fh)
        else:
            raise NotImplementedError(
                "unsupported superblock version: %i" % (version_hint))

        # verify contents
        if contents['format_signature'] != FORMAT_SIGNATURE:
            raise InvalidHDF5File('Incorrect file signature')
        if contents['offset_size'] != 8 or contents['length_size'] != 8:
            raise NotImplementedError('File uses none 64-bit addressing')
        self.version = contents['superblock_version']
        self._contents = contents
        self._offset = fh.tell()
        self._root_symbol_table = None
        self._fh = fh

    @property
    def offset_to_dataobjects(self):
        """ The offset to the data objects collection for the superblock. """
        if self.version == 0:
            sym_table = SymbolTable(self._fh, self._offset, root=True)
            self._root_symbol_table = sym_table
            return sym_table.group_offset
        elif self.version == 2:
            return self._contents['root_group_address']
        else:
            raise NotImplementedError


class BTree(object):
    """
    HDF5 version 1 B-Tree.
    """

    def __init__(self, fh, offset):
        """ initalize. """
        fh.seek(offset)
        self.nodes = []
        node = _unpack_struct_from_file(B_LINK_NODE_V1, fh)
        assert node['signature'] == b'TREE'

        keys = []
        addresses = []
        for _ in range(node['entries_used']):
            key = struct.unpack('<Q', fh.read(8))[0]
            address = struct.unpack('<Q', fh.read(8))[0]
            keys.append(key)
            addresses.append(address)
        # N+1 key
        keys.append(struct.unpack('<Q', fh.read(8))[0])
        node['keys'] = keys
        node['addresses'] = addresses

        self.nodes.append(node)

    def symbol_table_addresses(self):
        """ Return a list of all symbol table address. """
        all_address = []
        for node in self.nodes:
            all_address.extend(node['addresses'])
        return all_address


class BTreeRawDataChunks(object):
    """
    HDF5 version 1 B-Tree storing raw data chunk nodes (type 1).
    """

    def __init__(self, fh, offset, dims):
        """ initalize. """
        self.fh = fh
        self.dims = dims

        # read in the root node
        root_node = self._read_node(offset)
        self.root_node = root_node

        # read in all other nodes
        all_nodes = {}
        node_level = root_node['node_level']
        all_nodes[node_level] = [root_node]
        while node_level != 0:
            new_nodes = []
            for parent_node in all_nodes[node_level]:
                for addr in parent_node['addresses']:
                    new_nodes.append(self._read_node(addr))
            new_node_level = new_nodes[0]['node_level']
            all_nodes[new_node_level] = new_nodes
            node_level = new_node_level

        self.all_nodes = all_nodes

    def _read_node(self, offset):
        """ Return a single node in the b-tree located at a give offset. """
        self.fh.seek(offset)
        node = _unpack_struct_from_file(B_LINK_NODE_V1, self.fh)
        assert node['signature'] == b'TREE'
        assert node['node_type'] == 1

        keys = []
        addresses = []
        for _ in range(node['entries_used']):
            chunk_size, filter_mask = struct.unpack('<II', self.fh.read(8))
            fmt = '<' + 'Q' * self.dims
            fmt_size = struct.calcsize(fmt)
            chunk_offset = struct.unpack(fmt, self.fh.read(fmt_size))
            chunk_address = struct.unpack('<Q', self.fh.read(8))[0]

            keys.append(OrderedDict((
                ('chunk_size', chunk_size),
                ('filter_mask', filter_mask),
                ('chunk_offset', chunk_offset),
            )))
            addresses.append(chunk_address)
        node['keys'] = keys
        node['addresses'] = addresses
        return node

    def construct_data_from_chunks(self, chunk_shape, data_shape, dtype):
        """ Build a complete data array from chunks. """
        # create array to store data
        shape = [_padded_size(i, j) for i, j in zip(data_shape, chunk_shape)]
        data = np.zeros(shape, dtype=dtype)

        # loop over chunks reading each into the full data array
        count = np.prod(chunk_shape)
        chunk_buffer_size = count * np.dtype(dtype).itemsize
        for node in self.all_nodes[0]:
            for node_key, addr in zip(node['keys'], node['addresses']):
                self.fh.seek(addr)
                chunk_buffer = self.fh.read(chunk_buffer_size)
                chunk_data = np.frombuffer(chunk_buffer, dtype=dtype)
                start = node_key['chunk_offset'][:-1]
                region = [slice(i, i+j) for i, j in zip(start, chunk_shape)]
                data[region] = chunk_data.reshape(chunk_shape)

        non_padded_region = [slice(i) for i in data_shape]
        return data[non_padded_region]


class Heap(object):
    """
    HDF5 local heap.
    """

    def __init__(self, fh, offset):
        """ initalize. """

        fh.seek(offset)
        local_heap = _unpack_struct_from_file(LOCAL_HEAP, fh)
        assert local_heap['signature'] == b'HEAP'
        assert local_heap['version'] == 0
        fh.seek(local_heap['address_of_data_segment'])
        heap_data = fh.read(local_heap['data_segment_size'])
        local_heap['heap_data'] = heap_data
        self._contents = local_heap
        self.data = heap_data

    def get_object_name(self, offset):
        """ Return the name of the object indicated by the given offset. """
        end = self.data.index(b'\x00', offset)
        return self.data[offset:end]


class SymbolTable(object):
    """
    HDF5 Symbol Table.
    """

    def __init__(self, fh, offset, root=False):
        """ initialize, root=True for the root group, False otherwise. """

        fh.seek(offset)
        if root:
            # The root symbol table has no Symbol table node header
            # and contains only a single entry
            node = OrderedDict([('symbols', 1)])
        else:
            node = _unpack_struct_from_file(SYMBOL_TABLE_NODE, fh)
            assert node['signature'] == b'SNOD'
        entries = [_unpack_struct_from_file(SYMBOL_TABLE_ENTRY, fh) for i in
                   range(node['symbols'])]
        if root:
            self.group_offset = entries[0]['object_header_address']
        self.entries = entries
        self._contents = node

    def assign_name(self, heap):
        """ Assign link names to all entries in the symbol table. """
        for entry in self.entries:
            offset = entry['link_name_offset']
            link_name = heap.get_object_name(offset).decode('utf-8')
            entry['link_name'] = link_name
        return

    def get_links(self):
        """ Return a dictionary of links (dataset/group) and offsets. """
        return {e['link_name']: e['object_header_address'] for e in
                self.entries}


class GlobalHeap(object):
    """
    HDF5 Global Heap collection.
    """

    def __init__(self, fh, offset):

        fh.seek(offset)
        header = _unpack_struct_from_file(GLOBAL_HEAP_HEADER, fh)
        assert header['signature'] == b'GCOL'
        assert header['version'] == 1
        heap_data_size = header['collection_size'] - GLOBAL_HEAP_HEADER_SIZE
        heap_data = fh.read(heap_data_size)
        assert len(heap_data) == heap_data_size  # check for early end of file

        self.heap_data = heap_data
        self._header = header
        self._objects = None

    @property
    def objects(self):
        """ Dictionary of objects in the heap. """
        if self._objects is None:
            self._objects = OrderedDict()
            offset = 0
            while offset < len(self.heap_data):
                info = _unpack_struct_from(
                    GLOBAL_HEAP_OBJECT, self.heap_data, offset)
                if info['object_index'] == 0:
                    break
                offset += GLOBAL_HEAP_OBJECT_SIZE
                fmt = '<' + str(info['object_size']) + 's'
                obj_data = struct.unpack_from(fmt, self.heap_data, offset)[0]
                self._objects[info['object_index']] = obj_data
                offset += _padded_size(info['object_size'])
        return self._objects


class DataObjects(object):
    """
    HDF5 DataObjects.
    """

    def __init__(self, fh, offset):
        """ initalize. """
        fh.seek(offset)
        version_hint = struct.unpack_from('<B', fh.peek(1))[0]
        if version_hint == 1:
            msgs, msg_data, header = self._parse_v1_objects(fh)
        elif version_hint == ord('O'):   # first character of v2 signature
            msgs, msg_data, header = self._parse_v2_objects(fh)
        else:
            raise InvalidHDF5File('unknown Data Object Header')

        self.fh = fh
        self.msgs = msgs
        self.msg_data = msg_data
        self._global_heaps = {}
        self._header = header

    @staticmethod
    def _parse_v1_objects(fh):
        """ Parse a collection of version 1 Data Objects. """
        header = _unpack_struct_from_file(OBJECT_HEADER_V1, fh)
        assert header['version'] == 1
        msg_data = fh.read(header['object_header_size'])

        offset = 0
        msgs = []
        for _ in range(header['total_header_messages']):
            msg = _unpack_struct_from(HEADER_MSG_INFO_V1, msg_data, offset)
            msg['offset_to_message'] = offset + 8
            if msg['type'] == OBJECT_CONTINUATION_MSG_TYPE:
                fh_off, size = struct.unpack_from('<QQ', msg_data, offset + 8)
                fh.seek(fh_off)
                msg_data += fh.read(size)
            msgs.append(msg)
            offset += 8 + msg['size']
        return msgs, msg_data, header

    def _parse_v2_objects(self, fh):
        """ Parse a collection of version 2 Data Objects. """

        header, creation_order_size = self._parse_v2_header(fh)

        msgs = []
        msg_data = fh.read(header['size_of_chunk_0'])
        offset = 0
        chunk_sizes = [header['size_of_chunk_0']]
        current_chunk = 0
        size_of_processed_chunks = 0

        while offset < (len(msg_data) - 4):
            msg = _unpack_struct_from(HEADER_MSG_INFO_V2, msg_data, offset)
            msg['offset_to_message'] = offset + 4 + creation_order_size

            if msg['type'] == OBJECT_CONTINUATION_MSG_TYPE:
                fh_off, size = struct.unpack_from(
                    '<QQ', msg_data, offset + 4 + creation_order_size)
                fh.seek(fh_off)
                new_msg_data = fh.read(size)
                assert new_msg_data[:4] == b'OCHK'
                chunk_sizes.append(size-4)
                msg_data += new_msg_data[4:]

            msgs.append(msg)
            offset += 4 + msg['size'] + creation_order_size

            chunk_offset = offset - size_of_processed_chunks
            if (chunk_offset + 4) >= chunk_sizes[current_chunk]:
                # move to next chunk
                current_chunk_size = chunk_sizes[current_chunk]
                offset += (current_chunk_size - chunk_offset)
                size_of_processed_chunks += current_chunk_size
                current_chunk += 1

        return msgs, msg_data, header

    @staticmethod
    def _parse_v2_header(fh):
        """ Parse a version 2 data object header. """
        header = _unpack_struct_from_file(OBJECT_HEADER_V2, fh)
        assert header['version'] == 2
        if header['flags'] & 2**2:
            creation_order_size = 2
        else:
            creation_order_size = 0
        assert (header['flags'] & 2**4) == 0
        if header['flags'] & 2**5:
            times = struct.unpack('<4I', fh.read(16))
            header['access_time'] = times[0]
            header['modification_time'] = times[1]
            header['change_time'] = times[2]
            header['birth_time'] = times[3]
        chunk_fmt = ['<B', '<H', '<I', '<Q'][(header['flags'] & 3)]
        header['size_of_chunk_0'] = struct.unpack(
            chunk_fmt, fh.read(struct.calcsize(chunk_fmt)))[0]
        return header, creation_order_size

    def get_attributes(self):
        """ Return a dictionary of all attributes. """
        attrs = {}
        attr_msgs = self.find_msg_type(ATTRIBUTE_MSG_TYPE)
        for msg in attr_msgs:
            offset = msg['offset_to_message']
            name, value = self.unpack_attribute(offset)
            attrs[name] = value
        # TODO attributes may also be stored in objects reference in the
        # Attribute Info Message (0x0015, 21).
        return attrs

    def unpack_attribute(self, offset):
        """ Return the attribute name and value. """

        # read in the attribute message header
        version = struct.unpack_from('<B', self.msg_data, offset)[0]
        if version == 1:
            attr_dict = _unpack_struct_from(
                ATTR_MSG_HEADER_V1, self.msg_data, offset)
            assert attr_dict['version'] == 1
            offset += ATTR_MSG_HEADER_V1_SIZE
            padding_multiple = 8
        elif version == 3:
            attr_dict = _unpack_struct_from(
                ATTR_MSG_HEADER_V3, self.msg_data, offset)
            assert attr_dict['version'] == 3
            offset += ATTR_MSG_HEADER_V3_SIZE
            padding_multiple = 1    # no padding
        else:
            raise NotImplementedError(
                "unsupported attribute message version: %i" % (version))

        # read in the attribute name
        name_size = attr_dict['name_size']
        name = self.msg_data[offset:offset+name_size]
        name = name.strip(b'\x00').decode('utf-8')
        offset += _padded_size(name_size, padding_multiple)

        # read in the datatype information
        try:
            dtype = determine_dtype(self.msg_data, offset)
        except NotImplementedError:
            warnings.warn(
                'Attribute %s type not implemented, set to None.' % (name, ))
            return name, None
        offset += _padded_size(attr_dict['datatype_size'], padding_multiple)

        # read in the dataspace information
        offset += _padded_size(attr_dict['dataspace_size'], padding_multiple)

        # read in the value
        if isinstance(dtype, tuple):
            value = self._vlen_attr_value(offset, dtype)
        else:
            value = np.frombuffer(
                self.msg_data, dtype=dtype, count=1, offset=offset)[0]
        return name, value

    def _vlen_attr_value(self, offset, info):
        """ Return the value of a variable length attribute. """
        vlen_type, padding_type, character_set = info
        vlen_size, gheap_address, gheap_index = struct.unpack_from(
            '<IQI', self.msg_data, offset)
        if gheap_address not in self._global_heaps:
            # load the global heap and cache the instance
            gheap = GlobalHeap(self.fh, gheap_address)
            self._global_heaps[gheap_address] = gheap
        gheap = self._global_heaps[gheap_address]
        value = gheap.objects[gheap_index]
        if character_set == 0:
            # ascii character set, return as bytes
            value = value
        else:
            value = value.decode('utf-8')
        return value

    @property
    def shape(self):
        """ Shape of the dataset. """
        msg = self.find_msg_type(DATASPACE_MSG_TYPE)[0]
        msg_offset = msg['offset_to_message']
        return determine_data_shape(self.msg_data, msg_offset)

    @property
    def dtype(self):
        """ Datatype of the dataset. """
        msg = self.find_msg_type(DATATYPE_MSG_TYPE)[0]
        msg_offset = msg['offset_to_message']
        return determine_dtype(self.msg_data, msg_offset)

    def get_data(self):
        """ Return the data pointed to in the DataObject. """

        # offset and size from data storage message
        msg = self.find_msg_type(DATA_STORAGE_MSG_TYPE)[0]
        msg_offset = msg['offset_to_message']
        version, layout_class = struct.unpack_from(
            '<BB', self.msg_data, msg_offset)
        assert version == 3

        if layout_class == 2:  # chunked storage
            return self._get_chunked_data(msg_offset)

        assert layout_class == 1
        data_offset, size = struct.unpack_from(
                '<QQ', self.msg_data, msg_offset+2)
        if data_offset == UNDEFINED_ADDRESS:
            # no storage is backing array, return all zeros
            return np.zeros(self.shape, dtype=self.dtype)

        # return a memory-map to the stored array with copy-on-write
        return np.memmap(self.fh, dtype=self.dtype, mode='c',
                         offset=data_offset, shape=self.shape, order='C')

    def _get_chunked_data(self, offset):
        """ Return data which is chunked. """
        version, layout_class = struct.unpack_from(
            '<BB', self.msg_data, offset)
        offset += struct.calcsize('<BB')
        assert version == 3
        assert layout_class == 2

        dims, address = struct.unpack_from('<BQ', self.msg_data, offset)
        offset += struct.calcsize('<BQ')

        fmt = '<' + 'I' * (dims-1)
        chunk_shape = struct.unpack_from(fmt, self.msg_data, offset)
        offset += struct.calcsize(fmt)

        element_size = struct.unpack_from('<I', self.msg_data, offset)

        chunk_btree = BTreeRawDataChunks(self.fh, address, dims)

        return chunk_btree.construct_data_from_chunks(
            chunk_shape, self.shape, self.dtype)

    def find_msg_type(self, msg_type):
        """ Return a list of all messages of a given type. """
        return [m for m in self.msgs if m['type'] == msg_type]

    def get_links(self):
        """ Return a dictionary of link_name: offset """
        sym_tbl_msgs = self.find_msg_type(SYMBOL_TABLE_MSG_TYPE)
        if len(sym_tbl_msgs):
            return self._get_links_from_symbol_tables(sym_tbl_msgs)
        else:
            return self._get_links_from_link_msgs()

    def _get_links_from_symbol_tables(self, sym_tbl_msgs):
        """ Return a dict of link_name: offset from a symbol table. """
        assert len(sym_tbl_msgs) == 1
        assert sym_tbl_msgs[0]['size'] == 16
        symbol_table_message = _unpack_struct_from(
            SYMBOL_TABLE_MSG, self.msg_data,
            sym_tbl_msgs[0]['offset_to_message'])

        btree = BTree(self.fh, symbol_table_message['btree_address'])
        heap = Heap(self.fh, symbol_table_message['heap_address'])
        links = {}
        for symbol_table_address in btree.symbol_table_addresses():
            table = SymbolTable(self.fh, symbol_table_address)
            table.assign_name(heap)
            links.update(table.get_links())
        return links

    def _get_links_from_link_msgs(self):
        """ Retrieve links from link messages. """
        links = {}
        link_msgs = self.find_msg_type(LINK_MSG_TYPE)
        for link_msg in link_msgs:
            offset = link_msg['offset_to_message']
            version, flags = struct.unpack_from('<BB', self.msg_data, offset)
            offset += 2
            assert version == 1
            assert flags & 2**0 == 0
            assert flags & 2**1 == 0
            assert flags & 2**3 == 0
            assert flags & 2**4 == 0
            if flags & 2**2:
                # creation order present
                offset += 8

            encoding = 'ascii'

            name_size = struct.unpack_from('<B', self.msg_data, offset)[0]
            offset += 1
            name = self.msg_data[offset:offset+name_size].decode(encoding)
            offset += name_size

            address = struct.unpack_from('<Q', self.msg_data, offset)[0]
            links[name] = address
        return links

    @property
    def is_dataset(self):
        """ True when DataObjects points to a dataset, False for a group. """
        return len(self.find_msg_type(DATASPACE_MSG_TYPE)) > 0


def determine_data_shape(buf, offset):
    """ Return the shape of the dataset pointed to in a Dataspace message. """
    version = struct.unpack_from('<B', buf, offset)[0]
    if version == 1:
        header = _unpack_struct_from(DATASPACE_MSG_HEADER_V1, buf, offset)
        assert header['version'] == 1
        offset += DATASPACE_MSG_HEADER_V1_SIZE
    elif version == 2:
        header = _unpack_struct_from(DATASPACE_MSG_HEADER_V2, buf, offset)
        assert header['version'] == 2
        offset += DATASPACE_MSG_HEADER_V2_SIZE
    else:
        raise InvalidHDF5File('unknown dataspace message version')

    ndims = header['dimensionality']
    dim_sizes = struct.unpack_from('<' + 'Q' * ndims, buf, offset)
    # Dimension maximum size follows if header['flags'] bit 0 set
    # Permutation index follows if header['flags'] bit 1 set
    return dim_sizes


def determine_dtype(buf, offset):
    """
    Return the numpy dtype from a buffer pointing to a Datatype message.
    """
    datatype_msg = _unpack_struct_from(DATATYPE_MSG, buf, offset)
    datatype_class = datatype_msg['class_and_version'] & 0x0F  # last 4 bits

    if datatype_class == DATATYPE_FIXED_POINT:
        return _determine_dtype_fixed_point(datatype_msg)
    elif datatype_class == DATATYPE_FLOATING_POINT:
        return _determine_dtype_floating_point(datatype_msg)
    elif datatype_class == DATATYPE_TIME:
        raise NotImplementedError("Time datatype class not supported.")
    elif datatype_class == DATATYPE_STRING:
        return _determine_dtype_string(datatype_msg)
    elif datatype_class == DATATYPE_BITFIELD:
        raise NotImplementedError("Bitfield datatype class not supported.")
    elif datatype_class == DATATYPE_OPAQUE:
        raise NotImplementedError("Opaque datatype class not supported.")
    elif datatype_class == DATATYPE_COMPOUND:
        raise NotImplementedError("Compound datatype class not supported.")
    elif datatype_class == DATATYPE_REFERENCE:
        raise NotImplementedError("Reference datatype class not supported.")
    elif datatype_class == DATATYPE_ENUMERATED:
        raise NotImplementedError("Enumerated datatype class not supported.")
    elif datatype_class == DATATYPE_ARRAY:
        raise NotImplementedError("Array datatype class not supported.")
    elif datatype_class == DATATYPE_VARIABLE_LENGTH:
        return _determine_dtype_vlen(datatype_msg)
    else:
        raise InvalidHDF5File('Invalid datatype class %i' % (datatype_class))


def _determine_dtype_fixed_point(datatype_msg):
    """ Return the NumPy dtype for a fixed point class. """
    # fixed-point types are assumed to follow IEEE standard format
    length_in_bytes = datatype_msg['size']
    if length_in_bytes not in [1, 2, 4, 8]:
        raise NotImplementedError("Unsupported datatype size")

    signed = datatype_msg['class_bit_field_0'] & 0x08
    if signed > 0:
        dtype_char = 'i'
    else:
        dtype_char = 'u'

    byte_order = datatype_msg['class_bit_field_0'] & 0x01
    if byte_order == 0:
        byte_order_char = '<'  # little-endian
    else:
        byte_order_char = '>'  # big-endian

    return byte_order_char + dtype_char + str(length_in_bytes)


def _determine_dtype_floating_point(datatype_msg):
    """ Return the NumPy dtype for a floating point class. """
    # Floating point types are assumed to follow IEEE standard formats
    length_in_bytes = datatype_msg['size']
    if length_in_bytes not in [1, 2, 4, 8]:
        raise NotImplementedError("Unsupported datatype size")

    dtype_char = 'f'

    byte_order = datatype_msg['class_bit_field_0'] & 0x01
    if byte_order == 0:
        byte_order_char = '<'  # little-endian
    else:
        byte_order_char = '>'  # big-endian

    return byte_order_char + dtype_char + str(length_in_bytes)


def _determine_dtype_string(datatype_msg):
    """ Return the NumPy dtype for a string class. """
    return 'S' + str(datatype_msg['size'])


def _determine_dtype_vlen(datatype_msg):
    """ Return the dtype information for a variable length class. """
    vlen_type = datatype_msg['class_bit_field_0'] & 0x01
    if vlen_type != 1:
        raise NotImplementedError(
            "Non-string variable length datatypes not supported.")
    padding_type = datatype_msg['class_bit_field_0'] >> 4  # bits 4-7
    character_set = datatype_msg['class_bit_field_1'] & 0x01
    return ('VLEN_STRING', padding_type, character_set)


def _padded_size(size, padding_multipe=8):
    """ Return the size of a field padded to be a multiple a give value. """
    return int(np.ceil(size / padding_multipe) * padding_multipe)


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


# HDF5 Structures
# Values for all fields in this document should be treated as unsigned
# integers, unless otherwise noted in the description of a field. Additionally,
# all metadata fields are stored in little-endian byte order.

FORMAT_SIGNATURE = b'\211HDF\r\n\032\n'
UNDEFINED_ADDRESS = struct.unpack('<Q', b'\xff\xff\xff\xff\xff\xff\xff\xff')[0]

# Version 0 SUPERBLOCK
SUPERBLOCK_V0 = OrderedDict((
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

# Version 2 SUPERBLOCK
SUPERBLOCK_V2 = OrderedDict((
    ('format_signature', '8s'),

    ('superblock_version', 'B'),
    ('offset_size', 'B'),
    ('length_size', 'B'),
    ('file_consistency_flags', 'B'),

    ('base_address', 'Q'),                  # assume 8 byte addressing
    ('superblock_extension_address', 'Q'),  # assume 8 byte addressing
    ('end_of_file_address', 'Q'),           # assume 8 byte addressing
    ('root_group_address', 'Q'),            # assume 8 byte addressing

    ('superblock_checksum', 'I'),

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
ATTR_MSG_HEADER_V1 = OrderedDict((
    ('version', 'B'),
    ('reserved', 'B'),
    ('name_size', 'H'),
    ('datatype_size', 'H'),
    ('dataspace_size', 'H'),
))
ATTR_MSG_HEADER_V1_SIZE = _structure_size(ATTR_MSG_HEADER_V1)

ATTR_MSG_HEADER_V3 = OrderedDict((
    ('version', 'B'),
    ('flags', 'B'),
    ('name_size', 'H'),
    ('datatype_size', 'H'),
    ('dataspace_size', 'H'),
    ('character_set_encoding', 'B'),
))
ATTR_MSG_HEADER_V3_SIZE = _structure_size(ATTR_MSG_HEADER_V3)

# III.D Disk Format: Level 1D - Local Heaps
LOCAL_HEAP = OrderedDict((
    ('signature', '4s'),
    ('version', 'B'),
    ('reserved', '3s'),
    ('data_segment_size', 'Q'),         # 8 byte size of lengths
    ('offset_to_free_list', 'Q'),       # 8 bytes size of lengths
    ('address_of_data_segment', 'Q'),   # 8 byte addressing
))


# III.E Disk Format: Level 1E - Global Heap
GLOBAL_HEAP_HEADER = OrderedDict((
    ('signature', '4s'),
    ('version', 'B'),
    ('reserved', '3s'),
    ('collection_size', 'Q'),
))
GLOBAL_HEAP_HEADER_SIZE = _structure_size(GLOBAL_HEAP_HEADER)

GLOBAL_HEAP_OBJECT = OrderedDict((
    ('object_index', 'H'),
    ('reference_count', 'H'),
    ('reserved', 'I'),
    ('object_size', 'Q')    # 8 byte addressing
))
GLOBAL_HEAP_OBJECT_SIZE = _structure_size(GLOBAL_HEAP_OBJECT)

# IV.A.1.a Version 1 Data Object Header Prefix
OBJECT_HEADER_V1 = OrderedDict((
    ('version', 'B'),
    ('reserved', 'B'),
    ('total_header_messages', 'H'),
    ('object_reference_count', 'I'),
    ('object_header_size', 'I'),
    ('padding', 'I'),
))

# IV.A.1.b Version 2 Data Object Header Prefix
OBJECT_HEADER_V2 = OrderedDict((
    ('signature', '4s'),
    ('version', 'B'),
    ('flags', 'B'),
    # Access time (optional)
    # Modification time (optional)
    # Change time (optional)
    # Birth time (optional)
    # Maximum # of compact attributes
    # Maximum # of dense attributes
    # Size of Chunk #0

))


# IV.A.2.b The Dataspace Message
DATASPACE_MSG_HEADER_V1 = OrderedDict((
    ('version', 'B'),
    ('dimensionality', 'B'),
    ('flags', 'B'),
    ('reserved_0', 'B'),
    ('reserved_1', 'I'),
))
DATASPACE_MSG_HEADER_V1_SIZE = _structure_size(DATASPACE_MSG_HEADER_V1)

DATASPACE_MSG_HEADER_V2 = OrderedDict((
    ('version', 'B'),
    ('dimensionality', 'B'),
    ('flags', 'B'),
    ('type', 'B'),
))
DATASPACE_MSG_HEADER_V2_SIZE = _structure_size(DATASPACE_MSG_HEADER_V2)

# IV.A.2.d The Datatype Message

DATATYPE_MSG = OrderedDict((
    ('class_and_version', 'B'),
    ('class_bit_field_0', 'B'),
    ('class_bit_field_1', 'B'),
    ('class_bit_field_2', 'B'),
    ('size', 'I'),
))

#
HEADER_MSG_INFO_V1 = OrderedDict((
    ('type', 'H'),
    ('size', 'H'),
    ('flags', 'B'),
    ('reserved', '3s'),
))


HEADER_MSG_INFO_V2 = OrderedDict((
    ('type', 'B'),
    ('size', 'H'),
    ('flags', 'B'),
))


SYMBOL_TABLE_MSG = OrderedDict((
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
SHARED_MSG_TABLE_MSG_TYPE = 0x000F
OBJECT_CONTINUATION_MSG_TYPE = 0x0010
SYMBOL_TABLE_MSG_TYPE = 0x0011
OBJECT_MODIFICATION_TIME_MSG_TYPE = 0x0012
BTREE_K_VALUE_MSG_TYPE = 0x0013
DRIVER_INFO_MSG_TYPE = 0x0014
ATTRIBUTE_INFO_MSG_TYPE = 0x0015
OBJECT_REFERENCE_COUNT_MSG_TYPE = 0x0016
FILE_SPACE_INFO_MSG_TYPE = 0x0018

# Datatype message, datatype classes
DATATYPE_FIXED_POINT = 0
DATATYPE_FLOATING_POINT = 1
DATATYPE_TIME = 2
DATATYPE_STRING = 3
DATATYPE_BITFIELD = 4
DATATYPE_OPAQUE = 5
DATATYPE_COMPOUND = 6
DATATYPE_REFERENCE = 7
DATATYPE_ENUMERATED = 8
DATATYPE_VARIABLE_LENGTH = 9
DATATYPE_ARRAY = 10
