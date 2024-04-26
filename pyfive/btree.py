""" HDF5 B-Trees and contents. """

from collections import OrderedDict
import struct
import zlib

import numpy as np

from .core import _padded_size
from .core import _unpack_struct_from_file
from .core import Reference


class AbstractBTree(object):
    B_LINK_NODE = None
    NODE_TYPE = None

    def __init__(self, fh, offset):
        """ initalize. """
        self.fh = fh
        self.offset = offset
        self.depth = None
        self.all_nodes = {}

        self._read_root_node()
        self._read_children()

    def _read_children(self):
        # Leaf nodes: level 0
        # Root node: level "depth"
        node_level = self.depth
        for node_level in range(self.depth, 0, -1):
            for parent_node in self.all_nodes[node_level]:
                for child_addr in parent_node['addresses']:
                    child_node = self._read_node(child_addr, node_level-1)
                    self._add_node(child_node)

    def _read_root_node(self):
        root_node = self._read_node(self.offset, None)
        self._add_node(root_node)
        self.depth = root_node['node_level']

    def _add_node(self, node):
        node_level = node['node_level']
        if node_level in self.all_nodes:
            self.all_nodes[node_level].append(node)
        else:
            self.all_nodes[node_level] = [node]

    def _read_node(self, offset, node_level):
        """ Return a single node in the B-Tree located at a given offset. """
        node = self._read_node_header(offset, node_level)
        node['keys'] = []
        node['addresses'] = []
        return node

    def _read_node_header(self, offset):
        """ Return a single node header in the b-tree located at a give offset. """
        raise NotImplementedError


class BTreeV1(AbstractBTree):
    """
    HDF5 version 1 B-Tree.
    """

    # III.A.1. Disk Format: Level 1A1 - Version 1 B-trees
    B_LINK_NODE = OrderedDict((
        ('signature', '4s'),

        ('node_type', 'B'),
        ('node_level', 'B'),
        ('entries_used', 'H'),

        ('left_sibling', 'Q'),     # 8 byte addressing
        ('right_sibling', 'Q'),    # 8 byte addressing
    ))

    def _read_node_header(self, offset, node_level):
        """ Return a single node header in the b-tree located at a give offset. """
        self.fh.seek(offset)
        node = _unpack_struct_from_file(self.B_LINK_NODE, self.fh)
        assert node['signature'] == b'TREE'
        assert node['node_type'] == self.NODE_TYPE
        if node_level is not None:
            assert node["node_level"] == node_level
        return node


class BTreeV1Groups(BTreeV1):
    """
    HDF5 version 1 B-Tree storing group nodes (type 0).
    """
    NODE_TYPE = 0

    def _read_node(self, offset, node_level):
        """ Return a single node in the B-Tree located at a given offset. """
        node = self._read_node_header(offset, node_level)
        keys = []
        addresses = []
        for _ in range(node['entries_used']):
            key = struct.unpack('<Q', self.fh.read(8))[0]
            address = struct.unpack('<Q', self.fh.read(8))[0]
            keys.append(key)
            addresses.append(address)
        # N+1 key
        keys.append(struct.unpack('<Q', self.fh.read(8))[0])
        node['keys'] = keys
        node['addresses'] = addresses
        return node

    def symbol_table_addresses(self):
        """ Return a list of all symbol table address. """
        all_address = []
        for node in self.all_nodes[0]:
            all_address.extend(node['addresses'])
        return all_address


class BTreeV1RawDataChunks(BTreeV1):
    """
    HDF5 version 1 B-Tree storing raw data chunk nodes (type 1).
    """
    NODE_TYPE = 1

    def __init__(self, fh, offset, dims):
        """ initalize. """
        self.dims = dims
        super().__init__(fh, offset)

    def _read_node(self, offset, node_level):
        """ Return a single node in the b-tree located at a give offset. """
        node = self._read_node_header(offset, node_level)
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

    def construct_data_from_chunks(
            self, chunk_shape, data_shape, dtype, filter_pipeline):
        """ Build a complete data array from chunks. """
        if isinstance(dtype, tuple):
            true_dtype = tuple(dtype)
            dtype_class = dtype[0]
            if dtype_class == 'REFERENCE':
                size = dtype[1]
                if size != 8:
                    raise NotImplementedError('Unsupported Reference type')
                dtype = '<u8'
            else:
                raise NotImplementedError('datatype not implemented')
        else:
            true_dtype = None

        # create array to store data
        shape = [_padded_size(i, j) for i, j in zip(data_shape, chunk_shape)]
        data = np.zeros(shape, dtype=dtype)

        # loop over chunks reading each into the full data array
        count = np.prod(chunk_shape)
        itemsize = np.dtype(dtype).itemsize
        chunk_buffer_size = count * itemsize
        for node in self.all_nodes[0]:
            for node_key, addr in zip(node['keys'], node['addresses']):
                self.fh.seek(addr)
                if filter_pipeline is None:
                    chunk_buffer = self.fh.read(chunk_buffer_size)
                else:
                    chunk_buffer = self.fh.read(node_key['chunk_size'])
                    filter_mask = node_key['filter_mask']
                    chunk_buffer = self._filter_chunk(
                        chunk_buffer, filter_mask, filter_pipeline, itemsize)

                chunk_data = np.frombuffer(chunk_buffer, dtype=dtype)
                start = node_key['chunk_offset'][:-1]
                region = [slice(i, i+j) for i, j in zip(start, chunk_shape)]
                data[tuple(region)] = chunk_data.reshape(chunk_shape)

        if isinstance(true_dtype, tuple):
            if dtype_class == 'REFERENCE':
                to_reference = np.vectorize(Reference)
                data = to_reference(data)
            else:
                raise NotImplementedError('datatype not implemented')

        non_padded_region = tuple([slice(i) for i in data_shape])
        return data[non_padded_region]

    @classmethod
    def _filter_chunk(cls, chunk_buffer, filter_mask, filter_pipeline, itemsize):
        """ Apply decompression filters to a chunk of data. """
        num_filters = len(filter_pipeline)
        for i, pipeline_entry in enumerate(filter_pipeline[::-1]):

            # A filter is skipped is the bit corresponding to its index in the
            # pipeline is set in filter_mask
            filter_index = num_filters - i - 1   # 0 to num_filters - 1
            if filter_mask & (1 << filter_index):
                continue

            filter_id = pipeline_entry['filter_id']
            if filter_id == GZIP_DEFLATE_FILTER:
                chunk_buffer = zlib.decompress(chunk_buffer)
            elif filter_id == SHUFFLE_FILTER:
                buffer_size = len(chunk_buffer)
                unshuffled_buffer = bytearray(buffer_size)
                step = buffer_size // itemsize
                for j in range(itemsize):
                    start = j * step
                    end = (j+1) * step
                    unshuffled_buffer[j::itemsize] = chunk_buffer[start:end]
                chunk_buffer = unshuffled_buffer
            elif filter_id == FLETCH32_FILTER:
                cls._verify_fletcher32(chunk_buffer)
                # strip off 4-byte checksum from end of buffer
                chunk_buffer = chunk_buffer[:-4]
            else:
                raise NotImplementedError(
                    "Filter with id: %i import supported" % (filter_id))
        return chunk_buffer

    @staticmethod
    def _verify_fletcher32(chunk_buffer):
        """ Verify a chunk with a fletcher32 checksum. """
        # calculate checksums
        if len(chunk_buffer) % 2:
            arr = np.frombuffer(chunk_buffer[:-4]+b'\x00', '<u2')
        else:
            arr = np.frombuffer(chunk_buffer[:-4], '<u2')
        sum1 = sum2 = 0
        for i in arr:
            sum1 = (sum1 + i) % 65535
            sum2 = (sum2 + sum1) % 65535

        # extract stored checksums
        ref_sum1, ref_sum2 = np.frombuffer(chunk_buffer[-4:], '>u2')
        ref_sum1 = ref_sum1 % 65535
        ref_sum2 = ref_sum2 % 65535

        # compare
        if sum1 != ref_sum1 or sum2 != ref_sum2:
            raise ValueError("fletcher32 checksum invalid")
        return True


class BTreeV2(AbstractBTree):
    """
    HDF5 version 2 B-Tree.
    """

    # III.A.2. Disk Format: Level 1A2 - Version 2 B-trees
    B_TREE_HEADER = OrderedDict((
        ('signature', '4s'),

        ('version', 'B'),
        ('node_type', 'B'),
        ('node_size', 'I'),
        ('record_size', 'H'),
        ('depth', 'H'),
        ('split_percent', 'B'),
        ('merge_percent', 'B'),

        ('root_address', 'Q'),     # 8 byte addressing
        ('root_nrecords', 'H'),
        ('total_nrecords', 'Q'),     # 8 byte addressing
    ))

    B_LINK_NODE = OrderedDict((
        ('signature', '4s'),

        ('version', 'B'),
        ('node_type', 'B'),
    ))

    def _read_root_node(self):
        h = self._read_tree_header(self.offset)
        self.address_formats = self._calculate_address_formats(h)
        self.header = h
        self.depth = h["depth"]

        address = (h["root_address"], h["root_nrecords"], h["total_nrecords"])
        root_node = self._read_node(address, self.depth)
        self._add_node(root_node)

    def _read_tree_header(self, offset):
        self.fh.seek(self.offset)
        header = _unpack_struct_from_file(self.B_TREE_HEADER, self.fh)
        assert header['signature'] == b'BTHD'
        assert header['node_type'] == self.NODE_TYPE
        return header

    def _calculate_address_formats(self, header):
        node_size = header["node_size"]
        record_size = header["record_size"]

        nrecords_max = 0
        ntotalrecords_max = 0
        address_formats = []
        for node_level in range(header["depth"]+1):
            offset_fmt = ""
            num1_fmt = ""
            num2_fmt = ""
            if node_level == 0:  # leaf node
                offset_size = 0
                num1_size = 0
                num2_size = 0
            elif node_level == 1:  # internal node (twig node)
                offset_size = 8
                offset_fmt = "<Q"
                num1_size = self._required_bytes(nrecords_max)
                num1_fmt = "<{}s".format(num1_size)
                num2_size = 0
            else:  # internal node
                offset_size = 8
                offset_fmt = "<Q"
                num1_size = self._required_bytes(nrecords_max)
                num1_fmt = "<{}s".format(num1_size)
                num2_size = self._required_bytes(ntotalrecords_max)
                num2_fmt = "<{}s".format(num2_size)
            address_formats.append((
                offset_size, num1_size, num2_size,
                offset_fmt, num1_fmt, num2_fmt))
            if node_level < header["depth"]:
                addr_size = offset_size + num1_size + num2_size
                nrecords_max = self._nrecords_max(node_size, record_size, addr_size)
                if ntotalrecords_max:
                    ntotalrecords_max *= nrecords_max
                else:
                    ntotalrecords_max = nrecords_max

        return address_formats

    @staticmethod
    def _nrecords_max(node_size, record_size, addr_size):
        """ Calculate the maximal records a node can contain. """
        # node_size = overhead + nrecords_max*record_size + (nrecords_max+1)*addr_size
        #
        # overhead = size(B_LINK_NODE) + 4 (checksum)
        #
        # Leaf node (node_level = 0)
        #   addr_size = 0
        # Internal node (node_level = 1)
        #   addr_size = offset_size + num1_size
        # Internal node (node_level > 1)
        #   addr_size = offset_size + num1_size + num2_size
        return (node_size - 10 - addr_size)//(record_size + addr_size)

    @staticmethod
    def _required_bytes(integer):
        """ Calculate the minimal required bytes to contain an integer. """
        return (max(integer.bit_length(), 1) + 7) // 8

    def _read_node(self, address, node_level):
        """ Return a single node in the B-Tree located at a given offset. """
        offset, nrecords, ntotalrecords = address
        node = self._read_node_header(offset, node_level)

        record_size = self.header['record_size']

        keys = []
        for _ in range(nrecords):
            record = self.fh.read(record_size)
            keys.append(self._parse_record(record))

        addresses = []
        fmts = self.address_formats[node_level]
        if fmts[0]:
            offset_size, num1_size, num2_size, offset_fmt, num1_fmt, num2_fmt = fmts
            for _ in range(nrecords+1):
                offset = struct.unpack(offset_fmt, self.fh.read(offset_size))[0]
                num1 = struct.unpack(num1_fmt, self.fh.read(num1_size))[0]
                num1 = int.from_bytes(num1, byteorder="little", signed=False)
                if num2_size:
                    num2 = struct.unpack(num2_fmt, self.fh.read(num2_size))[0]
                    num2 = int.from_bytes(num2, byteorder="little", signed=False)
                else:
                    num2 = num1
                addresses.append((offset, num1, num2))

        node['keys'] = keys
        node['addresses'] = addresses
        return node

    def _read_node_header(self, offset, node_level):
        """ Return a single node header in the b-tree located at a give offset. """
        self.fh.seek(offset)
        node = _unpack_struct_from_file(self.B_LINK_NODE, self.fh)
        assert node['node_type'] == self.NODE_TYPE
        if node_level:
            # Internal node (has children)
            assert node['signature'] == b'BTIN'
        else:
            # Leaf node (has no children)
            assert node['signature'] == b'BTLF'
        node["node_level"] = node_level
        return node

    def iter_records(self):
        """ Iterate over all records. """
        for nodelist in self.all_nodes.values():
            for node in nodelist:
                yield from node["keys"]

    def _parse_record(self, record):
        raise NotImplementedError


class BTreeV2GroupNames(BTreeV2):
    """
    HDF5 version 2 B-Tree storing group names (type 5).
    """
    NODE_TYPE = 5

    def _parse_record(self, record):
        namehash = struct.unpack_from("<I", record, 0)[0]
        return {'namehash': namehash, 'heapid':record[4:4+7]}


class BTreeV2GroupOrders(BTreeV2):
    """
    HDF5 version 2 B-Tree storing group creation orders (type 6).
    """
    NODE_TYPE = 6

    def _parse_record(self, record):
        creationorder = struct.unpack_from("<Q", record, 0)[0]
        return {'creationorder': creationorder, 'heapid':record[8:8+7]}


# IV.A.2.l The Data Storage - Filter Pipeline message
RESERVED_FILTER = 0
GZIP_DEFLATE_FILTER = 1
SHUFFLE_FILTER = 2
FLETCH32_FILTER = 3
SZIP_FILTER = 4
NBIT_FILTER = 5
SCALEOFFSET_FILTER = 6
