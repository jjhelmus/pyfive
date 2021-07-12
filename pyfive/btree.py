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


# IV.A.2.l The Data Storage - Filter Pipeline message
RESERVED_FILTER = 0
GZIP_DEFLATE_FILTER = 1
SHUFFLE_FILTER = 2
FLETCH32_FILTER = 3
SZIP_FILTER = 4
NBIT_FILTER = 5
SCALEOFFSET_FILTER = 6
