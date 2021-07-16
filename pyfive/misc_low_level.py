""" Misc low-level representation of HDF5 objects. """

import struct
from math import log2
from collections import OrderedDict

from .core import _padded_size
from .core import _structure_size
from .core import _unpack_struct_from
from .core import _unpack_struct_from_file
from .core import _unpack_integer
from .core import InvalidHDF5File
from .core import UNDEFINED_ADDRESS


class SuperBlock(object):
    """
    HDF5 Superblock.
    """

    def __init__(self, fh, offset):
        """ initalize. """

        fh.seek(offset)
        version_hint = struct.unpack_from('<B', fh.read(9), 8)[0]
        fh.seek(offset)
        if version_hint == 0:
            contents = _unpack_struct_from_file(SUPERBLOCK_V0, fh)
            self._end_of_sblock = offset + SUPERBLOCK_V0_SIZE
        elif version_hint == 2 or version_hint == 3:
            contents = _unpack_struct_from_file(SUPERBLOCK_V2_V3, fh)
            self._end_of_sblock = offset + SUPERBLOCK_V2_V3_SIZE
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
        self._root_symbol_table = None
        self._fh = fh

    @property
    def offset_to_dataobjects(self):
        """ The offset to the data objects collection for the superblock. """
        if self.version == 0:
            sym_table = SymbolTable(self._fh, self._end_of_sblock, root=True)
            self._root_symbol_table = sym_table
            return sym_table.group_offset
        elif self.version == 2 or self.version == 3:
            return self._contents['root_group_address']
        else:
            raise NotImplementedError


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

    def get_links(self, heap):
        """ Return a dictionary of links (dataset/group) and offsets. """
        links = {}
        for e in self.entries:
            if e['cache_type'] in [0,1]:
                links[e['link_name']] = e['object_header_address']
            elif e['cache_type'] == 2:
                offset = struct.unpack('<4I', e['scratch'])[0]
                links[e['link_name']] = heap.get_object_name(offset).decode('utf-8')
        return links


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


class FractalHeap(object):
    """
    HDF5 Fractal Heap.
    """

    def __init__(self, fh, offset):

        fh.seek(offset)
        header = _unpack_struct_from_file(FRACTAL_HEAP_HEADER, fh)
        assert header['signature'] == b'FRHP'
        assert header['version'] == 0

        if header['filter_info_size']:
            raise NotImplementedError

        if header["btree_address_huge_objects"] == UNDEFINED_ADDRESS:
            header["btree_address_huge_objects"] = None
        else:
            raise NotImplementedError

        if header["root_block_address"] == UNDEFINED_ADDRESS:
            header["root_block_address"] = None

        nbits = header["log2_maximum_heap_size"]
        block_offset_size = self._min_size_nbits(nbits)
        h = OrderedDict((
            ('signature', '4s'),
            ('version', 'B'),
            ('heap_header_adddress', 'Q'),
            ('block_offset', '{}s'.format(block_offset_size))
        ))
        self.indirect_block_header = h.copy()
        self.indirect_block_header_size = _structure_size(h)
        if (header["flags"] & 2) == 2:
            h['checksum'] = 'I'
        self.direct_block_header = h
        self.direct_block_header_size = _structure_size(h)

        maximum_dblock_size = header['maximum_direct_block_size']
        nbits = header['log2_maximum_heap_size']
        self._managed_object_offset_size = self._min_size_nbits(nbits)
        value = min(maximum_dblock_size, header['max_managed_object_size'])
        self._managed_object_length_size = self._min_size_integer(value)

        start_block_size = header['starting_block_size']
        table_width = header['table_width']
        if not start_block_size:
            assert NotImplementedError

        log2_maximum_dblock_size = int(log2(maximum_dblock_size))
        assert 2**log2_maximum_dblock_size == maximum_dblock_size
        log2_start_block_size = int(log2(start_block_size))
        assert 2**log2_start_block_size == start_block_size
        self._max_direct_nrows = log2_maximum_dblock_size - log2_start_block_size + 2

        log2_table_width = int(log2(table_width))
        assert 2**log2_table_width == table_width
        self._indirect_nrows_sub = log2_table_width + log2_start_block_size - 1

        self.header = header
        self.nobjects = header["managed_object_count"] + header["huge_object_count"] + header["tiny_object_count"]

        managed = []
        root_address = header["root_block_address"]
        if root_address:
            nrows = header["indirect_current_rows_count"]
            if nrows:
                for data in self._iter_indirect_block(fh, root_address, nrows):
                    managed.append(data)
            else:
                data = self._read_direct_block(fh, root_address, start_block_size)
                managed.append(data)
        self.managed = b"".join(managed)

    def _read_direct_block(self, fh, offset, block_size):
        fh.seek(offset)
        data = fh.read(block_size)
        header = _unpack_struct_from(self.direct_block_header, data)
        header["signature"] == b"FHDB"
        return data

    def get_data(self, heapid):
        firstbyte = heapid[0]
        reserved = firstbyte & 15  # bit 0-3
        idtype = (firstbyte >> 4) & 3  # bit 4-5
        version = firstbyte >> 6  # bit 6-7
        data_offset = 1
        if idtype == 0: # managed
            assert version == 0
            nbytes = self._managed_object_offset_size
            offset = _unpack_integer(nbytes, heapid, data_offset)
            data_offset += nbytes

            nbytes = self._managed_object_length_size
            size = _unpack_integer(nbytes, heapid, data_offset)

            return self.managed[offset:offset+size]
        elif idtype == 1: # tiny
            raise NotImplementedError
        elif idtype == 2: # huge
            raise NotImplementedError
        else:
            raise NotImplementedError

    def _min_size_integer(self, integer):
        """ Calculate the minimal required bytes to contain an integer. """
        return self._min_size_nbits(integer.bit_length())

    @staticmethod
    def _min_size_nbits(nbits):
        """ Calculate the minimal required bytes to contain a number of bits. """
        return nbits // 8 + min(nbits % 8, 1)

    def _read_integral(self, fh, nbytes):
        num = fh.read(nbytes)
        num = struct.unpack("{}s".format(nbytes))[0]
        return int.from_bytes(num, byteorder="little", signed=False)

    def _iter_indirect_block(self, fh, offset, nrows):
        fh.seek(offset)
        header = _unpack_struct_from_file(self.indirect_block_header, fh)
        header["signature"] == b"FHIB"
        header["block_offset"] = int.from_bytes(header["block_offset"], byteorder="little", signed=False)
        ndirect, nindirect = self._indirect_info(nrows)

        direct_blocks = list()
        for i in range(ndirect):
            address = struct.unpack('<Q', fh.read(8))[0]
            if address == UNDEFINED_ADDRESS:
                break
            block_size = self._calc_block_size(i)
            direct_blocks.append((address, block_size))

        indirect_blocks = list()
        for i in range(ndirect, ndirect+nindirect):
            address = struct.unpack('<Q', fh.read(8))[0]
            if address == UNDEFINED_ADDRESS:
                break
            block_size = self._calc_block_size(i)
            nrows = self._iblock_nrows_from_block_size(block_size)
            indirect_blocks.append((address, nrows))

        for address, block_size in direct_blocks:
            obj = self._read_direct_block(fh, address, block_size)
            yield obj

        for address, nrows in indirect_blocks:
            for obj in self._iter_indirect_block(fh, address, nrows):
                yield obj

    def _calc_block_size(self, iblock):
        row = iblock//self.header["table_width"]
        return 2**max(row-1, 0) * self.header['starting_block_size']

    def _iblock_nrows_from_block_size(self, block_size):
        log2_block_size = int(log2(block_size))
        assert 2**log2_block_size == block_size
        return log2_block_size - self._indirect_nrows_sub

    def _indirect_info(self, nrows):
        table_width = self.header['table_width']
        nobjects = nrows * table_width
        ndirect_max = self._max_direct_nrows * table_width
        if nrows <= ndirect_max:
            ndirect = nobjects
            nindirect = 0
        else:
            ndirect = ndirect_max
            nindirect = nobjects - ndirect_max
        return ndirect, nindirect


FORMAT_SIGNATURE = b'\211HDF\r\n\032\n'

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
SUPERBLOCK_V0_SIZE = _structure_size(SUPERBLOCK_V0)

# Version 2 and 3 SUPERBLOCK
SUPERBLOCK_V2_V3 = OrderedDict((
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
SUPERBLOCK_V2_V3_SIZE = _structure_size(SUPERBLOCK_V2_V3)


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

# III.G. Disk Format: Level 1G - Fractal Heap
FRACTAL_HEAP_HEADER = OrderedDict((
    ('signature', '4s'),
    ('version', 'B'),

    ('object_index_size', 'H'),
    ('filter_info_size', 'H'),
    ('flags', 'B'),

    ('max_managed_object_size', 'I'),
    ('next_huge_object_index', 'Q'),       # 8 byte addressing
    ('btree_address_huge_objects', 'Q'),   # 8 byte addressing

    ('managed_freespace_size', 'Q'),       # 8 byte addressing
    ('freespace_manager_address', 'Q'),    # 8 byte addressing
    ('managed_space_size', 'Q'),           # 8 byte addressing
    ('managed_alloc_size', 'Q'),           # 8 byte addressing
    ('next_directblock_iterator_address', 'Q'), # 8 byte addressing

    ('managed_object_count', 'Q'),         # 8 byte addressing
    ('huge_objects_total_size', 'Q'),      # 8 byte addressing
    ('huge_object_count', 'Q'),            # 8 byte addressing
    ('tiny_objects_total_size', 'Q'),      # 8 byte addressing
    ('tiny_object_count', 'Q'),            # 8 byte addressing

    ('table_width', 'H'),
    ('starting_block_size', 'Q'),          # 8 byte addressing
    ('maximum_direct_block_size', 'Q'),    # 8 byte addressing
    ('log2_maximum_heap_size', 'H'),
    ('indirect_starting_rows_count', 'H'),
    ('root_block_address', 'Q'),           # 8 byte addressing
    ('indirect_current_rows_count', 'H'),
))
