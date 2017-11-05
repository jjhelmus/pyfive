""" Misc low-level representation of HDF5 objects. """

import struct
from collections import OrderedDict

from .core import _padded_size, _structure_size
from .core import _unpack_struct_from, _unpack_struct_from_file
from .core import InvalidHDF5File


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
