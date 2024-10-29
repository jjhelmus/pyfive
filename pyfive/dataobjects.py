""" HDF5 dataobjects objects.  """

from __future__ import division

from collections import OrderedDict
import struct
import warnings

import numpy as np

from .datatype_msg import DatatypeMessage
from .core import _padded_size, _structure_size
from .core import _unpack_struct_from, _unpack_struct_from_file
from .core import InvalidHDF5File
from .core import Reference
from .core import UNDEFINED_ADDRESS
from .btree import BTreeV1Groups, BTreeV1RawDataChunks
from .btree import BTreeV2GroupNames, BTreeV2GroupOrders
from .btree import GZIP_DEFLATE_FILTER, SHUFFLE_FILTER, FLETCH32_FILTER
from .misc_low_level import Heap, SymbolTable, GlobalHeap, FractalHeap

# these constants happen to have the same value...
UNLIMITED_SIZE = UNDEFINED_ADDRESS


class DataObjects(object):
    """
    HDF5 DataObjects.
    """

    def __init__(self, fh, offset):
        """ initalize. """
        fh.seek(offset)
        version_hint = struct.unpack_from('<B', fh.read(1))[0]
        fh.seek(offset)
        if version_hint == 1:
            msgs, msg_data, header = self._parse_v1_objects(fh)
        elif version_hint == ord('O'):   # first character of v2 signature
            msgs, msg_data, header = self._parse_v2_objects(fh)
        else:
            raise InvalidHDF5File('unknown Data Object Header')

        self.fh = fh
        self.msgs = msgs
        self.msg_data = msg_data
        self.offset = offset
        self._global_heaps = {}
        self._header = header

        # cached attributes
        self._filter_pipeline = None
        self._chunk_params_set = False
        self._chunks = None
        self._chunk_dims = None
        self._chunk_address = None

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
        # See section IV.A.2.m. The Attribute Message for details
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
            dtype = DatatypeMessage(self.msg_data, offset).dtype
        except NotImplementedError:
            warnings.warn(
                'Attribute %s type not implemented, set to None.' % (name, ))
            return name, None
        offset += _padded_size(attr_dict['datatype_size'], padding_multiple)

        # read in the dataspace information
        shape, maxshape = determine_data_shape(self.msg_data, offset)
        items = int(np.prod(shape))
        offset += _padded_size(attr_dict['dataspace_size'], padding_multiple)

        # read in the value(s)
        value = self._attr_value(dtype, self.msg_data, items, offset)

        if shape == ():
            value = value[0]
        else:
            value = value.reshape(shape)
        return name, value

    def _attr_value(self, dtype, buf, count, offset):
        """ Retrieve an HDF5 attribute value from a buffer. """
        if isinstance(dtype, tuple):
            dtype_class = dtype[0]
            value = np.empty(count, dtype=object)
            for i in range(count):
                if dtype_class == 'VLEN_STRING':
                    _, _, character_set = dtype
                    _, vlen_data = self._vlen_size_and_data(buf, offset)
                    if character_set == 0:
                        # ascii character set, return as bytes
                        value[i] = vlen_data
                    else:
                        value[i] = vlen_data.decode('utf-8')
                    offset += 16
                elif dtype_class == 'REFERENCE':
                    address, = struct.unpack_from('<Q', buf, offset=offset)
                    value[i] = Reference(address)
                    offset += 8
                elif dtype_class == "VLEN_SEQUENCE":
                    base_dtype = dtype[1]
                    vlen, vlen_data = self._vlen_size_and_data(buf, offset)
                    value[i] = self._attr_value(base_dtype, vlen_data, vlen, 0)
                    offset += 16
                else:
                    raise NotImplementedError
        else:
            value = np.frombuffer(buf, dtype=dtype, count=count, offset=offset)
        return value

    def _vlen_size_and_data(self, buf, offset):
        """ Extract the length and data of a variables length attr. """
        # offset should be incremented by 16 after calling this method
        vlen_size, = struct.unpack_from('<I', buf, offset=offset)
        # section IV.B
        # Data with a variable-length datatype is stored in the
        # global heap of the HDF5 file. Global heap identifiers are
        # stored in the data object storage.
        gheap_id = _unpack_struct_from(GLOBAL_HEAP_ID, buf, offset+4)
        gheap_address = gheap_id['collection_address']
        if gheap_address not in self._global_heaps:
            # load the global heap and cache the instance
            gheap = GlobalHeap(self.fh, gheap_address)
            self._global_heaps[gheap_address] = gheap
        gheap = self._global_heaps[gheap_address]
        vlen_data = gheap.objects[gheap_id['object_index']]
        return vlen_size, vlen_data

    @property
    def shape(self):
        """ Shape of the dataset. """
        msg = self.find_msg_type(DATASPACE_MSG_TYPE)[0]
        msg_offset = msg['offset_to_message']
        shape, maxshape = determine_data_shape(self.msg_data, msg_offset)
        return shape

    @property
    def maxshape(self):
        """ Maximum Shape of the dataset. (None for unlimited dimension) """
        msg = self.find_msg_type(DATASPACE_MSG_TYPE)[0]
        msg_offset = msg['offset_to_message']
        shape, maxshape = determine_data_shape(self.msg_data, msg_offset)
        return maxshape

    @property
    def fillvalue(self):
        """ Fillvalue of the dataset. """
        msg = self.find_msg_type(FILLVALUE_MSG_TYPE)[0]
        offset = msg['offset_to_message']
        version = struct.unpack_from('<B', self.msg_data, offset)[0]
        if version == 1 or version == 2:
            info = _unpack_struct_from(FILLVAL_MSG_V1V2, self.msg_data, offset)
            offset += FILLVAL_MSG_V1V2_SIZE
            is_defined = info['fillvalue_defined']
        elif version == 3:
            info = _unpack_struct_from(FILLVAL_MSG_V3, self.msg_data, offset)
            offset += FILLVAL_MSG_V3_SIZE
            is_defined = info['flags'] & 2**5
        else:
            raise InvalidHDF5File(
                "Unknown fillvalue msg version:" + str(version))

        if is_defined:
            size = struct.unpack_from('<I', self.msg_data, offset)[0]
            offset += 4
        else:
            size = 0

        if size:
            payload = self.msg_data[offset:offset+size]
            fillvalue = np.frombuffer(payload, self.dtype, count=1)[0]
        else:
            fillvalue = 0
        return fillvalue

    @property
    def dtype(self):
        """ Datatype of the dataset. """
        msg = self.find_msg_type(DATATYPE_MSG_TYPE)[0]
        msg_offset = msg['offset_to_message']
        return DatatypeMessage(self.msg_data, msg_offset).dtype

    @property
    def chunks(self):
        """ Tuple describing the chunk size, None if not chunked. """
        self._get_chunk_params()
        return self._chunks

    @property
    def compression(self):
        """ str describing compression filter, None if no compression. """
        if self._filter_ids is None:
            return None
        if GZIP_DEFLATE_FILTER in self._filter_ids:
            return 'gzip'
        return None

    @property
    def compression_opts(self):
        """ Compression filter options, None is no options/compression. """
        if self._filter_ids is None:
            return None
        if GZIP_DEFLATE_FILTER in self._filter_ids:
            gzip_entry = [d for d in self.filter_pipeline
                          if d['filter_id'] == GZIP_DEFLATE_FILTER][0]
            return gzip_entry['client_data'][0]
        return None

    @property
    def shuffle(self):
        """ Boolean indicator if shuffle filter was applied. """
        if self._filter_ids is None:
            return False
        return SHUFFLE_FILTER in self._filter_ids

    @property
    def fletcher32(self):
        """ Boolean indicator if fletcher32 filter was applied. """
        if self._filter_ids is None:
            return False
        return FLETCH32_FILTER in self._filter_ids

    @property
    def _filter_ids(self):
        """ List of filter id in the filter pipeline, None if no pipeline. """
        if self.filter_pipeline is None:
            return None
        return [d['filter_id'] for d in self.filter_pipeline]

    @property
    def filter_pipeline(self):
        """ Dict describing filter pipeline, None if no pipeline. """
        if self._filter_pipeline is not None:
            return self._filter_pipeline  # use cached value

        filter_msgs = self.find_msg_type(DATA_STORAGE_FILTER_PIPELINE_MSG_TYPE)
        if not filter_msgs:
            self._filter_pipeline = None
            return self._filter_pipeline

        offset = filter_msgs[0]['offset_to_message']
        version, nfilters = struct.unpack_from('<BB', self.msg_data, offset)
        offset += struct.calcsize('<BB')

        filters = []
        if version == 1:
            res0, res1 = struct.unpack_from('<HI', self.msg_data, offset)
            offset += struct.calcsize('<HI')

            for _ in range(nfilters):
                filter_info = _unpack_struct_from(
                    FILTER_PIPELINE_DESCR_V1, self.msg_data, offset)
                offset += FILTER_PIPELINE_DESCR_V1_SIZE

                padded_name_length = _padded_size(filter_info['name_length'], 8)
                fmt = '<' + str(padded_name_length) + 's'
                filter_name = struct.unpack_from(fmt, self.msg_data, offset)[0]
                filter_info['filter_name'] = filter_name
                offset += padded_name_length

                fmt = '<' + str(filter_info['client_data_values']) + 'I'
                client_data = struct.unpack_from(fmt, self.msg_data, offset)
                filter_info['client_data'] = client_data
                offset += 4 * filter_info['client_data_values']

                if filter_info['client_data_values'] % 2:
                    offset += 4  # odd number of client data values padded

                filters.append(filter_info)

        elif version == 2:
            for _ in range(nfilters):
                filter_info = OrderedDict()
                filter_id = struct.unpack_from("<H", self.msg_data, offset)[0]
                offset += 2
                filter_info['filter_id'] = filter_id
                name_length = 0
                if filter_id > 255:
                    # name and name length not encoded for built-in filters
                    name_length = struct.unpack_from('<H', self.msg_data, offset)[0]
                    offset += 2
                flags = struct.unpack_from('<H', self.msg_data, offset)[0]
                offset += 2
                filter_info['optional'] = (flags & 1) > 0
                num_client_values = struct.unpack_from('<H', self.msg_data, offset)[0]
                offset += 2
                name = ""
                if name_length > 0:
                    name = struct.unpack_from("{:d}s".format(name_length), self.msg_data, offset)[0]
                    offset += name_length
                filter_info['name'] = name
                client_values = struct.unpack_from("<{:d}i".format(num_client_values), self.msg_data, offset)
                offset += (4 * num_client_values)
                filter_info['client_data_values'] = client_values

                filters.append(filter_info)
        else:
            raise NotImplementedError("filter pipeline description version > 2 is not supported")
        self._filter_pipeline = filters
        return self._filter_pipeline

    def get_data(self):
        """ Return the data pointed to in the DataObject. """

        # offset and size from data storage message
        msg = self.find_msg_type(DATA_STORAGE_MSG_TYPE)[0]
        msg_offset = msg['offset_to_message']
        version, dims, layout_class, property_offset = (
            self._get_data_message_properties(msg_offset))

        if layout_class == 0:  # compact storage
            raise NotImplementedError("Compact storage")
        elif layout_class == 1:  # contiguous storage
            return self._get_contiguous_data(property_offset)
        if layout_class == 2:  # chunked storage
            return self._get_chunked_data(msg_offset)

    def _get_data_message_properties(self, msg_offset):
        """ Return the message properties of the DataObject. """
        dims, layout_class, property_offset = None, None, None
        version, arg1, arg2 = struct.unpack_from(
            '<BBB', self.msg_data, msg_offset)
        if (version == 1) or (version == 2):
            dims = arg1
            layout_class = arg2
            property_offset = msg_offset
            property_offset += struct.calcsize('<BBB')
            # reserved fields: 1 byte, 1 int
            property_offset += struct.calcsize('<BI')
            # compact storage (layout class 0) not supported:
            assert (layout_class == 1) or (layout_class == 2)
        elif (version == 3) or (version == 4):
            layout_class = arg1
            property_offset = msg_offset
            property_offset += struct.calcsize('<BB')
        assert (version >= 1) and (version <= 4)
        return version, dims, layout_class, property_offset

    def _get_contiguous_data(self, property_offset):
        data_offset, = struct.unpack_from('<Q', self.msg_data, property_offset)

        if data_offset == UNDEFINED_ADDRESS:
            # no storage is backing array, return all zeros
            return np.zeros(self.shape, dtype=self.dtype)

        if not isinstance(self.dtype, tuple):
            # return a memory-map to the stored array with copy-on-write
            return np.memmap(self.fh, dtype=self.dtype, mode='c',
                             offset=data_offset, shape=self.shape, order='C')
        else:
            dtype_class = self.dtype[0]
            if dtype_class == 'REFERENCE':
                size = self.dtype[1]
                if size != 8:
                    raise NotImplementedError('Unsupported Reference type')
                ref_addresses = np.memmap(
                    self.fh, dtype=('<u8'), mode='c', offset=data_offset,
                    shape=self.shape, order='C')
                return np.array([Reference(addr) for addr in ref_addresses])
            else:
                raise NotImplementedError('datatype not implemented')

    def _get_chunked_data(self, offset):
        """ Return data which is chunked. """
        self._get_chunk_params()
        chunk_btree = BTreeV1RawDataChunks(
            self.fh, self._chunk_address, self._chunk_dims)
        return chunk_btree.construct_data_from_chunks(
            self.chunks, self.shape, self.dtype, self.filter_pipeline)

    def _get_chunk_params(self):
        """
        Get and cache chunked data storage parameters.

        This method should be called prior to accessing any _chunk_*
        attributes. Calling this method multiple times is fine, it will not
        re-read the parameters.
        """
        if self._chunk_params_set:  # parameter have already need retrieved
            return
        self._chunk_params_set = True
        msg = self.find_msg_type(DATA_STORAGE_MSG_TYPE)[0]
        offset = msg['offset_to_message']
        version, dims, layout_class, property_offset = (
            self._get_data_message_properties(offset))

        if layout_class != 2:  # not chunked storage
            return

        address = None
        if (version == 1) or (version == 2):
            address, = struct.unpack_from('<Q', self.msg_data, property_offset)
            data_offset = property_offset + struct.calcsize('<Q')
        elif version == 3:
            dims, address = struct.unpack_from(
                '<BQ', self.msg_data, property_offset)
            data_offset = property_offset + struct.calcsize('<BQ')
        assert (version >= 1) and (version <= 3)

        fmt = '<' + 'I' * (dims-1)
        chunk_shape = struct.unpack_from(fmt, self.msg_data, data_offset)
        self._chunks = chunk_shape
        self._chunk_dims = dims
        self._chunk_address = address
        return

    def find_msg_type(self, msg_type):
        """ Return a list of all messages of a given type. """
        return [m for m in self.msgs if m['type'] == msg_type]

    def get_links(self):
        """ Return a dictionary of link_name: offset """
        return dict(self.iter_links())

    def iter_links(self):
        for msg in self.msgs:
            if msg['type'] == SYMBOL_TABLE_MSG_TYPE:
                yield from self._iter_links_from_symbol_tables(msg)
            elif msg['type'] == LINK_MSG_TYPE:
                yield self._get_link_from_link_msg(msg)
            elif msg['type'] == LINK_INFO_MSG_TYPE:
                yield from self._iter_link_from_link_info_msg(msg)

    def _iter_links_from_symbol_tables(self, sym_tbl_msg):
        """ Return a dict of link_name: offset from a symbol table. """
        assert sym_tbl_msg['size'] == 16
        data = _unpack_struct_from(
            SYMBOL_TABLE_MSG, self.msg_data,
            sym_tbl_msg['offset_to_message'])
        yield from self._iter_links_btree_v1(data['btree_address'], data['heap_address'])

    def _iter_links_btree_v1(self, btree_address, heap_address):
        """ Retrieve links from symbol table message. """
        btree = BTreeV1Groups(self.fh, btree_address)
        heap = Heap(self.fh, heap_address)
        for symbol_table_address in btree.symbol_table_addresses():
            table = SymbolTable(self.fh, symbol_table_address)
            table.assign_name(heap)
            yield from table.get_links(heap).items()

    def _get_link_from_link_msg(self, link_msg):
        """ Retrieve link from link message. """
        offset = link_msg['offset_to_message']
        return self._decode_link_msg(self.msg_data, offset)[1]

    @staticmethod
    def _decode_link_msg(data, offset):
        version, flags = struct.unpack_from('<BB', data, offset)
        offset += 2
        assert version == 1

        size_of_length_of_link_name = 2**(flags & 3)
        link_type_field_present = flags & 2**3
        link_name_character_set_field_present = flags & 2**4
        ordered = flags & 2**2

        if link_type_field_present:
            link_type = struct.unpack_from('<B', data, offset)[0]
            offset += 1
        else:
            link_type = 0
        assert link_type in [0,1]

        if ordered:
            creationorder = struct.unpack_from('<Q', data, offset)[0]
            offset += 8
        else:
            creationorder = None

        if link_name_character_set_field_present:
            link_name_character_set = struct.unpack_from('<B', data, offset)[0]
            offset += 1
        else:
            link_name_character_set = 0

        encoding = 'ascii' if link_name_character_set == 0 else 'utf-8'

        name_size_fmt = ["<B", "<H", "<I", "<Q"][flags & 3]
        name_size = struct.unpack_from(name_size_fmt, data, offset)[0]
        offset += size_of_length_of_link_name

        name = data[offset:offset+name_size].decode(encoding)
        offset += name_size

        if link_type == 0:
            # hard link
            address = struct.unpack_from('<Q', data, offset)[0]
        elif link_type == 1:
            # soft link
            length_of_soft_link_value = struct.unpack_from('<H', data, offset)[0]
            offset += 2
            address = data[offset:offset+length_of_soft_link_value].decode(encoding)

        return creationorder, (name, address)

    def _iter_link_from_link_info_msg(self, info_msg):
        """ Retrieve links from link info message. """
        offset = info_msg['offset_to_message']
        data = self._decode_link_info_msg(self.msg_data, offset)

        heap_address = data["heap_address"]
        name_btree_address = data["name_btree_address"]
        order_btree_address = data.get("order_btree_address", None)
        if name_btree_address:
            yield from self._iter_links_btree_v2(name_btree_address, order_btree_address, heap_address)

    def _iter_links_btree_v2(self, name_btree_address, order_btree_address, heap_address):
        """ Retrieve links from symbol table message. """
        heap = FractalHeap(self.fh, heap_address)
        ordered = (order_btree_address is not None)
        if ordered:
            btree = BTreeV2GroupOrders(self.fh, order_btree_address)
        else:
            btree = BTreeV2GroupNames(self.fh, name_btree_address)
        adict = dict()
        for record in btree.iter_records():
            data = heap.get_data(record["heapid"])
            creationorder, item = self._decode_link_msg(data, 0)
            key = creationorder if ordered else item[0]
            adict[key] = item
        for creationorder, value in sorted(adict.items()):
            yield value

    @staticmethod
    def _decode_link_info_msg(data, offset):
        version, flags = struct.unpack_from('<BB', data, offset)
        assert version == 0
        offset += 2
        if flags & 1:
            # creation order present
            offset += 8

        if flags & 2:
            fmt = LINK_INFO_MSG2
        else:
            fmt = LINK_INFO_MSG1
        data = _unpack_struct_from(fmt, data, offset)
        return {k: None if v == UNDEFINED_ADDRESS else v for k, v in data.items()}

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
    offset += 8 * ndims
    # Dimension maximum size follows if header['flags'] bit 0 set
    if header['flags'] & 2**0:
        maxshape = struct.unpack_from('<' + 'Q' * ndims, buf, offset)
        maxshape = tuple((None if d == UNLIMITED_SIZE else d) for d in maxshape)
    else:
        maxshape = dim_sizes
    # Permutation index follows if header['flags'] bit 1 set
    return dim_sizes, maxshape


# HDF5 Structures
# Values for all fields in this document should be treated as unsigned
# integers, unless otherwise noted in the description of a field. Additionally,
# all metadata fields are stored in little-endian byte order.


GLOBAL_HEAP_ID = OrderedDict((
    ('collection_address', 'Q'),  # 8 byte addressing
    ('object_index', 'I'),
))
GLOBAL_HEAP_ID_SIZE = _structure_size(GLOBAL_HEAP_ID)

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


LINK_INFO_MSG1 = OrderedDict((
    ('heap_address', 'Q'),         # 8 byte addressing
    ('name_btree_address', 'Q'),   # 8 bytes addressing
))


LINK_INFO_MSG2 = OrderedDict((
    ('heap_address', 'Q'),         # 8 byte addressing
    ('name_btree_address', 'Q'),   # 8 bytes addressing
    ('order_btree_address', 'Q')   # 8 bytes addressing
))


# IV.A.2.f. The Data Storage - Fill Value Message
FILLVAL_MSG_V1V2 = OrderedDict((
    ('version', 'B'),
    ('space_allocation_time', 'B'),
    ('fillvalue_write_time', 'B'),
    ('fillvalue_defined', 'B'),
))
FILLVAL_MSG_V1V2_SIZE = _structure_size(FILLVAL_MSG_V1V2)

FILLVAL_MSG_V3 = OrderedDict((
    ('version', 'B'),
    ('flags', 'B'),
))
FILLVAL_MSG_V3_SIZE = _structure_size(FILLVAL_MSG_V3)


# IV.A.2.l The Data Storage - Filter Pipeline message
FILTER_PIPELINE_DESCR_V1 = OrderedDict((
    ('filter_id', 'H'),
    ('name_length', 'H'),
    ('flags', 'H'),
    ('client_data_values', 'H'),
))
FILTER_PIPELINE_DESCR_V1_SIZE = _structure_size(FILTER_PIPELINE_DESCR_V1)

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
