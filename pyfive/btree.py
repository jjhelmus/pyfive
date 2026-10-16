"""HDF5 B-Trees and contents."""

import struct
import zlib

import numpy as np
from numcodecs import Fletcher32

from .core import _unpack_struct_from_file
from .core import _unpack_struct_from
from .core import _unpack_integer


_FLETCHER32_CODEC = Fletcher32()


class AbstractBTree(object):
    B_LINK_NODE = None
    NODE_TYPE = None

    def __init__(self, fh, offset):
        """initalize."""
        self.fh = fh
        self.offset = offset
        self.depth = None
        self.all_nodes = {}
        self.last_offset = offset

        self._read_root_node()
        self._read_children()

    def _read_children(self):
        # Leaf nodes: level 0
        # Root node: level "depth"
        node_level = self.depth
        for node_level in range(self.depth, 0, -1):
            for parent_node in self.all_nodes[node_level]:
                for child_addr in parent_node["addresses"]:
                    child_node = self._read_node(child_addr, node_level - 1)
                    self._add_node(child_node)

    def _read_root_node(self):
        root_node = self._read_node(self.offset, None)
        self._add_node(root_node)
        self.depth = root_node["node_level"]

    def _add_node(self, node):
        node_level = node["node_level"]
        if node_level in self.all_nodes:
            self.all_nodes[node_level].append(node)
        else:
            self.all_nodes[node_level] = [node]

    def _read_node(self, offset, node_level):
        """Return a single node in the B-Tree located at a given offset."""
        node = self._read_node_header(offset, node_level)
        node["keys"] = []
        node["addresses"] = []
        self.last_offset = max(offset, self.last_offset)
        return node

    def _read_node_header(self, offset):
        """Return a single node header in the b-tree located at a give offset."""
        raise NotImplementedError


class BTreeV1(AbstractBTree):
    """
    HDF5 version 1 B-Tree.
    """

    # III.A.1. Disk Format: Level 1A1 - Version 1 B-trees
    B_LINK_NODE = dict(
        (
            ("signature", "4s"),
            ("node_type", "B"),
            ("node_level", "B"),
            ("entries_used", "H"),
            ("left_sibling", "Q"),  # 8 byte addressing
            ("right_sibling", "Q"),  # 8 byte addressing
        )
    )  # type: ignore[assignment]

    def _read_node_header(self, offset, node_level):
        """Return a single node header in the b-tree located at a give offset."""
        self.fh.seek(offset)
        node = _unpack_struct_from_file(self.B_LINK_NODE, self.fh)
        assert node["signature"] == b"TREE"
        assert node["node_type"] == self.NODE_TYPE
        if node_level is not None:
            assert node["node_level"] == node_level
        return node


class BTreeV1Groups(BTreeV1):
    """
    HDF5 version 1 B-Tree storing group nodes (type 0).
    """

    NODE_TYPE = 0  # type: ignore[assignment]

    def _read_node(self, offset, node_level):
        """Return a single node in the B-Tree located at a given offset."""
        node = self._read_node_header(offset, node_level)
        keys = []
        addresses = []
        for _ in range(node["entries_used"]):
            key = struct.unpack("<Q", self.fh.read(8))[0]
            address = struct.unpack("<Q", self.fh.read(8))[0]
            keys.append(key)
            addresses.append(address)
        # N+1 key
        keys.append(struct.unpack("<Q", self.fh.read(8))[0])
        node["keys"] = keys
        node["addresses"] = addresses
        return node

    def symbol_table_addresses(self):
        """Return a list of all symbol table address."""
        all_address = []
        for node in self.all_nodes[0]:
            all_address.extend(node["addresses"])
        return all_address


class BTreeV1RawDataChunks(BTreeV1):
    """
    HDF5 version 1 B-Tree storing raw data chunk nodes (type 1).
    """

    NODE_TYPE = 1  # type: ignore[assignment]

    def __init__(self, fh, offset, dims, fetch_fn=None):
        """initalize."""
        self.dims = dims
        self._fetch_fn = fetch_fn
        super().__init__(fh, offset)

    def _read_children(self):
        """Read children; fetch leaf nodes via fetch_fn when provided."""
        if self._fetch_fn is None:
            super()._read_children()
            return

        # Leaf-level root node: already read in _read_root_node.
        if self.depth == 0:
            return

        # Traverse internal levels sequentially; each level depends on the prior.
        for node_level in range(self.depth, 0, -1):
            for parent_node in self.all_nodes[node_level]:
                for child_addr in parent_node["addresses"]:
                    if node_level - 1 > 0:
                        child_node = self._read_node(child_addr, node_level - 1)
                        self._add_node(child_node)

        # Collect leaf addresses from the lowest internal level.
        leaf_addresses = []
        for node in self.all_nodes.get(1, []):
            leaf_addresses.extend(node["addresses"])

        if not leaf_addresses:
            return

        # Leaf nodes can differ in entries_used, so fetch just headers first,
        # then group addresses by exact node size.
        header_size = struct.calcsize("<" + "".join(self.B_LINK_NODE.values()))
        header_buffers = self._fetch_fn(leaf_addresses, header_size)

        size_to_addresses = dict()
        entry_size = 8 + self.dims * 8 + 8
        for addr, header in zip(leaf_addresses, header_buffers):
            entries_used = struct.unpack_from("<H", header, 6)[0]
            node_size = header_size + entries_used * entry_size
            size_to_addresses.setdefault(node_size, []).append(addr)

        for node_size, addresses in size_to_addresses.items():
            raw_buffers = self._fetch_fn(addresses, node_size)
            for addr, raw in zip(addresses, raw_buffers):
                node = self._parse_node_from_buffer(raw, addr, node_level=0)
                self._add_node(node)

    def _read_node(self, offset, node_level):
        """Return a single node in the b-tree located at a give offset."""
        node = self._read_node_header(offset, node_level)

        keys = []
        addresses = []
        n_entries = node["entries_used"]
        entry_size = 8 + self.dims * 8 + 8
        read_size = n_entries * entry_size
        node_data = self.fh.read(read_size)
        mem_view = memoryview(node_data)
        offset_fmt = f"<{self.dims}Q"
        offset_size = self.dims * 8

        pos = 0
        for _ in range(n_entries):
            # chunk_size, filter_mask
            chunk_size, filter_mask = struct.unpack_from("<II", mem_view, pos)
            pos += 8
            # chunk_offset dims × uint64
            chunk_offset = struct.unpack_from(offset_fmt, mem_view, pos)
            pos += offset_size
            # address
            (chunk_address,) = struct.unpack_from("<Q", mem_view, pos)
            pos += 8

            keys.append(
                dict(
                    (
                        ("chunk_size", chunk_size),
                        ("filter_mask", filter_mask),
                        ("chunk_offset", chunk_offset),
                    )
                )
            )
            addresses.append(chunk_address)
        node["keys"] = keys
        node["addresses"] = addresses
        self.last_offset = max(offset, self.last_offset)
        return node

    def _leaf_node_size(self):
        """Compute the byte size of a leaf node by peeking at the first one."""
        header_size = struct.calcsize("<" + "".join(self.B_LINK_NODE.values()))
        entry_size = 8 + 8 * self.dims + 8

        first_leaf_addr = self.all_nodes[1][0]["addresses"][0]
        self.fh.seek(first_leaf_addr)
        header_bytes = self.fh.read(header_size)
        entries_used = struct.unpack_from("<H", header_bytes, 6)[0]
        return header_size + entries_used * entry_size

    def _parse_node_from_buffer(self, raw, addr, node_level):
        """Parse a raw-data-chunk b-tree node from an in-memory bytes buffer."""
        node = _unpack_struct_from(self.B_LINK_NODE, raw)
        assert node["signature"] == b"TREE"
        assert node["node_type"] == self.NODE_TYPE
        assert node["node_level"] == node_level

        keys = []
        addresses = []
        mv = memoryview(raw)
        offset_fmt = f"<{self.dims}Q"
        offset_fmt_size = self.dims * 8
        cursor = struct.calcsize("<" + "".join(self.B_LINK_NODE.values()))

        for _ in range(node["entries_used"]):
            chunk_size, filter_mask = struct.unpack_from("<II", mv, cursor)
            cursor += 8
            chunk_offset = struct.unpack_from(offset_fmt, mv, cursor)
            cursor += offset_fmt_size
            chunk_address = struct.unpack_from("<Q", mv, cursor)[0]
            cursor += 8

            keys.append(
                dict(
                    (
                        ("chunk_size", chunk_size),
                        ("filter_mask", filter_mask),
                        ("chunk_offset", chunk_offset),
                    )
                )
            )
            addresses.append(chunk_address)

        node["keys"] = keys
        node["addresses"] = addresses
        self.last_offset = max(addr, self.last_offset)
        return node

    @classmethod
    def _filter_chunk(cls, chunk_buffer, filter_mask, filter_pipeline, itemsize):
        """Apply decompression filters to a chunk of data."""
        num_filters = len(filter_pipeline)
        for filter_index in range(num_filters - 1, -1, -1):
            pipeline_entry = filter_pipeline[filter_index]

            # A filter is skipped if the bit corresponding to its index in the
            # pipeline is set in filter_mask.
            if filter_mask & (1 << filter_index):
                continue

            filter_id = pipeline_entry["filter_id"]
            if filter_id == GZIP_DEFLATE_FILTER:
                chunk_buffer = zlib.decompress(chunk_buffer)
            elif filter_id == SHUFFLE_FILTER:
                chunk_buffer = cls._unshuffle(chunk_buffer, itemsize)
            elif filter_id == FLETCH32_FILTER:
                cls._verify_fletcher32(chunk_buffer)
                # strip off 4-byte checksum from end of buffer
                chunk_buffer = chunk_buffer[:-4]
            elif filter_id == LZF_FILTER:
                try:
                    import lzf
                except ImportError as e:
                    raise ModuleNotFoundError(
                        "LZF codec requires optional package 'python-neo-lzf'."
                        "Could be installed from conda-forge or PyPI."
                    ) from e
                uncompressed_len = struct.unpack(">H", chunk_buffer[:2])[0]
                chunk_buffer = lzf.decompress(chunk_buffer, uncompressed_len)
            else:
                raise NotImplementedError(
                    "Filter with id: %i import not supported" % (filter_id)
                )
        return chunk_buffer

    @staticmethod
    def _unshuffle(chunk_buffer, itemsize):
        """
        Reverse the HDF5 shuffle filter and return the result as ``bytes``.

        Shuffling stores byte 0 of every element, then byte 1, and so on;
        this interleaves them back into elements. NumPy releases the GIL while
        copying, unlike a bytearray slice assignment, so chunks can be
        unshuffled on several threads at once.
        """
        # The HDF5 shuffle filter only operates on the largest prefix of the
        # buffer that is an exact multiple of itemsize; any trailing remainder
        # bytes (e.g. from a filter applied before shuffle in the pipeline,
        # such as fletcher32) are left unshuffled and simply copied through.
        buffer_size = len(chunk_buffer)
        main_size = buffer_size - buffer_size % itemsize
        count = main_size // itemsize
        if itemsize == 1 or count == 0:
            return bytes(chunk_buffer)

        lanes = np.frombuffer(chunk_buffer, dtype=np.uint8, count=main_size)
        unshuffled = np.empty(buffer_size, dtype=np.uint8)
        elements = unshuffled[:main_size].reshape(count, itemsize)
        for lane in range(itemsize):
            elements[:, lane] = lanes[lane * count : (lane + 1) * count]
        if main_size != buffer_size:
            unshuffled[main_size:] = np.frombuffer(
                chunk_buffer[main_size:], dtype=np.uint8
            )
        return unshuffled.tobytes()

    @staticmethod
    def _verify_fletcher32(chunk_buffer):
        """Verify a chunk with a fletcher32 checksum."""
        if len(chunk_buffer) < 4:
            raise ValueError("fletcher32 checksum invalid")
        # numcodecs cannot encode an empty payload; its checksum is zero.
        checksum = (
            _FLETCHER32_CODEC.encode(chunk_buffer[:-4])[-4:]
            if len(chunk_buffer) > 4
            else b"\x00\x00\x00\x00"
        )
        # Fletcher32 represents modulo-65535 zero as either 0 or 0xffff.
        calculated = tuple(word % 65535 for word in struct.unpack(">HH", checksum))
        stored = tuple(word % 65535 for word in struct.unpack(">HH", chunk_buffer[-4:]))
        if calculated != stored:
            raise ValueError("fletcher32 checksum invalid")
        return True


class BTreeV2(AbstractBTree):
    """
    HDF5 version 2 B-Tree.
    """

    # III.A.2. Disk Format: Level 1A2 - Version 2 B-trees
    B_TREE_HEADER = dict(
        (
            ("signature", "4s"),
            ("version", "B"),
            ("node_type", "B"),
            ("node_size", "I"),
            ("record_size", "H"),
            ("depth", "H"),
            ("split_percent", "B"),
            ("merge_percent", "B"),
            ("root_address", "Q"),  # 8 byte addressing
            ("root_nrecords", "H"),
            ("total_nrecords", "Q"),  # 8 byte addressing
        )
    )

    B_LINK_NODE = dict(
        (
            ("signature", "4s"),
            ("version", "B"),
            ("node_type", "B"),
        )
    )  # type: ignore[assignment]

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
        assert header["signature"] == b"BTHD"
        assert header["node_type"] == self.NODE_TYPE
        return header

    def _calculate_address_formats(self, header):
        node_size = header["node_size"]
        record_size = header["record_size"]

        nrecords_max = 0
        ntotalrecords_max = 0
        address_formats = []
        for node_level in range(header["depth"] + 1):
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
            address_formats.append(
                (offset_size, num1_size, num2_size, offset_fmt, num1_fmt, num2_fmt)
            )
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
        """Calculate the maximal records a node can contain."""
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
        return (node_size - 10 - addr_size) // (record_size + addr_size)

    @staticmethod
    def _required_bytes(integer):
        """Calculate the minimal required bytes to contain an integer."""
        return (max(integer.bit_length(), 1) + 7) // 8

    def _read_node(self, address, node_level):
        """Return a single node in the B-Tree located at a given offset."""
        offset, nrecords, ntotalrecords = address
        node = self._read_node_header(offset, node_level)

        record_size = self.header["record_size"]

        keys = []
        for _ in range(nrecords):
            record = self.fh.read(record_size)
            keys.append(self._parse_record(record))

        addresses = []
        fmts = self.address_formats[node_level]
        if fmts[0]:
            offset_size, num1_size, num2_size, offset_fmt, num1_fmt, num2_fmt = fmts
            for _ in range(nrecords + 1):
                offset = struct.unpack(offset_fmt, self.fh.read(offset_size))[0]
                num1 = struct.unpack(num1_fmt, self.fh.read(num1_size))[0]
                num1 = int.from_bytes(num1, byteorder="little", signed=False)
                if num2_size:
                    num2 = struct.unpack(num2_fmt, self.fh.read(num2_size))[0]
                    num2 = int.from_bytes(num2, byteorder="little", signed=False)
                else:
                    num2 = num1
                addresses.append((offset, num1, num2))

        node["keys"] = keys
        node["addresses"] = addresses
        return node

    def _read_node_header(self, offset, node_level):
        """Return a single node header in the b-tree located at a give offset."""
        self.fh.seek(offset)
        node = _unpack_struct_from_file(self.B_LINK_NODE, self.fh)
        assert node["node_type"] == self.NODE_TYPE
        if node_level:
            # Internal node (has children)
            assert node["signature"] == b"BTIN"
        else:
            # Leaf node (has no children)
            assert node["signature"] == b"BTLF"
        node["node_level"] = node_level
        return node

    def iter_records(self):
        """Iterate over all records."""
        for nodelist in self.all_nodes.values():
            for node in nodelist:
                yield from node["keys"]

    def _parse_record(self, record):
        raise NotImplementedError


class BTreeV2GroupNames(BTreeV2):
    """
    HDF5 version 2 B-Tree storing group names (type 5).
    """

    NODE_TYPE = 5  # type: ignore[assignment]

    def _parse_record(self, record):
        namehash = struct.unpack_from("<I", record, 0)[0]
        return {"namehash": namehash, "heapid": record[4 : 4 + 7]}


class BTreeV2GroupOrders(BTreeV2):
    """
    HDF5 version 2 B-Tree storing group creation orders (type 6).
    """

    NODE_TYPE = 6  # type: ignore[assignment]

    def _parse_record(self, record):
        creationorder = struct.unpack_from("<Q", record, 0)[0]
        return {"creationorder": creationorder, "heapid": record[8 : 8 + 7]}


class BTreeV2AttrCreationOrder(BTreeV2):
    """
    HDF5 version 2 B-Tree storing attribute creation orders (type 9).
    See the Type 9 Record Layout, note the different ordering from type 6.
    """

    NODE_TYPE = 9  # type: ignore[assignment]

    def _parse_record(self, record):
        return _unpack_struct_from(V2_BTREE_NODE_TYPE_9_LAYOUT, record)


class BTreeV2AttrNames(BTreeV2):
    """
    HDF5 version 2 B-Tree storing attribute names (type 8).
    """

    NODE_TYPE = 8  # type: ignore[assignment]

    def _parse_record(self, record):
        return _unpack_struct_from(V2_BTREE_NODE_TYPE_8_LAYOUT, record)


class BTreeV2HugeObjectsIndirect(BTreeV2):
    """
    HDF5 version 2 B-Tree storing indirectly accessed, non-filtered huge objects.

    Record layout (type 1):
    - huge object address
    - huge object length
    - huge object ID (lookup key)
    """

    NODE_TYPE = 1  # type: ignore[assignment]

    def __init__(self, fh, offset, sizeof_offsets, sizeof_lengths):
        self._sizeof_offsets = sizeof_offsets
        self._sizeof_lengths = sizeof_lengths
        self.record_by_id = {}
        super().__init__(fh, offset)

    def _parse_record(self, record):
        rec_offset = _unpack_integer(self._sizeof_offsets, record, 0)
        rec_length = _unpack_integer(self._sizeof_lengths, record, self._sizeof_offsets)
        rec_id = _unpack_integer(
            self._sizeof_lengths,
            record,
            self._sizeof_offsets + self._sizeof_lengths,
        )
        return {"address": rec_offset, "length": rec_length, "id": rec_id}

    def find(self, huge_object_id):
        """Return the record matching a huge-object ID/key."""
        # read the b-tree and cache.
        if self.record_by_id == {}:
            for record in self.iter_records():
                self.record_by_id[record["id"]] = record
        try:
            return self.record_by_id[huge_object_id]
        except KeyError as e:
            raise KeyError(
                f"Huge object ID not found in v2 B-tree: {huge_object_id}"
            ) from e


# IV.A.2.l The Data Storage - Filter Pipeline message
RESERVED_FILTER = 0
GZIP_DEFLATE_FILTER = 1
SHUFFLE_FILTER = 2
FLETCH32_FILTER = 3
SZIP_FILTER = 4
NBIT_FILTER = 5
SCALEOFFSET_FILTER = 6
LZF_FILTER = 32000


# Attribute message B-Tree node types
# haven't tested type 8 yet, not sure how to get some.
#
V2_BTREE_NODE_TYPE_8_LAYOUT = dict(
    (("heapid", "8s"), ("flags", "B"), ("creationorder", "I"), ("namehash", "I"))
)

V2_BTREE_NODE_TYPE_9_LAYOUT = dict(
    (("heapid", "8s"), ("flags", "B"), ("creationorder", "I"))
)
