import numpy as np
import fsspec.utils as futils
from bisect import bisect_right
from collections import namedtuple
from operator import mul
from pyfive.indexing import OrthogonalIndexer, ZarrArrayStub
from pyfive.btree import BTreeV1RawDataChunks
from pyfive.core import Reference, UNDEFINED_ADDRESS
from pyfive.misc_low_level import (
    get_vlen_string_data_contiguous,
    get_vlen_string_data_from_chunk,
    _decode_array,
    dtype_replace_refs_with_object,
)
from pyfive.p5t import P5CompoundType, P5VlenStringType, P5ReferenceType, P5SequenceType
from io import UnsupportedOperation
from time import time

import os
import struct
import threading
import logging
from importlib.metadata import version
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

StoreInfo = namedtuple("StoreInfo", "chunk_offset filter_mask byte_offset size")
ChunkIndex = namedtuple("ChunkIndex", "chunk_address chunk_dims")


class ChunkRead:
    """
    Mixin providing parallel and bulk chunk-reading strategies.

    ``DatasetID`` inherits from this class so that the hot path in
    ``_get_selection_via_chunks`` can dispatch to the best available I/O strategy:

    * **Case A - fsspec ``cat_ranges``**: a single bulk request issued to the
      filesystem; ideal for remote stores (S3, GCS, https,  …).
    * **Case B - ``os.pread`` thread pool**: parallel POSIX reads sharing a
      single file descriptor without seek contention.
    * **Case C - serial fallback**: safe for in-memory buffers and any
      custom file-like wrapper.
    """

    # ------------------------------------------------------------------ #
    # Shared helpers                                                       #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _cat_ranges_raise(fs, paths, starts, stops, batch_size=None):
        """Return cat_ranges buffers, raising any read exceptions immediately."""
        # Not every backend accepts ``batch_size`` or ``on_error``; drop what a
        # backend rejects rather than failing, so each range is still fetched once.
        kwargs = {"on_error": "raise"}
        if batch_size is not None:
            kwargs["batch_size"] = batch_size
        while True:
            try:
                buffers = fs.cat_ranges(paths, starts, stops, **kwargs)
                break
            except TypeError as e:
                rejected = [
                    name
                    for name in kwargs
                    if "unexpected keyword" in str(e) and name in str(e)
                ]
                if not rejected:
                    # could be something else, so re-raise
                    raise
                for name in rejected:
                    del kwargs[name]

        # and handle the case of not honouring the on_error argument
        for buffer in buffers:
            if isinstance(buffer, Exception):
                raise buffer

        return buffers

    def set_parallelism(
        self, thread_count=0, cat_range_allowed=True, btree_parallel=False
    ):
        """
        Configure chunk-read parallelism.

        ``thread_count`` sets the number of worker threads used to read and
        decode chunks:

        - ``0`` (the default) uses no worker threads: chunks are decoded
          one at a time on the calling thread
        - ``>0`` decodes chunks (decompression, shuffle, checksum) on that many
          worker threads, for POSIX files (also reading with ``os.pread``) and
          for fsspec files read in bulk with ``cat_ranges``

        The default is ``0`` because pyfive is often called from threads or
        task schedulers (for example ``dask``) of the caller's own; enabling
        this on top of them multiplies the number of busy threads. Serial reads
        of other file-like objects are not threaded, as they share one handle.

        ``cat_range_allowed`` enables fsspec bulk reads via ``cat_ranges``
        for compatible non-posix file handles. Default True

        ``btree_parallel`` enables parallel reads for b-tree nodes when building
        the chunk index. Default False.

        This is a ``pyfive`` API extension. It is recommended to enable it when working
        with remote files, but it may not be suitable for local files. Hence defaults
        are that cat_ranges is on (for remote files) and threads are off (for local files).

        """
        if thread_count is None:
            thread_count = 0
        thread_count = int(thread_count)
        if thread_count < 0:
            raise ValueError("thread_count must be >= 0")

        self._thread_count = thread_count
        self._cat_range_allowed = cat_range_allowed
        self._btree_parallel = btree_parallel
        logger.info(
            "Parallelism: thread_count=%d, cat_range_allowed=%s, btree_parallel=%s",
            self._thread_count,
            self._cat_range_allowed,
            self._btree_parallel,
        )

    def _get_required_chunks(self, indexer):
        """
        Walk *indexer* and return a list of
        ``(chunk_coords, chunk_selection, out_selection, storeinfo)``
        tuples for every chunk needed to satisfy the selection.
        """
        result = []
        for chunk_coords, chunk_selection, out_selection in indexer:
            chunk_coords = tuple(map(mul, chunk_coords, self.chunks))
            storeinfo = self._index[chunk_coords]
            result.append((chunk_coords, chunk_selection, out_selection, storeinfo))
        return result

    def _decode_chunk(self, chunk_buffer, filter_mask, dtype):
        """
        Apply the filter pipeline (if any) and return a shaped ndarray
        """
        if self.filter_pipeline is not None:
            chunk_buffer = BTreeV1RawDataChunks._filter_chunk(
                chunk_buffer,
                filter_mask,
                self.filter_pipeline,
                self.dtype.itemsize,
            )
        return np.frombuffer(chunk_buffer, dtype=dtype).reshape(
            self.chunks, order=self._order
        )

    def _select_chunks(
        self,
        indexer,
        out,
        dtype,
        max_block: int | None = None,
        batch_size: int | None = None,
    ):
        """
        Collect required chunks and dispatch I/O to the best strategy.
        Called by ``_get_selection_via_chunks`` in place of the serial loop.
        """
        chunks = self._get_required_chunks(indexer)
        if not chunks:
            return

        # Case A: fsspec - bulk parallel fetch via cat_ranges
        if not self.posix and self._cat_range_allowed:  # type: ignore[attr-defined]
            fh = self._fh  # type: ignore[attr-defined]
            actual_fh = getattr(fh, "fh", fh)  # support wrapped file-like objects
            if hasattr(actual_fh, "fs") and hasattr(actual_fh.fs, "cat_ranges"):
                logger.info(
                    f"[pyfive] chunk read strategy: fsspec_cat_ranges ({len(chunks)} chunks)"
                )
                self._read_bulk_fsspec(
                    fh, chunks, out, dtype, max_block=max_block, batch_size=batch_size
                )
                return

        # Case B: POSIX - thread-parallel reads via os.pread
        if self.posix and hasattr(os, "pread") and self._thread_count != 0:  # type: ignore[attr-defined]
            logger.info(
                "[pyfive] chunk read strategy: posix_pread_threads workers=%s (%d chunks)",
                self._thread_count,
                len(chunks),
            )
            self._read_parallel_threads(chunks, out, dtype)
            return

        # Case C: serial fallback
        logger.info("[pyfive] chunk read strategy: serial  (%d chunks)", len(chunks))
        self._read_serial(chunks, out, dtype)

    # ------------------------------------------------------------------ #
    # Strategy implementations                                             #
    # ------------------------------------------------------------------ #

    def _read_serial(self, chunks, out, dtype):
        """
        Read one chunk at a time (safe for any file-like object).
        """
        fh = self._fh
        for _coords, chunk_sel, out_sel, storeinfo in chunks:
            fh.seek(storeinfo.byte_offset)
            chunk_buffer = fh.read(storeinfo.size)
            out[out_sel] = self._decode_chunk(
                chunk_buffer, storeinfo.filter_mask, dtype
            )[chunk_sel]
        if self.posix:
            fh.close()

    def _run_chunk_tasks(self, task, items):
        """
        Call ``task(item)`` for every item, on ``thread_count`` worker threads
        when that is non-zero, otherwise serially on the calling thread.

        Each task decodes one chunk and writes its own, disjoint, region of the
        output array, so tasks need no locking. Exceptions raised by a task are
        re-raised here.
        """
        workers = min(self._thread_count, len(items))
        if workers == 0:
            for item in items:
                task(item)
            return
        with ThreadPoolExecutor(max_workers=workers) as executor:
            # Consuming the iterator surfaces any exception from a worker.
            for _ in executor.map(task, items):
                pass

    def _read_parallel_threads(self, chunks, out, dtype):
        """
        Thread-parallel read and decode via ``os.pread``.

        ``os.pread`` does not advance the file-position pointer, so all
        worker threads share a single open file descriptor safely. Each worker
        reads, decodes and stores one chunk at a time, so reads overlap with
        decoding and only ``thread_count`` compressed chunks are held at once.
        """
        fh = open(self._filename, "rb")
        fd = fh.fileno()

        def _read_decode_store(item):
            _coords, chunk_sel, out_sel, storeinfo = item
            chunk_buffer = os.pread(fd, storeinfo.size, storeinfo.byte_offset)
            out[out_sel] = self._decode_chunk(
                chunk_buffer, storeinfo.filter_mask, dtype
            )[chunk_sel]

        try:
            self._run_chunk_tasks(_read_decode_store, chunks)
        finally:
            fh.close()

        logger.info(
            "pyfive thread pool read completed using %d threads", self._thread_count
        )

    def _read_bulk_fsspec(
        self,
        fh,
        chunks,
        out,
        dtype,
        max_block: int | None = None,
        batch_size: int | None = None,
    ):
        """
        Bulk read via ``fsspec`` ``cat_ranges``.

        Issues a single pipelined request for all required byte-ranges,
        which on object stores typically translates to a small number of
        HTTP range requests rather than one round-trip per chunk
        (reaching through the MetadataBufferingWrapper).
        """
        actual_fh = getattr(fh, "fh", fh)  # support wrapped file-like objects
        starts = [si.byte_offset for _, _, _, si in chunks]
        stops = [si.byte_offset + si.size for _, _, _, si in chunks]

        paths = [actual_fh.path] * len(chunks)
        chunk_ranges = list(zip(paths, starts, stops))
        if max_block is not None:
            paths, starts, stops = futils.merge_offset_ranges(
                paths, starts, stops, max_block=max_block
            )

        buffers = list(
            self._cat_ranges_raise(
                actual_fh.fs, paths, starts, stops, batch_size=batch_size
            )
        )

        # Fetched ranges never overlap, so the only range that can contain a
        # chunk is the last one starting at or before it. Bisecting keeps this
        # O(K log K); scanning every range for every chunk was O(K^2).
        fetched = sorted(
            (path, start, stop, index)
            for index, (path, start, stop) in enumerate(zip(paths, starts, stops))
        )
        fetched_keys = [(path, start) for path, start, _stop, _index in fetched]

        # Locate each chunk within a fetched block, and count the chunks still
        # needing each block, so that a block can be freed as soon as its last
        # chunk is decoded rather than holding everything fetched until the end.
        spans = []
        pending = [0] * len(buffers)
        for path, start, stop in chunk_ranges:
            position = bisect_right(fetched_keys, (path, start)) - 1
            if position >= 0:
                merged_path, merged_start, merged_stop, index = fetched[position]
                if merged_path == path and stop <= merged_stop:
                    spans.append((index, start - merged_start, stop - merged_start))
                    pending[index] += 1
                    continue
            raise RuntimeError("Merged range does not contain requested chunk")

        release = threading.Lock()

        def _decode_store(i):
            _coords, chunk_sel, out_sel, storeinfo = chunks[i]
            index, low, high = spans[i]
            chunk_buffer = buffers[index][low:high]
            with release:
                pending[index] -= 1
                if pending[index] == 0:
                    buffers[index] = None
            out[out_sel] = self._decode_chunk(
                chunk_buffer, storeinfo.filter_mask, dtype
            )[chunk_sel]

        self._run_chunk_tasks(_decode_store, range(len(chunks)))


class DatasetID(ChunkRead):
    """
    Implements an "HDF5 dataset identifier", which despite the name, actually
    represents the data of a dataset in a file, and not an identifier. It includes all
    the low level methods for working with chunked data, lazily or not.

    This class has been deliberately implemented in such as way so as to cache all
    the relevant metadata, so that once you have an instance,
    it is completely independent of the parent file, and it can be used
    efficiently in distributed threads without thread contention to the b-tree etc.
    *This behaviour may differ from* ``h5py``, *which cannot isolate the dataset access
    from the parent file access as both share underlying C-structures.*

    """

    def __init__(
        self,
        dataobject: "DataObjects",  # type: ignore[name-defined]  # noqa: F821
        noindex: bool = False,
        pseudo_chunking_size_MB: int = 4,
        max_request_block: int | None = None,
        batch_request_size: int | None = None,
        global_heaps: dict | None = None,
    ) -> None:
        """
        Instantiated with the ``pyfive`` ``datasetdataobject``, we copy and cache everything
        we want so that the only file operations are now data accesses.

        noindex provides a method for controlling how lazy the data load
        actually is. This version supports values of False (normal behaviour
        index is read when datasetid first instantiated) or True (index
        is only read when the data is accessed).

        if ``pseudo_chunking_size_MB`` is set to a value greater than zero, and
        if the storage is not local posix (and hence ``np.mmap``is not available) then
        when accessing contiguous variables, we attempt to find a suitable
        chunk shape to approximate that volume and read the contigous variable
        as if were chunked. This is to facilitate lazy loading of partial data
        from contiguous storage.

        (Currently the only way to change this value is by explicitly using
        the ``set_pseudo_chunk_size method``. Most users will not need to change
        it.)

        """

        self._max_block = max_request_block
        self._batch_size = batch_request_size

        self._order = dataobject.order
        fh = dataobject.fh

        try:
            # See if 'fh' is an underlying file descriptor
            fh.fileno()
        except (AttributeError, OSError):
            #  No file descriptor => Not Posix
            self.posix = False
            self.__fh = fh
            self.pseudo_chunking_size = pseudo_chunking_size_MB * 1024 * 1024
            try:
                # maybe this is an S3File instance?
                self._filename = getattr(fh, "path")
            except:
                # maybe a remote https file opened as bytes?
                # failing that, maybe a memory file, return as None
                # or even a Pyfive Dataset instance
                self._filename = getattr(fh, "full_name", "None")
                if self._filename == "None":
                    fh = getattr(fh, "fh", None)
                    if fh is not None:
                        self._filename = fh.path
        else:
            # Has a file descriptor => Posix
            self.posix = True
            self._filename = fh.name
            self.pseudo_chunking_size = 0

        self.filter_pipeline = dataobject.filter_pipeline
        self.shape = dataobject.shape
        self.rank = len(self.shape)
        self.chunks = dataobject.chunks
        self._decode_strings = dataobject.decode_strings
        self.set_parallelism()

        # Share the per-file global-heap cache so all datasets/attrs that point into
        # the same GCOL reuse the same parsed collection instead of re-reading it.
        self._global_heaps: dict = {} if global_heaps is None else global_heaps

        self._msg_offset, self.layout_class, self.property_offset = (
            dataobject.get_id_storage_params()
        )
        self._unique = (self._filename, self.shape, self._msg_offset)

        self._ptype = dataobject.ptype

        self._meta = DatasetMeta(dataobject)

        self._index = None
        self.__index_built = False
        self._index_params = None
        match self.layout_class:
            case 0:  # compact storage
                self._data = self._get_compact_data(dataobject)
            case 1:  # contiguous storage
                (self.data_offset,) = struct.unpack_from(
                    "<Q", dataobject.msg_data, self.property_offset
                )
            case 2:  # chunked storage
                self._index_params = ChunkIndex(
                    dataobject._chunk_address, dataobject._chunk_dims
                )
                if not noindex:
                    self._build_index()

    def __hash__(self):
        """The hash is based on assuming the file path, the location
        of the data in the file, and the data shape are a unique
        combination.
        """
        return hash(self._unique)

    def __eq__(self, other):
        """
        Equality is based on the filename, location of the data in the file
        and the shape of the data.
        """
        return self._unique == other._unique

    def __chunk_init_check(self):
        """
        Used by all the chunk methods to see if this dataset is
        chunked, and if so, if the index is present, and if not,
        build it. Otherwise handle errors etc.
        """
        if self.layout_class != 2:
            raise TypeError("Dataset is not chunked ")
        return not self.index == {}

    def get_chunk_info(self, index):
        """
        Retrieve storage information about a chunk specified by its index.
        """
        if self.__chunk_init_check():
            return self._index[self._nthindex[index]]
        else:
            raise TypeError("Dataset is not chunked ")

    def get_chunk_info_by_coord(self, coordinate_index):
        """
        Retrieve information about a chunk specified by the array address of the chunk’s
        first element in each dimension.
        """
        if self.__chunk_init_check():
            return self._index[coordinate_index]
        else:
            raise TypeError("Dataset is not chunked ")

    def get_num_chunks(self):
        """
        Return total number of chunks in dataset
        """
        if self.__chunk_init_check():
            return len(self._index)
        else:
            return 0

    def read_direct_chunk(self, chunk_position, **kwargs):
        """
        Returns a tuple containing the filter_mask and the raw data storing this chunk as bytes.
        Additional arguments supported by ``h5py`` are not supported here.
        """
        if not self.__chunk_init_check():
            raise TypeError("Dataset is not chunked ")
        if chunk_position not in self._index:
            raise OSError("Chunk coordinates must lie on chunk boundaries")
        storeinfo = self._index[chunk_position]
        return storeinfo.filter_mask, self._get_raw_chunk(storeinfo)

    def get_data(self, args, fillvalue):
        """Called by the dataset getitem method"""
        no_storage = False
        match self.layout_class:
            case 0:  # compact storage
                if self._data is None:
                    no_storage = True
                else:
                    return self._read_compact_data(args)
            case 1:  # contiguous storage
                if self.data_offset == UNDEFINED_ADDRESS:
                    no_storage = True
                else:
                    return self._get_contiguous_data(args, fillvalue)
            case 2:  # chunked storage
                if not self.__index_built:
                    self._build_index()
                if not self._index:
                    no_storage = True
                else:
                    if isinstance(self._ptype, P5ReferenceType):
                        # references need to read all the chunks for now
                        return self._get_selection_via_chunks(())[args]
                    else:
                        # this is lazily reading only the chunks we need
                        return self._get_selection_via_chunks(args)

        if no_storage:
            return np.full(self.shape, fillvalue, dtype=self.dtype)[args]

    def iter_chunks(self, args):
        """Iterate over chunks in a chunked dataset.

        The args argument is a (possibly empty) sequence of indices
        that defines the region to be used. If an empty sequence then
        the entire dataspace will be used for the iterator.

        For each chunk within the given region, the iterator yields a
        tuple of indices that gives the intersection of the given
        chunk with the selection area. This can be used to read data
        in that chunk.

        """
        if not self.__chunk_init_check():
            return None

        def convert_selection(tuple_of_slices):
            # while a slice of the form slice(a,b,None) is equivalent
            # in function to a slice of form (a,b,1) it is not the same.
            # For compatability I've gone for "the same"
            def convert_slice(aslice):
                if aslice.step is None:
                    return slice(aslice.start, aslice.stop, 1)
                return aslice

            return tuple([convert_slice(a) for a in tuple_of_slices])

        array = ZarrArrayStub(self.shape, self.chunks)

        if args:
            # We have implemented what the docstring says it does below,
            # but that's not what h5py actually does, and what is it
            # actually does is useless, so we haven't implemented that
            raise NotImplementedError(
                "h5py does something silly, and our implementation does not"
            )
            indexer = OrthogonalIndexer(args[0], array)
        else:
            indexer = OrthogonalIndexer(args, array)
        for chunk_coords, chunk_selection, out_selection in indexer:
            if args:
                yield convert_selection(chunk_selection)
            else:
                yield convert_selection(out_selection)

    ##### The following property is made available to support ActiveStorage
    ##### and to help those who may want to generate kerchunk indices and
    ##### bypass the iterator methods.
    @property
    def index(self):
        """Direct access to the chunk index, if there is one. This is a ``pyfive`` API extension."""
        # can't use init_chunk_check because that would be an infinite regression
        if self.layout_class != 2:
            raise TypeError("Data is not chunked")
        if not self._index:
            self._build_index()
        return self._index

    ##### This property is made available to help understand object store performance
    @property
    def btree_range(self):
        """A tuple with the addresses of the first b-tree node
        for this variable, and the address of the furthest away node
        (Which may not be the last one in the chunk index). This property
        may be of use in understanding the read performance of chunked
        data in object stores.  ``btree_range`` is a ``pyfive`` API extension.
        """
        self.__chunk_init_check()
        return (self._btree_start, self._btree_end)

    ##### This property is made available to help understand object store performance
    @property
    def first_chunk(self):
        """The integer address of the first data chunk for this variable.

        This property may be of use in understanding the read
        performance of chunked data in object stores.  ``first_chunk``
        is a ``pyfive`` API extension.

        """
        self.__chunk_init_check()
        min_offset = None
        for k in self._index:
            if min_offset is None or self._index[k].byte_offset < min_offset:
                min_offset = self._index[k].byte_offset
        return min_offset

    #### The following method can be used to set pseudo chunking size after the
    #### file has been closed and before data transactions. This is pyfive specific
    def set_pseudo_chunk_size(self, newsize_MB):
        """Set pseudo chunking size for contiguous variables.
        This is a ``pyfive`` API extension.
        The default value is 4 MB which should be suitable for most applications.
        For arrays smaller than this value, no pseudo chunking is used.
        Larger arrays will be accessed in in roughly ``newsize_MB`` reads."""
        if self.layout_class == 1:
            if not self.posix:
                self.pseudo_chunking_size = newsize_MB * 1024 * 1024
            else:
                pass  # silently ignore it, we'll be using a np.memmap
        else:
            raise ValueError("Attempt to set pseudo chunking on non-contigous variable")

    def get_chunk_info_from_chunk_coord(self, chunk_coords):
        """
        Retrieve storage information about a chunk specified by its index.
        This is a ``pyfive`` API extension.
        This index is in chunk space (as used by ``zarr``) and needs to be converted
        to HDF5 coordinate space.
        Additionally, if this file is not chunked, the storeinfo
        is returned for the contiguous data as if it were one chunk.
        """
        if not self._index:
            dummy = StoreInfo(
                None, None, self.data_offset, self.dtype.itemsize * np.prod(self.shape)
            )
            return dummy
        else:
            coord_index = tuple(map(mul, chunk_coords, self.chunks))
            return self.get_chunk_info_by_coord(coord_index)

    ######
    # The following DatasetID methods are used by PyFive and you wouldn't expect
    # third parties to use them. They are not H5Py methods.
    ######

    def _build_index(self):
        """
        Build the chunk index if it doesn't exist. This is only
        called for chunk data, and only when the variable is accessed.
        That is, it is not called when we an open a file, or when
        we list the variables in a file, but only when we do
        ``v = open_file['var_name']`` where ``var_name`` is chunked.
        """

        if self._index is not None:
            return

        if self._index_params is None:
            raise RuntimeError("Attempt to build index with no chunk index parameters")

        # look out for an empty dataset, which will have no btree
        if (
            np.prod(self.shape) == 0
            or self._index_params.chunk_address == UNDEFINED_ADDRESS
        ):
            self._index = {}
            # FIXME: There are other edge cases for self._index = {} to handle
            self._btree_end, self._btree_start = None, None
            return

        logger.info(
            "[pyfive] Building chunk index (pyfive version=%s)",
            version("pyfive"),
        )

        # FIXME: How do we know it's a V1 B-tree?
        # There are potentially five different chunk indexing options according to
        # https://docs.hdfgroup.org/archive/support/HDF5/doc/H5.format.html#AppendixC
        t0 = time()
        fh = self._fh
        chunk_btree = BTreeV1RawDataChunks(
            fh,
            self._index_params.chunk_address,
            self._index_params.chunk_dims,
            fetch_fn=self._make_btree_fetch_fn(),
        )
        if self.posix:
            fh.close()

        self._index = {}
        self._nthindex = []

        for node in chunk_btree.all_nodes[0]:
            for node_key, addr in zip(node["keys"], node["addresses"]):
                start = node_key["chunk_offset"][:-1]
                key = start
                size = node_key["chunk_size"]
                filter_mask = node_key["filter_mask"]
                self._nthindex.append(key)
                self._index[key] = StoreInfo(key, filter_mask, addr, size)

        self._btree_start = chunk_btree.offset
        self._btree_end = chunk_btree.last_offset

        t1 = time() - t0
        if t1 < 1.0:
            elapsed = f"{t1 * 1000:.0f}ms"
        else:
            elapsed = f"{t1:.1f}s"
        logger.info(
            "[pyfive] Chunk index built: btree range=%s; elapsed=%s",
            (self._btree_start, self._btree_end),
            elapsed,
        )

        self.__index_built = True

    def _make_btree_fetch_fn(self):
        """
        Return fetch_fn(addresses, size) for b-tree leaf reads, or None.
        The default here is None (self._btree_parallel is None),
        which means that b-tree nodes will be read serially via the file handle's read method.
        """

        if not self._btree_parallel:
            return None

        actual_fh = None
        if not self.posix:
            fh = self._fh
            actual_fh = getattr(fh, "fh", fh)

        # Case A: fsspec cat_ranges for remote file-like handles.
        if self._cat_range_allowed:
            fs = getattr(actual_fh, "fs", None)
            path = getattr(actual_fh, "path", None)
            if fs is not None and path is not None and hasattr(fs, "cat_ranges"):

                def fetch_cat_ranges(addresses, size):
                    stops = [addr + size for addr in addresses]
                    return self._cat_ranges_raise(
                        fs,
                        [path] * len(addresses),
                        addresses,
                        stops,
                    )

                return fetch_cat_ranges

        # Case B: POSIX thread pool with os.pread.
        if self.posix and hasattr(os, "pread") and self._thread_count != 0:
            filename = self._filename
            thread_count = self._thread_count

            def fetch_pread(addresses, size):
                fh_local = open(filename, "rb")
                fd = fh_local.fileno()
                try:
                    with ThreadPoolExecutor(max_workers=thread_count) as executor:
                        return list(
                            executor.map(
                                lambda addr: os.pread(fd, size, addr), addresses
                            )
                        )
                finally:
                    fh_local.close()

            return fetch_pread

        # Case C: serial fallback in BTreeV1RawDataChunks.
        return None

    def _get_contiguous_data(self, args, fillvalue):
        if isinstance(self._ptype, P5ReferenceType):
            size = self._ptype.size
            if size != 8:
                raise NotImplementedError(f"Unsupported Reference type - size {size}")

            fh = self._fh
            ref_addresses = np.memmap(
                fh,
                dtype=("<u8"),
                mode="c",
                offset=self.data_offset,
                shape=self.shape,
                order=self._order,
            )
            result = np.array([Reference(addr) for addr in ref_addresses])[args]
            if self.posix:
                fh.close()

            return result
        elif isinstance(self._ptype, P5VlenStringType):
            fh = self._fh
            array = get_vlen_string_data_contiguous(
                fh,
                self.data_offset,
                self._global_heaps,
                self.shape,
                self._ptype,
                fillvalue,
                self._decode_strings,
            )
            if self.posix:
                fh.close()

            return array.reshape(self.shape, order=self._order)[args]
        elif isinstance(self._ptype, P5SequenceType):
            raise NotImplementedError(
                f"datatype not implemented - {self._ptype.__class__.__name__}"
            )

        if not self.posix:
            # Not posix
            return self._get_direct_from_contiguous(args)
        else:
            # posix
            try:
                # Create a memory-map to the stored array, which
                # means that we will end up only copying the
                # sub-array into in memory.
                fh = self._fh
                view = np.memmap(
                    fh,
                    dtype=self.dtype,
                    mode="c",
                    offset=self.data_offset,
                    shape=self.shape,
                    order=self._order,
                )
                # Create the sub-array
                result = view[args]
                # Copy the data from disk to physical memory
                result = result.view(type=np.ndarray)
                if not self._ptype.is_atomic:
                    # if we have a type which is not atomic
                    # we have to get a view
                    result = result.view(self.dtype)
                    # and for compounds we have to wrap any References properly
                    # todo: check for Enum etc types
                    if isinstance(self._ptype, P5CompoundType):
                        new_dtype = dtype_replace_refs_with_object(self.dtype)
                        new_array = np.empty(result.shape, dtype=new_dtype)
                        new_array[:] = result
                        result = _decode_array(result, new_array)
                fh.close()
                return result
            except UnsupportedOperation:
                return self._get_direct_from_contiguous(args)

    def _get_compact_data(self, dataobject):
        data = None
        layout = None
        for msg in dataobject.msgs:
            if msg["type"] == 8:
                layout = msg
                break
        if layout is None:
            raise ValueError("No layout message in compact dataset?")
        byts = dataobject.msg_data[
            msg["offset_to_message"] : msg["offset_to_message"] + msg["size"]
        ]
        layout_version = byts[0]
        if layout_version == 1 or layout_version == 2:
            raise NotImplementedError("Compact layout v1 and v2.")
        elif layout_version == 3 or layout_version == 4:
            size = int.from_bytes(byts[2:4], "little")
            data = byts[4 : 4 + size]
        else:
            raise ValueError("Unknown layout version.")
        return data

    def _read_compact_data(self, args):
        view = np.frombuffer(
            self._data,
            dtype=self.dtype,
        ).reshape(self.shape)
        # Create the sub-array
        result = view[args]
        return result

    def _get_direct_from_contiguous(self, args=None):
        """
        This is a fallback situation if we can't use a memory map which would otherwise be lazy.
        If pseudo_chunking_size is set, we attempt to read the contiguous data in chunks
        otherwise we have to read the entire array. This is a fallback situation if we
        can't use a memory map which would otherwise be lazy. This will normally be when
        we don't have a true Posix file. We should never end up here with compressed
        data.
        """

        def __get_pseudo_shape():
            """Determine an appropriate chunk and stride for a given pseudo chunk size"""
            element_size = self.dtype.itemsize
            chunk_shape = np.copy(self.shape)
            while True:
                chunk_size = np.prod(chunk_shape) * element_size
                if chunk_size < self.pseudo_chunking_size:
                    break
                for i in range(len(chunk_shape)):
                    if chunk_shape[i] > 1:
                        chunk_shape[i] //= 2
                        break
            return chunk_shape, chunk_size

        class LocalOffset:
            def __init__(self, shape, chunk_shape, stride):
                chunks_per_dim = [
                    int(np.ceil(a / c)) for a, c in zip(shape, chunk_shape)
                ]
                self.chunk_strides = np.cumprod([1] + chunks_per_dim[::-1])[:-1][::-1]
                self.stride = stride

            def coord_to_offset(self, chunk_coords):
                linear_offset = sum(
                    idx * stride
                    for idx, stride in zip(chunk_coords, self.chunk_strides)
                )
                return linear_offset * self.stride

        fh = self._fh
        if self.pseudo_chunking_size:
            chunk_shape, stride = __get_pseudo_shape()
            stride = int(stride)
            offset_finder = LocalOffset(self.shape, chunk_shape, stride)
            array = ZarrArrayStub(self.shape, chunk_shape)
            indexer = OrthogonalIndexer(args, array)
            out_shape = indexer.shape
            out = np.empty(out_shape, dtype=self.dtype, order=self._order)
            chunk_size = np.prod(chunk_shape)

            for chunk_coords, chunk_selection, out_selection in indexer:
                index = self.data_offset + offset_finder.coord_to_offset(chunk_coords)
                index = int(index)
                fh.seek(index)
                chunk_buffer = fh.read(stride)
                chunk_data = np.frombuffer(chunk_buffer, dtype=self.dtype).copy()
                if len(chunk_data) < chunk_size:
                    # last chunk over end of file
                    padded_chunk_data = np.zeros(chunk_size, dtype=self.dtype)
                    padded_chunk_data[: len(chunk_data)] = chunk_data
                    chunk_data = padded_chunk_data
                out[out_selection] = chunk_data.reshape(chunk_shape, order=self._order)[
                    chunk_selection
                ]

            if self.posix:
                fh.close()

            return out

        else:
            itemsize = self.dtype.itemsize
            num_elements = np.prod(self.shape, dtype=int)
            num_bytes = num_elements * itemsize

            # we need it all, let's get it all (i.e. this really does
            # read the lot)
            fh.seek(self.data_offset)
            chunk_buffer = fh.read(num_bytes)
            chunk_data = np.frombuffer(chunk_buffer, dtype=self.dtype).copy()
            chunk_data = chunk_data.reshape(self.shape, order=self._order)
            chunk_data = chunk_data[args]
            if self.posix:
                fh.close()

            return chunk_data

    def _get_raw_chunk(self, storeinfo):
        """
        Obtain the bytes associated with a chunk.
        """
        fh = self._fh
        fh.seek(storeinfo.byte_offset)
        out = fh.read(storeinfo.size)
        if self.posix:
            fh.close()

        return out

    def _get_selection_via_chunks(self, args):
        """Use the zarr orthogonal indexer to extract data for a specfic
        selection within the dataset array and in doing so, only load
        the relevant chunks.

        """
        if self._index is None:
            raise RuntimeError("Attempt to read chunked data with no index")
        # need a local dtype as we may override it for a reference read.
        dtype = self.dtype
        if isinstance(self._ptype, P5ReferenceType):
            # this is a reference and we're returning that
            size = self._ptype.size
            dtype = "<u8"
            if size != 8:
                raise NotImplementedError("Unsupported Reference type")
        else:
            if np.prod(self.shape) == 0:
                return np.zeros(self.shape)

        array = ZarrArrayStub(self.shape, self.chunks)
        indexer = OrthogonalIndexer(args, array)
        out_shape = indexer.shape
        out = np.empty(out_shape, dtype=dtype, order=self._order)

        if isinstance(self._ptype, P5VlenStringType):
            fh = self._fh

            chunk_shape = self.chunks
            global_heaps = self._global_heaps
            index = self._index
            for chunk_coords, chunk_selection, out_selection in indexer:
                chunk_coords = tuple(map(mul, chunk_coords, self.chunks))
                chunk_data = get_vlen_string_data_from_chunk(
                    fh,
                    index[chunk_coords].byte_offset,
                    global_heaps,
                    chunk_shape,
                    self._ptype,
                    self._decode_strings,
                )
                chunk_data = chunk_data.reshape(chunk_shape)
                out[out_selection] = chunk_data[chunk_selection]

            if self.posix:
                fh.close()

        else:
            self._select_chunks(
                indexer,
                out,
                dtype,
                max_block=self._max_block,
                batch_size=self._batch_size,
            )

        if isinstance(self._ptype, P5ReferenceType):
            to_reference = np.vectorize(Reference)
            out = to_reference(out)

        return out

    @property
    def _fh(self):
        """Return an open file handle to the parent file.

        When the parent file has been closed, we will need to reopen it
        to continue to access data. This facility is provided to support
        thread safe data access. However, now the file is open outside
        a context manager, the user is responsible for closing it,
        though it should get closed when the variable instance is
        garbage collected.

        """

        if self.posix:
            # Posix: Open the file, without caching it.
            return open(self._filename, "rb")

        # Not posix: Use the cached file if it's open, otherwise open
        #            the file and cache it.
        fh = self.__fh
        if fh.closed:
            fh = open(self._filename, "rb")
            self.__fh = fh

        return fh

    @property
    def dtype(self):
        """
        Return numpy dtype of the dataset.
        """
        return self._ptype.dtype

    def get_type(self):
        """
        Return pyfive type of the dataset.
        """
        return self._ptype


class DatasetMeta:
    """
    This is a convenience class to bundle up and cache the metadata
    exposed by the Dataset when DatasetId is constructed.
    """

    def __init__(self, dataobject):
        self.attributes = dataobject.compression
        self.maxshape = dataobject.maxshape
        self.compression = dataobject.compression
        self.compression_opts = dataobject.compression_opts
        self.shuffle = dataobject.shuffle
        self.fletcher32 = dataobject.fletcher32
        self.fillvalue = dataobject.fillvalue
        self.attributes = dataobject.get_attributes()
        self.datatype = dataobject.ptype

        # horrible kludge for now, this isn't really the same sort of thing
        # https://github.com/NCAS-CMS/pyfive/issues/13#issuecomment-2557121461
        # this is used directly in the Dataset init method.
        self.offset = dataobject.offset
