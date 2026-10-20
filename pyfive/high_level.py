"""High-level classes for reading HDF5 files."""

from __future__ import annotations
from collections import deque
from collections.abc import Callable
from collections.abc import Mapping, Sequence
from abc import ABC
import os
import warnings
import posixpath
import warnings
import logging
import numpy as np
from typing import Any, BinaryIO, cast
from pyfive.core import Reference
from pyfive.core import InvalidHDF5File
from pyfive.dataobjects import DataObjects, DatasetID
from pyfive.misc_low_level import FORMAT_SIGNATURE, SuperBlock
from pyfive.h5py import Datatype
from pyfive.layout import check_datasets
from pyfive.p5t import P5VlenStringType, P5ReferenceType, P5SequenceType
from pyfive.utilities import HDF5OffsetWrapper, MetadataBufferingWrapper

logger = logging.getLogger(__name__)

FileHandle = BinaryIO | MetadataBufferingWrapper | HDF5OffsetWrapper


class Group(Mapping):
    """
    An HDF5 Group which may hold attributes, datasets, or other groups.

    Attributes
    ----------
    attrs : dict
        Attributes for this group.
    name : str
        Full path to this group.
    file : File
        File instance where this group resides.
    parent : Group
        Group instance containing this group.

    """

    def __init__(
        self,
        name: str,
        dataobjects: DataObjects,
        parent: "Group",
        max_request_block: int | None = None,
        batch_request_size: int | None = None,
    ) -> None:
        """initalize."""

        self.parent = parent
        self.file = parent.file  # type: ignore[has-type]
        self.name = name

        self._links = dataobjects.get_links()
        self._dataobjects = dataobjects
        self._attrs = None  # cached property

        self._max_request_block = max_request_block
        self._batch_request_size = batch_request_size

    def __repr__(self):
        return '<HDF5 group "%s" (%d members)>' % (self.name, len(self))

    def __len__(self):
        """Number of links in the group."""
        return len(self._links)

    def _dereference(self, ref):
        """Dereference a Reference object."""
        if not ref:
            raise ValueError("cannot deference null reference")
        obj = self.file._get_object_by_address(ref.address_of_reference)
        if obj is None:
            raise ValueError("reference not found in file")
        return obj

    def __getitem__(self, y):
        """x.__getitem__(y) <==> x[y]."""
        return self.__getitem_lazy_control(y, noindex=False)

    def get_lazy_view(self, y: object) -> object:
        """
        This instantiates the object y, and if it is a
        chunked dataset, does so without reading the b-tree
        index. This is useful for inspecting a variable
        that you are not expecting to access. If you know you
        want to access the data, and in particular, if you are
        going to hand the data to Dask or something else, you
        almost certainly want to read the index now, so
        just do x[y] rather than x.get_lazy_view(y).

        This is a ``pyfive`` extension to the standard h5py API.
        """

        return self.__getitem_lazy_control(y, noindex=True)

    def __getitem_lazy_control(self, y, noindex):
        """
        This is the routine which actually does the get item
        but does it in such a way that we control how much laziness
        is possible where we have chunked variables with b-trees.

        We want to return y, but if y is a chunked dataset we
        normally return it with a cached b-tree (noindex=false).
        If noindex is True, we do not read the b-tree, and that
        will be done when data is first read - which is fine
        in a single-threaded environment, but in a parallel
        environment you only want to read the index once
        (so use noindex=False, which you get via the
        normal getitem interface - x[y]).
        """

        if isinstance(y, Reference):
            return self._dereference(y)

        path = posixpath.normpath(y)
        if path == ".":
            return self
        if path.startswith("/"):
            return self.file[path[1:]]

        if posixpath.dirname(path) != "":
            next_obj, additional_obj = path.split("/", 1)
        else:
            next_obj = path
            additional_obj = "."

        if next_obj not in self._links:
            raise KeyError("%s not found in group" % (next_obj))

        obj_name = posixpath.join(self.name, next_obj)
        link_target = self._links[next_obj]

        if isinstance(link_target, str):
            try:
                return self.__getitem__(link_target)
            except KeyError:
                return None

        logger.info(
            f"[pyfive] Accessing object '{obj_name}' with link target {link_target} (lazy access: {noindex})"
        )
        dataobjs = self.file._get_dataobjects(link_target)
        if dataobjs.is_dataset:
            if additional_obj != ".":
                raise KeyError("%s is a dataset, not a group" % (obj_name))
            return Dataset(
                obj_name,
                DatasetID(
                    dataobjs,
                    noindex=noindex,
                    max_request_block=self._max_request_block,
                    batch_request_size=self._batch_request_size,
                    global_heaps=dataobjs._global_heaps,
                ),
                self,
            )

        try:
            # if true, this may well raise a NotImplementedError, if so, we need
            # to warn the user, who may be able to use other parts of the data.
            is_datatype = dataobjs.is_datatype
        except NotImplementedError as e:
            warnings.warn(
                f"Found datatype {obj_name} but pyfive cannot read this data: {e}"
            )
            is_datatype = True

        if is_datatype:
            return Datatype(obj_name, self.file, dataobjs.ptype)
        else:
            return Group(obj_name, dataobjs, self)[additional_obj]

    def __iter__(self):
        for k in self._links.keys():
            yield k

    def _lazy_datasets(self) -> list:
        """List of (path, dataset) for every dataset below this group, without reading any chunk index."""
        found: list = []

        def collect(name, obj):
            if isinstance(obj, Dataset):
                found.append((obj.name, obj))

        self.visititems(collect, noindex=True)
        return found

    def visit(self, func: Callable) -> object:
        """
        Recursively visit all names in the group and subgroups.

        func should be a callable with the signature:

            func(name) -> None or return value

        Returning None continues iteration, return anything else stops and
        return that value from the visit method.

        """
        return self.visititems(lambda name, obj: func(name))

    def visititems(self, func: Callable, noindex: bool = False) -> object:
        """
        Recursively visit all objects in this group and subgroups.

        func should be a callable with the signature:

            func(name, object) -> None or return value

        Returning None continues iteration, return anything else stops and
        return that value from the visit method.

        Use of the optional noindex=True will ensure that
        all operations are not only lazy wrt data, but lazy
        wrt to any chunked data indices. This keyword argument is a ``pyfive``
        extension to the standard h5py API.

        """
        root_name_length = len(self.name)
        if not self.name.endswith("/"):
            root_name_length += 1

        # Use either normal access or lazy access:
        if noindex:
            # Avoid loading dataset indices
            get_obj = self.get_lazy_view
        else:
            get_obj = self.__getitem__

        seen = {id(self._dataobjects)}  # guards against link cycles

        # Initialize queue using the correct getter
        queue = deque(get_obj(k) for k in self._links.keys())

        while queue:
            obj = queue.popleft()
            name = obj.name[root_name_length:]  # type: ignore[attr-defined]
            ret = func(name, obj)
            if ret is not None:
                return ret
            if isinstance(obj, Group):
                key = id(
                    obj._dataobjects
                )  # cached per file, so the same group has the same key
                if key in seen:
                    continue
                seen.add(key)
                # Mapping.values() would build chunk indexes even when noindex is set
                if noindex:
                    queue.extend(obj.get_lazy_view(k) for k in obj)
                else:
                    queue.extend(obj.values())
        return None

    @property
    def attrs(self):
        """attrs attribute."""
        if self._attrs is None:
            self._attrs = self._dataobjects.get_attributes()
        return self._attrs


class File(Group):
    """
    Open a HDF5 file.

    Note in addition to having file specific methods the File object also
    inherit the full interface of **Group**.

    File is also a context manager and therefore supports the with statement.
    Files opened by the class will be closed after the with block, file-like
    object are not closed.

    Parameters
    ----------
    filename : str or file-like
        Name of file (string or unicode) or file like object which has read
        and seek methods which behaved like a Python file object.
    mode : str
        File open mode (default: "r", read-only).
    metadata_buffer_size : int
        Size of metadata buffer for S3/remote files in MiB (default: 1MiB).
        Larger values reduce network calls but use more memory.
        (This is a pyfive extension for optimizing remote file access, ignored for local files.)
    decode_strings : bool (default False)
        If False, variable-length strings are returned as bytes like h5py.
        If True, variable-length strings in dataset data, fill values, and
        attributes are decoded to Python str objects.
        (This is a pyfive extension for user convenience.)

    Attributes
    ----------
    filename : str
        Name of the file on disk, None if not available.
    mode : str
        String indicating that the file is open readonly ("r").
    userblock_size : int
        Size of the user block in bytes before the HDF5 superblock.

    """

    def __init__(
        self,
        filename: str | FileHandle,
        mode: str = "r",
        metadata_buffer_size: int = 1,
        decode_strings: bool = False,
        **kwargs,
    ) -> None:
        """initalize."""

        if mode != "r":
            raise NotImplementedError(
                "pyfive only provides support for reading and treats all reads as binary"
            )
        self._close = False
        self.decode_strings = decode_strings
        if hasattr(filename, "read"):
            if not hasattr(filename, "seek"):
                raise ValueError("File like object must have a seek method")
            fh = cast(BinaryIO, filename)
            self.filename = getattr(filename, "name", "None")
        else:
            fh = open(filename, "rb")
            self._close = True
            self.filename = filename

        # Wrap S3 file handles with metadata buffering to reduce network calls
        self._fh: FileHandle
        if isinstance(fh, MetadataBufferingWrapper):
            # Already wrapped
            self._fh = fh
        elif type(fh).__name__ == "S3File" or hasattr(fh, "fs"):
            # fsspec file handle - wrap with buffering
            # We check for the S3File type by name to avoid a hard dependency on s3fs,
            # but also check for an 'fs' attribute which is common in s3fs file-like objects.
            # This may yet be too broad, but it is unlikely to cause issues for non-S3 files.
            logger.info(
                "[pyfive] Detected remote file, enabling metadata buffering (%d MB)",
                metadata_buffer_size,
            )
            self._fh = MetadataBufferingWrapper(fh, buffer_size=metadata_buffer_size)
        else:
            # Local file or other
            # NOTE mypy detects incompatible types:
            # str | BytesIO = MetadataBufferingWrapper
            self._fh = fh

        superblock_offset = self._find_superblock_offset(self._fh)
        initial_superblock = SuperBlock(self._fh, superblock_offset)
        base_address = initial_superblock._contents["base_address"]
        if base_address:
            self._fh = HDF5OffsetWrapper(self._fh, base_address)
            logical_superblock_offset = superblock_offset - base_address
        else:
            logical_superblock_offset = superblock_offset
        self._superblock = SuperBlock(self._fh, logical_superblock_offset)
        self._dataobjects_cache: dict = {}
        self._global_heaps: dict = {}
        offset = self._superblock.offset_to_dataobjects
        dataobjects = self._get_dataobjects(offset)

        self.file = self
        self.mode = "r"
        self.userblock_size = base_address
        super(File, self).__init__("/", dataobjects, self, **kwargs)

    @staticmethod
    def _find_superblock_offset(fh: FileHandle) -> int:
        """Locate the HDF5 signature at the standard superblock offsets."""
        original_offset = fh.tell()
        try:
            fh.seek(0, os.SEEK_END)
            file_size = fh.tell()
            offset = 0

            while offset + len(FORMAT_SIGNATURE) <= file_size:
                fh.seek(offset)
                if fh.read(len(FORMAT_SIGNATURE)) == FORMAT_SIGNATURE:
                    return offset
                offset = 512 if offset == 0 else offset * 2

            # Not an HDF5 file - check if it's a classic NetCDF3 file instead,
            # so we can raise a more specific/useful error.
            fh.seek(0)
            magic = fh.read(4)
            if magic[:3] == b"CDF" and magic[3:4] in (b"\x01", b"\x02", b"\x05"):
                raise InvalidHDF5File("NetCDF3 files are not supported")
        finally:
            fh.seek(original_offset)

        raise InvalidHDF5File("Unable to locate HDF5 file signature")

    @property
    def consolidated_metadata(self) -> bool:
        """Returns True if all B-tree nodes for chunked datasets are located before the first chunk in the file."""
        return not check_datasets(
            self._lazy_datasets(), self.filename
        ).fragmented_metadata

    def __repr__(self) -> str:
        return '<HDF5 file "%s" (mode r)>' % (os.path.basename(self.filename))

    def _get_dataobjects(self, obj_addr):
        """Return cached DataObjects for an object header address."""
        cached = self._dataobjects_cache.get(obj_addr)
        if cached is not None:
            return cached
        dataobjects = DataObjects(
            self._fh,
            obj_addr,
            decode_strings=self.decode_strings,
            global_heaps=self._global_heaps,
        )
        self._dataobjects_cache[obj_addr] = dataobjects
        return dataobjects

    def _get_object_by_address(self, obj_addr: BinaryIO) -> Any | Any | None:  # type: ignore[return]
        """Return the object pointed to by a given address."""
        if self._dataobjects.offset == obj_addr:
            return self

        queue = deque([(self.name.rstrip("/"), self)])
        while queue:
            base, grp = queue.popleft()
            for name, link_addr in grp._links.items():
                full_path = f"{base}/{name}" if base else f"/{name}"
                # check address without instantiating
                if link_addr == obj_addr:
                    # return instantiated object
                    return grp[name]
                # descend only if it's a subgroup (need to instantiate minimally)
                if self._get_dataobjects(link_addr).is_group:
                    queue.append((full_path, grp[name]))

    def close(self):
        """Close the file."""
        if self._close:
            self._fh.close()

    __del__ = close

    def __enter__(self):
        return self

    def __exit__(self, exc_type, value, traceback):
        self.close()


class Dataset(ABC):
    """
    A HDF5 Dataset containing an n-dimensional array and meta-data attributes.

    Attributes
    ----------
    shape : tuple
        Dataset dimensions.
    dtype : dtype
        Dataset's type.
    size : int
        Total number of elements in the dataset.
    chunks : tuple or None
        Chunk shape, or NOne is chunked storage not used.
    compression : str or None
        Compression filter used on dataset.  None if compression is not enabled
        for this dataset.
    compression_opts : dict or None
        Options for the compression filter.
    scaleoffset : dict or None
        Setting for the HDF5 scale-offset filter, or None if scale-offset
        compression is not used for this dataset.
    shuffle : bool
        Whether the shuffle filter is applied for this dataset.
    fletcher32 : bool
        Whether the Fletcher32 checksumming is enabled for this dataset.
    fillvalue : float or None
        Value indicating uninitialized portions of the dataset. None is no fill
        values has been defined.
    dim : int
        Number of dimensions.
    dims : None
        Dimension scales.
    attrs : dict
        Attributes for this dataset.
    name : str
        Full path to this dataset.
    file : File
        File instance where this dataset resides.
    parent : Group
        Group instance containing this dataset.

    """

    # Dataset is now an ABC to allow for the possibility of other dataset types, which
    # may not support all of the same attributes or methods. Examples already include
    # the ppfive Datastet class, which does not support accessing HDF5, but does
    # support the same interface for accessing data and attributes in pp/fields files.

    def __init__(self, name: str, datasetid: DatasetID, parent: Group) -> None:
        """initalize."""
        self.parent = parent
        self.file = parent.file
        self.name = name
        self._attrs = None
        self._astype = None

        self.id = datasetid
        """ This is the DatasetID instance which provides the actual data access methods. """

        # horrible kludge for now,
        # https://github.com/NCAS-CMS/pyfive/issues/13#issuecomment-2557121461
        # we hide stuff we need here
        self._dataobjects = self.id._meta

    def __repr__(self):
        info = (os.path.basename(self.name), self.shape, self.dtype)
        return '<HDF5 dataset "%s": shape %s, type "%s">' % info

    def __getitem__(self, args):
        data = self.id.get_data(args, self.fillvalue)
        if self._astype is None:
            return data
        return data.astype(self._astype)

    def read_direct(
        self,
        array: np.ndarray,
        source_sel: None | tuple = None,
        dest_sel: None | tuple = None,
    ) -> None:
        """
        Read from a HDF5 dataset directly into a NumPy array.

        This is equivalent to dset[source_sel] = arr[dset_sel].

        Creation of intermediates is not avoided. This method if provided from
        compatibility with h5py, it is not efficient.

        """
        array[dest_sel] = self[source_sel]

    def astype(self, dtype: str) -> AstypeContext:
        """
        Return a context manager which returns data as a particular type.

        Conversion is handled by NumPy after reading extracting the data.
        """
        return AstypeContext(self, dtype)

    def len(self):
        """Return the size of the first axis."""
        return self.shape[0]

    def iter_chunks(self, sel=()):
        return self.id.iter_chunks(sel)

    @property
    def shape(self):
        """shape attribute."""
        return self.id.shape

    @property
    def maxshape(self):
        """maxshape attribute. (None for unlimited dimensions)"""
        return self.id._meta.maxshape

    @property
    def ndim(self):
        """number of dimensions."""
        return len(self.shape)

    @property
    def dtype(self):
        """dtype attribute."""
        return self.id.dtype

    @property
    def value(self):
        """alias for dataset[()]."""
        DeprecationWarning(
            "dataset.value has been deprecated. Use dataset[()] instead."
        )
        return self[()]

    @property
    def size(self):
        """size attribute."""
        return np.prod(self.shape)

    @property
    def chunks(self):
        """chunks attribute."""
        return self.id.chunks

    @property
    def compression(self):
        """compression attribute."""
        return self.id._meta.compression

    @property
    def compression_opts(self):
        """compression_opts attribute."""
        return self.id._meta.compression_opts

    @property
    def scaleoffset(self):
        """scaleoffset attribute."""
        return None  # TODO support scale-offset filter

    @property
    def shuffle(self):
        """shuffle attribute."""
        return self.id._meta.shuffle

    @property
    def fletcher32(self):
        """fletcher32 attribute."""
        return self.id._meta.fletcher32

    @property
    def fillvalue(self):
        """fillvalue attribute."""
        return self.id._meta.fillvalue

    @property
    def dims(self):
        """dims attribute."""
        return DimensionManager(self)

    @property
    def attrs(self):
        """attrs attribute."""
        return self.id._meta.attributes

    @property
    def __orthogonal_indexing__(self):
        """Flag to indicate whether indexing is orthogonal.

        In general, the flag will be `True` if:

        * The data is chunked.
        * The data is contiguous and memory mapped access is not
          available.

        """
        if self.id.chunks is not None:
            # Chunked data indexed with
            # `DatasetID._get_selection_via_chunks`
            return True

        if (
            not (
                isinstance(
                    self.id._ptype, (P5ReferenceType, P5VlenStringType, P5SequenceType)
                )
            )
            and not self.id.posix
        ):
            # Contiguous data indexed with
            # `DatasetID._get_direct_from_contiguous`
            return True

        # All other cases
        return False


class DimensionManager(Sequence):
    """Represents a collection of dimensions associated with a dataset."""

    def __init__(self, dset):
        ndim = len(dset.shape)
        dim_list = [[]] * ndim
        if "DIMENSION_LIST" in dset.attrs:
            dim_list = dset.attrs["DIMENSION_LIST"]
        dim_labels = [b""] * ndim
        if "DIMENSION_LABELS" in dset.attrs:
            dim_labels = dset.attrs["DIMENSION_LABELS"]
        self._dims = [
            DimensionProxy(dset, dset.file, label, refs, axis)
            for axis, (label, refs) in enumerate(zip(dim_labels, dim_list))
        ]

    def __len__(self):
        return len(self._dims)

    def __getitem__(self, x):
        return self._dims[x]


class DimensionProxy(Sequence):
    """Represents a HDF5 "dimension"."""

    def __init__(self, dset, dset_file, label, refs, axis):
        try:
            # decode a byte string
            label = label.decode("utf-8")
        except AttributeError:
            # str doesn't have a decode method
            pass

        self._dset = dset
        self.label = label
        self._refs = refs
        self._file = dset_file
        self._axis = axis

    def __len__(self):
        return len(self._refs)

    def __getitem__(self, x):
        dscale = self._file[self._refs[x]]
        dscale_shape = dscale.shape
        dset_shape = self._dset.shape

        # If a dimension scale size does not match this dataset axis size,
        # present an alias (for example time_counter_1) to expose corruption.
        if dscale_shape and self._axis < len(dset_shape):
            axis_size = dset_shape[self._axis]
            scale_size = dscale_shape[0]
            if axis_size != scale_size:
                alias_name = _append_dimension_suffix(dscale.name, scale_size)
                warnings.warn(
                    (
                        f"Dimension scale '{os.path.basename(dscale.name)}' has size {scale_size}, "
                        f"but dataset '{os.path.basename(self._dset.name)}' axis {self._axis} "
                        f"has size {axis_size}. Using alias '{os.path.basename(alias_name)}'."
                    ),
                    UserWarning,
                    stacklevel=2,
                )
                return _AliasedDimensionScale(dscale, alias_name)

        return dscale


def _append_dimension_suffix(path, suffix):
    """Append a suffix to the final component of an HDF5 path."""
    if "/" in path:
        prefix, base = path.rsplit("/", 1)
        return f"{prefix}/{base}_{suffix}"
    return f"{path}_{suffix}"


class _AliasedDimensionScale:
    """Proxy around a dimension scale dataset with an aliased name."""

    def __init__(self, dset, alias_name):
        self._dset = dset
        self.name = alias_name

    def __repr__(self):
        info = (os.path.basename(self.name), self.shape, self.dtype)
        return '<HDF5 dataset "%s": shape %s, type "%s">' % info

    def __getitem__(self, args):
        return self._dset[args]

    def __getattr__(self, attr):
        return getattr(self._dset, attr)


class AstypeContext(object):
    """
    Context manager which allows changing the type read from a dataset.
    """

    # FIXME:ENUM should this allow a conversion from enum base types to values using dictionary?
    # Probably not, as it would be additional functionality to the h5py interface???

    def __init__(self, dset, dtype):
        self._dset = dset
        self._dtype = np.dtype(dtype)

    def __enter__(self):
        self._dset._astype = self._dtype

    def __exit__(self, *args):
        self._dset._astype = None
