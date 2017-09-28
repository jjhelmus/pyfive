""" High-level classes for reading HDF5 files.  """

from collections import Mapping, deque, Sequence
from io import open     # Python 2.7 requires for a Buffered Reader

import numpy as np

from .low_level import SuperBlock, DataObjects, Reference


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

    def __init__(self, name, dataobjects, parent, alt_file=None):
        """ initalize. """

        self.parent = parent
        if alt_file is None:
            self.file = parent.file
        else:
            self.file = alt_file
        self.name = name

        self._links = dataobjects.get_links()
        self._dataobjects = dataobjects
        self._attrs = None  # cached property

    def __len__(self):
        """ Number of links in the group. """
        return len(self._links)

    def __getitem__(self, y):
        """ x.__getitem__(y) <==> x[y] """
        if isinstance(y, Reference):
            if not y:
                raise ValueError('cannot deference null reference')
            obj = self.file._get_object_by_address(y.address_of_reference)
            if obj is None:
                dataobjects = DataObjects(
                    self.file._fh, y.address_of_reference)
                if dataobjects.is_dataset:
                    return Dataset(None, dataobjects, None, alt_file=self.file)
                else:
                    return Group(None, dataobjects, None, alt_file=self.file)
            else:
                return obj

        y = y.strip('/')

        if y not in self._links:
            raise KeyError('%s not found in group' % (y))

        if self.name == '/':
            sep = ''
        else:
            sep = '/'

        dataobjects = DataObjects(self.file._fh, self._links[y])
        if dataobjects.is_dataset:
            return Dataset(self.name + sep + y, dataobjects, self)
        else:
            return Group(self.name + sep + y, dataobjects, self)

    def __iter__(self):
        for k in self._links.keys():
            yield k

    def visit(self, func):
        """
        Recursively visit all names in the group and subgroups.

        func should be a callable with the signature:

            func(name) -> None or return value

        Returning None continues iteration, return anything else stops and
        return that value from the visit method.

        """
        raise NotImplementedError

    def visititems(self, func):
        """
        Recursively visit all objects in this group and subgroups.

        func should be a callable with the signature:

            func(name, object) -> None or return value

        Returning None continues iteration, return anything else stops and
        return that value from the visit method.

        """
        raise NotImplementedError

    @property
    def attrs(self):
        """ attrs attribute. """
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

    Attributes
    ----------
    filename : str
        Name of the file on disk, None if not available.
    mode : str
        String indicating that the file is open readonly ("r").
    userblock_size : int
        Size of the user block in bytes (currently always 0).

    """

    def __init__(self, filename):
        """ initalize. """
        if hasattr(filename, 'read'):
            if not hasattr(filename, 'seek'):
                raise ValueError(
                    'File like object must have a seek method')
            self._fh = filename
            self._close = False
            self.filename = getattr(filename, 'name', None)
        else:
            self._fh = open(filename, 'rb')
            self._close = True
            self.filename = filename
        self._superblock = SuperBlock(self._fh, 0)
        offset = self._superblock.offset_to_dataobjects
        dataobjects = DataObjects(self._fh, offset)

        self.file = self
        self.mode = 'r'
        self.userblock_size = 0
        super(File, self).__init__('/', dataobjects, self)

    def _get_object_by_address(self, obj_addr):
        """ Return the object pointed to by a given address. """
        # breadth first search of the file hierarchy for the given address
        queue = deque([self])
        while queue:
            obj = queue.popleft()
            if obj._dataobjects.offset == obj_addr:
                return obj
            if isinstance(obj, Group):
                queue.extend(child for child in obj.values())
        return None

    def close(self):
        """ Close the file. """
        if self._close:
            self._fh.close()
    __del__ = close

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()


class Dataset(object):
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

    def __init__(self, name, dataobjects, parent, alt_file=None):
        """ initalize. """
        self.parent = parent
        if alt_file is None:
            self.file = parent.file
        else:
            self.file = alt_file
        self.name = name

        self._dataobjects = dataobjects
        self._attrs = None

    def __getitem__(self, args):
        return self._dataobjects.get_data()[args]

    def read_direct(self, array, source_sel=None, dset_sel=None):
        """ Read from a HDF5 dataset directly into a NumPy array. """
        raise NotImplementedError

    def astype(self, dtype):
        """
        Return a context manager which returns data as a particular type.
        """
        raise NotImplementedError

    def len(self):
        """ Return the size of the first axis. """
        return self.shape[0]

    @property
    def shape(self):
        """ shape attribute. """
        return self._dataobjects.shape

    @property
    def ndim(self):
        """ number of dimensions. """
        return len(self.shape)

    @property
    def dtype(self):
        """ dtype attribute. """
        return self._dataobjects.dtype

    @property
    def size(self):
        """ size attribute. """
        return np.prod(self.shape)

    @property
    def chunks(self):
        """ chunks attribute. """
        return self._dataobjects.chunks

    @property
    def compression(self):
        """ compression attribute. """
        return self._dataobjects.compression

    @property
    def compression_opts(self):
        """ compression_opts attribute. """
        return self._dataobjects.compression_opts

    @property
    def scaleoffset(self):
        """ scaleoffset attribute. """
        return None  # TODO support scale-offset filter

    @property
    def shuffle(self):
        """ shuffle attribute. """
        return self._dataobjects.shuffle

    @property
    def fletcher32(self):
        """ fletcher32 attribute. """
        return self._dataobjects.fletcher32

    @property
    def fillvalue(self):
        """ fillvalue attribute. """
        return self._dataobjects.fillvalue

    @property
    def dims(self):
        """ dims attribute. """
        return DimensionManager(self)

    @property
    def attrs(self):
        """ attrs attribute. """
        if self._attrs is None:
            self._attrs = self._dataobjects.get_attributes()
        return self._attrs


class DimensionManager(Sequence):
    """ Represents a collection of dimensions associated with a dataset. """
    def __init__(self, dset):
        ndim = len(dset.shape)
        dim_list = [[]]*ndim
        if 'DIMENSION_LIST' in dset.attrs:
            dim_list = dset.attrs['DIMENSION_LIST']
        dim_labels = [b'']*ndim
        if 'DIMENSION_LABELS' in dset.attrs:
            dim_labels = dset.attrs['DIMENSION_LABELS']
        self._dims = [
            DimensionProxy(dset.file, label, refs) for
            label, refs in zip(dim_labels, dim_list)]

    def __len__(self):
        return len(self._dims)

    def __getitem__(self, x):
        return self._dims[x]


class DimensionProxy(Sequence):
    """ Represents a HDF5 "dimension". """

    def __init__(self, dset_file, label, refs):
        self.label = label.decode('utf-8')
        self._refs = refs
        self._file = dset_file

    def __len__(self):
        return len(self._refs)

    def __getitem__(self, x):
        return self._file[self._refs[x]]
