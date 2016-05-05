""" High-level classes for reading HDF5 files.  """

from collections import Mapping
from io import open     # Python 2.7 requires for a Buffered Reader

import numpy as np

from .low_level import SuperBlock, DataObjects


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

    def __init__(self, name, dataobjects, parent):
        """ initalize. """

        self.parent = parent
        self.file = parent.file
        self.name = name

        self._links = dataobjects.get_links()
        self._dataobjects = dataobjects
        self._attrs = None  # cached property

    def __len__(self):
        """ Number of links in the group. """
        return len(self._links)

    def __getitem__(self, y):
        """ x.__getitem__(y) <==> x[y] """
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

    Parameters
    ----------
    filename : str
        Name of file (string or unicode).

    Attributes
    ----------
    filename : str
        Name of the file on disk.
    mode : str
        String indicating that the file is open readonly ("r").
    userblock_size : int
        Size of the user block in bytes (currently always 0).

    """

    def __init__(self, filename):
        """ initalize. """
        self._fh = open(filename, 'rb')
        self._superblock = SuperBlock(self._fh, 0)
        offset = self._superblock.offset_to_dataobjects
        dataobjects = DataObjects(self._fh, offset)

        self.filename = filename
        self.file = self
        self.mode = 'r'
        self.userblock_size = 0
        super(File, self).__init__('/', dataobjects, self)

    def close(self):
        """ Close the file. """
        self._fh.close()


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

    def __init__(self, name, dataobjects, parent):
        """ initalize. """
        self.parent = parent
        self.file = parent.file
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
        return None  # TODO support chunks

    @property
    def compression(self):
        """ compression attribute. """
        return None  # TODO support compression

    @property
    def compression_opts(self):
        """ compression_opts attribute. """
        return None  # TODO support compression

    @property
    def scaleoffset(self):
        """ scaleoffset attribute. """
        return None  # TODO support scale-offset filter

    @property
    def shuffle(self):
        """ shuffle attribute. """
        return False  # TODO support shuffle filter

    @property
    def fletcher32(self):
        """ fletcher32 attribute. """
        return False  # TODO support fletcher32 checksumming

    @property
    def fillvalue(self):
        """ fillvalue attribute. """
        raise NotImplementedError

    @property
    def dims(self):
        """ dims attribute. """
        raise NotImplementedError

    @property
    def attrs(self):
        """ attrs attribute. """
        if self._attrs is None:
            self._attrs = self._dataobjects.get_attributes()
        return self._attrs
