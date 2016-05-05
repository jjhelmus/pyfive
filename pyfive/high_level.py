""" High-level classes for reading HDF5 files.  """

# Requires in Python 2.7 for open to return a BufferedReader
from io import open
from collections import Mapping

import numpy as np

from .low_level import SuperBlock, DataObjects


class Group(Mapping):
    """
    An HDF5 Group which may hold attributes, datasets, or other groups.

    Attributes
    ----------
    * name : str
        Name of this group
    * attrs : dict
        Dictionary of attributes for this group.
    * datasets : dict
        Datasets which belong to this group.
    * groups : dict
        Groups which are sub-groups of this group.

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
        """ Dictionary of attribute in the group. """
        if self._attrs is None:
            self._attrs = self._dataobjects.get_attributes()
        return self._attrs


class File(Group):
    """
    Open a HDF5 file.

    Note in addition to having file specific methods the HDF5File object also
    inherit the full interface of **Group**.

    Parameters
    ----------
    filename : str
        Name of file (string or unicode).

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
    A HDF5 Dataset containing an n-dimensional array and associated meta-data
    stored as attributes

    Attributes
    ----------
    * name : str
        Name of dataset
    * attrs : dict
        Dictionary of attributes associated with the dataset
    * data : ndarray
        NumPy array containing the datasets data.

    """

    def __init__(self, name, dataobjects, parent):
        """ initalize with fh position at the Data Object Header. """
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
        """ NumPy-style shape tuple giving dataset dimensions. """
        return self._dataobjects.shape

    @property
    def dtype(self):
        """ NumPy-style dtype object giving the datasets type. """
        return self._dataobjects.dtype

    @property
    def size(self):
        """ Integer giving the total number of elements in the dataset. """
        return np.prod(self.shape)

    @property
    def chunks(self):
        return None  # TODO support chunks

    @property
    def compression(self):
        return None  # TODO support compression

    @property
    def compression_opts(self):
        return None  # TODO support compression

    @property
    def scaleoffset(self):
        return None  # TODO support scale-offset filter

    @property
    def shuffle(self):
        return False  # TODO support shuffle filter

    @property
    def fletcher32(self):
        return False  # TODO support fletcher32 checksumming

    @property
    def fillvalue(self):
        raise NotImplementedError

    @property
    def dims(self):
        raise NotImplementedError

    @property
    def attrs(self):
        """ Dictionary of attribute associated with the dataset. """
        if self._attrs is None:
            self._attrs = self._dataobjects.get_attributes()
        return self._attrs
