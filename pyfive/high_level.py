""" High-level classes for reading HDF5 files.  """

from .low_level import SuperBlock, BTree, Heap, SymbolTable, DataObjects


class Group(object):
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

    def __init__(self, name, offset, fh):
        """ initalize. """

        fh.seek(offset)
        dataobjects = DataObjects(fh)
        btree_address, heap_address = dataobjects.get_btree_heap_addresses()

        fh.seek(btree_address)
        btree = BTree(fh)

        fh.seek(heap_address)
        heap = Heap(fh)

        symboltables = []
        dataset_offsets = {}
        group_offsets = {}
        for symbol_table_addreess in btree.symbol_table_addresses():
            fh.seek(symbol_table_addreess)
            table = SymbolTable(fh)
            table.assign_name(heap)
            dataset_offsets.update(table.find_datasets())
            group_offsets.update(table.find_groups())
            symboltables.append(table)

        # required
        self.name = name
        self._dataset_offsets = dataset_offsets
        self._group_offsets = group_offsets

        # low-level objects stored for debugging
        self._fh = fh
        self._dataobjects = dataobjects
        self._btree = btree
        self._heap = heap
        self._symboltables = symboltables

        # cached properties
        self._datasets = None
        self._groups = None
        self._attrs = None

    @property
    def datasets(self):
        """ Dictionary of datasets in group. """
        if self._datasets is None:
            self._datasets = {k: Dataset(k, v, self._fh) for k, v in
                              self._dataset_offsets.items()}
        return self._datasets

    @property
    def groups(self):
        """ Dictionary of sub-groups in the group. """
        if self._groups is None:
            self._groups = {k: Group(k, v, self._fh) for k, v in
                            self._group_offsets.items()}
        return self._groups

    @property
    def attrs(self):
        """ Dictionary of attribute in the group. """
        if self._attrs is None:
            self._attrs = self._dataobjects.get_attributes()
        return self._attrs


class HDF5File(Group):
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
        fh = open(filename, 'rb')
        self._superblock = SuperBlock(fh)
        sym_table = SymbolTable(fh, root=True)
        super(HDF5File, self).__init__(None, sym_table.group_offset, fh)

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

    def __init__(self, name, offset, fh):
        """ initalize with fh position at the Data Object Header. """
        self.name = name
        fh.seek(offset)
        self._dataobjects = DataObjects(fh)
        self._attrs = None

    @property
    def attrs(self):
        """ Dictionary of attribute associated with the dataset. """
        if self._attrs is None:
            self._attrs = self._dataobjects.get_attributes()
        return self._attrs

    @property
    def data(self):
        """ N-dimensional array of data in the dataset. """
        return self._dataobjects.get_data()
