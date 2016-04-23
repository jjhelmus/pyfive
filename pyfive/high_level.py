""" High-level classes for reading HDF5 files.  """

from .low_level import SuperBlock, BTree, Heap, SymbolTable, DataObjects


class Group(object):
    """

    Attributes
    ----------
    * Public
    name
    datasets
    groups

    * Private?
    btree
    heap
    symboltables
    _dataobjects
    _fh

    """

    def __init__(self, name, offset, fh):

        # read the group data objects
        fh.seek(offset)
        dataobjects = DataObjects(fh)

        btree_address, heap_address = dataobjects.get_btree_heap_addresses()

        self.name = name
        self._dataobjects = dataobjects
        self._fh = fh

        self._fh.seek(btree_address)
        self.btree = BTree(self._fh)

        self._fh.seek(heap_address)
        self.heap = Heap(self._fh)

        self.symboltables = []
        self._dataset_offsets = {}
        self._group_offsets = {}
        for symbol_table_addreess in self.btree.symbol_table_addresses():
            self._fh.seek(symbol_table_addreess)
            table = SymbolTable(self._fh)
            table.assign_name(self.heap)

            self.symboltables.append(table)
            self._dataset_offsets.update(table.find_datasets())
            self._group_offsets.update(table.find_groups())

        self._datasets = None
        self._groups = None
        self._attrs = None

    @property
    def datasets(self):
        if self._datasets is None:
            self._datasets = {k: Dataset(k, v, self._fh) for k, v in
                              self._dataset_offsets.items()}
        return self._datasets

    @property
    def groups(self):
        if self._groups is None:
            self._groups = {k: Group(k, v, self._fh) for k, v in
                            self._group_offsets.items()}
        return self._groups

    @property
    def attrs(self):
        if self._attrs is None:
            self._attrs = self._dataobjects.get_attributes()
        return self._attrs


class HDF5File(Group):
    """
    Class for reading data from HDF5 files.

    """

    def __init__(self, filename):
        """ initalize. """

        fh = open(filename, 'rb')
        self.superblock = SuperBlock(fh)
        sym_table = SymbolTable(fh, root=True)
        super(HDF5File, self).__init__(None, sym_table.group_offset, fh)

    def close(self):
        """ Close the file. """
        self._fh.close()


class Dataset(object):
    """
    Class providing access to attribute and data stored in a HDF5 Dataset.

    Parameters
    ----------

    """

    def __init__(self, name, offset, fh):
        """ initalize with fh position at the Data Object Header. """
        self.name = name
        fh.seek(offset)
        self._dataobjects = DataObjects(fh)
        self._attrs = None

    @property
    def attrs(self):
        if self._attrs is None:
            self._attrs = self._dataobjects.get_attributes()
        return self._attrs

    @property
    def data(self):
        return self._dataobjects.get_data()
