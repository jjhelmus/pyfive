import numpy as np
from collections import namedtuple
from .indexing import OrthogonalIndexer, ZarrArrayStub
from .btree import BTreeV1RawDataChunks

StoreInfo = namedtuple('StoreInfo',"chunk_offset filter_mask byte_offset size")

class H5Dataset:
    """ 
    Represents an HDF5 dataset identifier.
    
    Also, many H5D* functions which take a dataset instance as their first argument 
    are presented as methods of this class. This is a subset of those supported
    by H5Py's module H5D.
    
    """
    def __init__(self, dataobject):
        """ 
        Instantiated with the pyfive datasetdataobject
        """
        self.parent_object = dataobject

        # Should we read this at instantiation?
        # I figure yes, given folks will likely only
        # go this low if they want to manipulate chunks
        # Otherwise we'd have to instantiate it as None and
        # call the build routine on every chunk manipulation.
        # Even if that's just a return, it's a lot of empty function calls
        # on an iteration over chunks.
        self.index  = self.__build_index()

    def __hash__(self):
        """ 
        H5py says this is hasable, we haven't implemented that.
        """
        raise NotImplementedError
        
    def __eq__(self, other):
        """
        H5Py says that equality is determined by true HDF5 identity.
        """
        # We kick that upstairs. 
        return self.parent_object == other.parent_object

    @property
    def shape(self):
        return self.parent_object.shape
    @property
    def rank(self):
        return self.parent_object.rank
    @property
    def dtype(self):
        return np.dtype(self.parent_object.dtype)
    
    def get_chunk_info(self, index):
        """
        Retrieve storage information about a chunk specified by its index.
        """
        return self.index[self._nthindex[index]]

    def get_chunk_info_by_coord(self, coordinate_index):
        return self.index(coordinate_index)
    
    def read_direct_chunk(self, chunk_position, **kwargs):
        """
        Returns a tuple containing the filter_mask and the raw data storing this chunk as bytes.
        Additional arugments supported by H5Py are not supported here.
        """
        storeinfo = self.index[chunk_position]
        return storeinfo.filter_mask, self._get_raw_chunk(storeinfo)
        
    ######
    # The following H5Dataset methods are used by PyFive and you wouldn't expect
    # third parties to use them. They are not H5Py methods.
    ######

    def __build_index(self):
        """ 
        Build the chunk index if it doesn't exist
        """
        if self.index is not None: 
            return
        chunk_btree = BTreeV1RawDataChunks(
                self.fh, self._chunk_address, self._chunk_dims)
        count = np.prod(self.shape)
        itemsize = np.dtype(self.dtype).itemsize
        
        self.index = {}

        # we do this to avoid either using an iterator or many 
        # temporary list creations if there are repeated chunk accesses.
        self._nthindex = []
        
        # The zarr orthogonal indexer returns the position in chunk
        # space, whereas pyfive wants the position in array space.
        # Here we index the pyfive chunk_index in zarr index space.
    
        ichunks = [1/c for c in self.chunks]
        
        for node in chunk_btree.all_nodes[0]:
            for node_key, addr in zip(node['keys'], node['addresses']):
                key = tuple([int(i*d) for i,d in zip(list(start),ichunks)])
                size = node_key['chunk_size']
                filter_mask = node_key['filter_mask']
                start = node_key['chunk_offset'][:-1]
                self._nthindex.append(key)
                self.index[key] = StoreInfo(key, filter_mask, start, size)

    def _iter_chunks(self, sel=None):
        """
        Provides internal support for iter_chunks method on parent.
        Errors should be trapped there. 
        """

        if sel is None:
            yield from self.index.values()
        else:
            raise NotImplementedError
    
    def _get_raw_chunk(self, storeinfo):
        """ 
        Obtain the bytes associated with a chunk.
        """

        self.parent_object.fh.seek(storeinfo.byte_offset)
        return self.parent_object.fh.read(storeinfo.size)  

    def _get_selection_via_chunks(self, args):
        """
        Use the zarr orthogonal indexer to extract data for a specfic selection within
        the dataset array and in doing so, only load the relevant chunks.
        """

        array = ZarrArrayStub(self.shape, self.chunks)
        indexer = OrthogonalIndexer(args, array) 
        out_shape = indexer.shape
        out = np.empty(out_shape, dtype=self.dtype, order=self.order)

        for chunk_coords, chunk_selection, out_selection in indexer:
            chunk_info = self.get_chunk_info_by_coord(chunk_coords)
            filter_mask, chunk_buffer = self.read_direct_chunk(chunk_coords.chunk_offset)
            if self.filter_pipeline is not None:
                chunk_buffer = BTreeV1RawDataChunks._filter_chunk(chunk_buffer, filter_mask, self.filter_pipeline, self.itemsize)
            chunk_buffer = self._unpack_chunk(chunk_buffer, chunk_info)
            chunk_data = np.frombuffer(chunk_buffer, dtype=self.dtype)
            out[out_selection] = chunk_data.reshape(self.chunks, order=self.order)[chunk_selection]

        return out
