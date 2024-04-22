import numpy as np
from collections import namedtuple
from operator import mul
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
        self._chunks = self.parent_object.chunks
        self._order = self.parent_object.order
        self._fh = self.parent_object.fh
        self.filter_pipeline = self.parent_object.filter_pipeline

        self.index =  None
        # Should we read this at instantiation?
        # I figure yes, given folks will likely only
        # go this low if they want to manipulate chunks.
        # Otherwise we'd call the (cached) build routine on
        # each chunk manipulation. That could be a lot of
        # empty function calls, even if they are cheap cf I/O. 
        self.__build_index()

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
        # FIXME: Not sure what H5Py is doing here need to find out,
        # but I'm sure it's not this.
        if self.parent_object.dtype == ('REFERENCE',8):
            return self.parent_object.dtype
        else:
            return np.dtype(self.parent_object.dtype)
    
    def get_chunk_info(self, index):
        """
        Retrieve storage information about a chunk specified by its index.
        Our index is in chunk space, but H5Py wants it in coordinate space.
        """
        return self.index[self._nthindex[index]]

    def get_chunk_info_by_coord(self, coordinate_index):
        """
        Retrieve information about a chunk specified by the array address of the chunk’s 
        first element in each dimension.
        """
        return self.index[coordinate_index]
    
    def get_num_chunks(self):
        """ 
        Return total number of chunks in dataset
        """
        return len(self.index)
    
    def read_direct_chunk(self, chunk_position, **kwargs):
        """
        Returns a tuple containing the filter_mask and the raw data storing this chunk as bytes.
        Additional arugments supported by H5Py are not supported here.
        """
        if chunk_position not in self.index:
            raise OSError("Chunk coordinates must lie on chunk boundaries")
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
                self._fh, self.parent_object._chunk_address, self.parent_object._chunk_dims)
        
        self.index = {}
        # we do this to avoid either using an iterator or many 
        # temporary list creations if there are repeated chunk accesses.
        self._nthindex = []
        
        # The zarr orthogonal indexer returns the position in chunk
        # space, whereas pyfive wants the position in array space.
        # Here we index the pyfive chunk_index in zarr index space.
        
        for node in chunk_btree.all_nodes[0]:
            for node_key, addr in zip(node['keys'], node['addresses']):
                start = node_key['chunk_offset'][:-1]
                key = start
                size = node_key['chunk_size']
                filter_mask = node_key['filter_mask']
                self._nthindex.append(key)
                self.index[key] = StoreInfo(key, filter_mask, addr, size)


    def _iter_chunks(self, args):
        """
        Provides internal support for iter_chunks method on parent.
        Errors should be trapped there. 
        """
        def convert_selection(tuple_of_slices):
            # while a slice of the form slice(a,b,None) is equivalent
            # in funtion to a slice of form (a,b,1) it is not the same.
            # For compatability I've gone for "the same"
            def convert_slice(aslice):
                if aslice.step is None:
                    return slice(aslice.start,aslice.stop,1)
                return aslice
            return tuple([convert_slice(a) for a in tuple_of_slices])
    
        array = ZarrArrayStub(self.shape, self._chunks)
        indexer = OrthogonalIndexer(args, array) 
        for chunk_coords, chunk_selection, out_selection in indexer:
            yield convert_selection(out_selection)
    
    def _get_raw_chunk(self, storeinfo):
        """ 
        Obtain the bytes associated with a chunk.
        """

        self._fh.seek(storeinfo.byte_offset)
        return self._fh.read(storeinfo.size) 

    def _get_reference_chunks(self, offset):
        """ 
        Return reference data which is chunked. At the moment
        we re-read the b-tree to do this, since we didn't cache
        it at index construction. #FIXME
        """
        chunk_btree = BTreeV1RawDataChunks(
            self._fh, self.parent_object._chunk_address, self.parent_object._chunk_dims)
        return chunk_btree.construct_data_from_chunks(
            self._chunks, self.shape, self.dtype, self.filter_pipeline) 

    def _get_selection_via_chunks(self, args):
        """
        Use the zarr orthogonal indexer to extract data for a specfic selection within
        the dataset array and in doing so, only load the relevant chunks.
        """

        array = ZarrArrayStub(self.shape, self._chunks)
        indexer = OrthogonalIndexer(args, array) 
        out_shape = indexer.shape
        out = np.empty(out_shape, dtype=self.dtype, order=self._order)

        for chunk_coords, chunk_selection, out_selection in indexer:
            chunk_coords = tuple(map(mul,chunk_coords,self._chunks))
            filter_mask, chunk_buffer = self.read_direct_chunk(chunk_coords)
            if self.filter_pipeline is not None:
                chunk_buffer = BTreeV1RawDataChunks._filter_chunk(chunk_buffer, filter_mask, self.filter_pipeline, self.dtype.itemsize)
            chunk_data = np.frombuffer(chunk_buffer, dtype=self.dtype)
            out[out_selection] = chunk_data.reshape(self._chunks, order=self._order)[chunk_selection]

        return out