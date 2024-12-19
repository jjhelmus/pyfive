import numpy as np
from collections import namedtuple
from operator import mul
from pyfive.indexing import OrthogonalIndexer, ZarrArrayStub
from pyfive.btree import BTreeV1RawDataChunks
from pyfive.core import Reference

StoreInfo = namedtuple('StoreInfo',"chunk_offset filter_mask byte_offset size")

class H5Dataset:
    """ 
    Represents an HDF5 dataset identifier.
    
    Also, many H5D* functions which take a dataset instance as their first argument 
    are presented as methods of this class. This is a subset of those supported
    by H5Py's module H5D, but includes all the low level methods for working with 
    chunked data, lazily or not. This class has been deliberately implemented in
    such as way so that once you have an instance, it is completely independent
    of the parent file, and it can be used efficiently in threads without rereading
    the btree etc.
    """
    def __init__(self, dataobject):
        """ 
        Instantiated with the pyfive datasetdataobject, we copy and cache everything 
        we want so it can be used after the parent file is closed, without needing 
        to go back to storage.
        """
        self._chunks = dataobject.chunks
        self._order = dataobject.order
        self._filename = dataobject.fh.name
        self.filter_pipeline = dataobject.filter_pipeline
        self.shape = dataobject.shape
        self.rank = len(self.shape)
        self._msg_offset = dataobject.msg_offset
        self._unique = (self._filename, self.shape, self._msg_offset)

        if dataobject.dtype == ('REFERENCE', 8):
            # this may not behave the same as h5py, do we care? #FIXME
            self.dtype = dataobject.dtype
        else:
            self.dtype = np.dtype(dataobject.dtype)

        self.index =  None
        
        # This reads the b-tree and caches it in a form suitable for use with
        # the zarr indexer we use to lazily get chunks.

        self.__build_index(dataobject)

    def __hash__(self):
        """ The hash is based on assuming the file path, the location
        of the data in the file, and the data shape are a unique
        combination.
        """
        return hash(self.unique)
        
    def __eq__(self, other):
        """
        Equality is based on the filename, location of the data in the file
        and the shape of the data.
        """
        return self._unique == other._unique
    

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

    def __build_index(self, dataobject):
        """ 
        Build the chunk index if it doesn't exist
        """
        
        if self.index is not None: 
            return
        
        chunk_btree = BTreeV1RawDataChunks(
                dataobject.fh, dataobject._chunk_address, dataobject._chunk_dims)
        
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
                    return slice(aslice.start, aslice.stop, 1)
                return aslice
            return tuple([convert_slice(a) for a in tuple_of_slices])
    
        array = ZarrArrayStub(self.shape, self._chunks)
        indexer = OrthogonalIndexer(args, array) 
        for _, _, out_selection in indexer:
            yield convert_selection(out_selection)
    
    def _get_raw_chunk(self, storeinfo):
        """ 
        Obtain the bytes associated with a chunk.
        """
        with open(self._filename,'rb') as open_file:
            open_file.seek(storeinfo.byte_offset)
            return open_file.read(storeinfo.size) 

    def _get_selection_via_chunks(self, args):
        """
        Use the zarr orthogonal indexer to extract data for a specfic selection within
        the dataset array and in doing so, only load the relevant chunks.
        """
        # need a local dtype as we may override it for a reference read.
        dtype = self.dtype

        if isinstance(self.dtype, tuple): 
            # this is a reference and we're returning that
            true_dtype = tuple(dtype)
            dtype_class = dtype[0]
            if dtype_class == 'REFERENCE':
                size = dtype[1]
                if size != 8:
                    raise NotImplementedError('Unsupported Reference type')
                dtype = '<u8'
            else:
                raise NotImplementedError('datatype not implemented')
        else:
            true_dtype = None

        array = ZarrArrayStub(self.shape, self._chunks)
        indexer = OrthogonalIndexer(args, array) 
        out_shape = indexer.shape
        out = np.empty(out_shape, dtype=dtype, order=self._order)

        for chunk_coords, chunk_selection, out_selection in indexer:
            chunk_coords = tuple(map(mul, chunk_coords, self._chunks))
            filter_mask, chunk_buffer = self.read_direct_chunk(chunk_coords)
            if self.filter_pipeline is not None:
                chunk_buffer = BTreeV1RawDataChunks._filter_chunk(chunk_buffer, filter_mask, self.filter_pipeline, self.dtype.itemsize)
            chunk_data = np.frombuffer(chunk_buffer, dtype=dtype)
            out[out_selection] = chunk_data.reshape(self._chunks, order=self._order)[chunk_selection]
       
        if true_dtype is not None:
            # no idea if this is going to work!
            if dtype_class == 'REFERENCE':
                to_reference = np.vectorize(Reference)
                out = to_reference(out)
            else:
                raise NotImplementedError('datatype not implemented')

        return out