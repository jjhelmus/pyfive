import numpy as np
from collections import namedtuple
from operator import mul
from pyfive.indexing import OrthogonalIndexer, ZarrArrayStub
from pyfive.btree import BTreeV1RawDataChunks
from pyfive.core import Reference, UNDEFINED_ADDRESS
from io import UnsupportedOperation
import struct
import logging
from importlib.metadata import version

StoreInfo = namedtuple('StoreInfo',"chunk_offset filter_mask byte_offset size")

class DatasetID:
    """ 
    Represents an HDF5 dataset identifier.
    
    Also, many H5D* functions which take a dataset instance as their first argument 
    are presented as methods of this class. This is a subset of those supported
    by H5Py's module H5D, but includes all the low level methods for working with 
    chunked data, lazily or not. This class has been deliberately implemented in
    such as way so as to cache all the relevant metadata, so that once you have an 
    instance, it is completely independent of the parent file, and it can be used 
    efficiently in distributed threads without thread contention to the b-tree etc.
    """
    def __init__(self, dataobject):
        """ 
        Instantiated with the pyfive datasetdataobject, we copy and cache everything 
        we want so it can be used after the parent file is closed, without needing 
        to go back to storage.
        """

        self._order = dataobject.order
        self._filename = dataobject.fh.name
        self.filter_pipeline = dataobject.filter_pipeline
        self.shape = dataobject.shape
        self.rank = len(self.shape)
        self.chunks = dataobject.chunks
        
        self._msg_offset, self.layout_class,self.property_offset = dataobject.get_id_storage_params()
        self._unique = (self._filename, self.shape, self._msg_offset)

        try:
            dataobject.fh.fileno()
            self.avoid_mmap = False
        except (AttributeError,OSError):
            # not a posix file on a posix filesystem
            self.avoid_mmap = True

        if isinstance(dataobject.dtype,tuple):
            # this may not behave the same as h5py, do we care? #FIXME
            self.dtype = dataobject.dtype
        else:
            self.dtype = np.dtype(dataobject.dtype)

        self._meta = DatasetMeta(dataobject)

        self._index =  None
        match self.layout_class:
            case 0:  #compact storage
                raise NotImplementedError("Compact Storage")
            case 1:  # contiguous storage
                self.data_offset, = struct.unpack_from('<Q', dataobject.msg_data, self.property_offset)
            case 2:  # chunked storage
                self._build_index(dataobject)

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
        return self._index[self._nthindex[index]]

    def get_chunk_info_by_coord(self, coordinate_index):
        """
        Retrieve information about a chunk specified by the array address of the chunk’s 
        first element in each dimension.
        """
        return self._index[coordinate_index]
    
    def get_num_chunks(self):
        """ 
        Return total number of chunks in dataset
        """
        return len(self._index)
    
    def read_direct_chunk(self, chunk_position, **kwargs):
        """
        Returns a tuple containing the filter_mask and the raw data storing this chunk as bytes.
        Additional arugments supported by H5Py are not supported here.
        """
        if chunk_position not in self._index:
            raise OSError("Chunk coordinates must lie on chunk boundaries")
        storeinfo = self._index[chunk_position]
        return storeinfo.filter_mask, self._get_raw_chunk(storeinfo)
    
    def get_data(self, args):
        """ Called by the dataset getitem method """

        match self.layout_class:
            case 0:  #compact storage
                raise NotImplementedError("Compact Storage")
            case 1:  # contiguous storage
                return self._get_contiguous_data(args)
            case 2:  # chunked storage
                if isinstance(self.dtype, tuple):
                # references need to read all the chunks for now
                    return self._get_selection_via_chunks(())[args]
                else:
                    # this is lazily reading only the chunks we need
                    return self._get_selection_via_chunks(args)
                
    def iter_chunks(self, args):
        """ 
        Iterate over chunks in a chunked dataset. 
        The optional sel argument is a slice or tuple of slices that defines the region to be used. 
        If not set, the entire dataspace will be used for the iterator.
        For each chunk within the given region, the iterator yields a tuple of slices that gives the
        intersection of the given chunk  with the selection area. 
        This can be used to read data in that chunk.
        """
        if self.chunks is None:
            raise TypeError('Dataset is not chunked')
        
        def convert_selection(tuple_of_slices):
            # while a slice of the form slice(a,b,None) is equivalent
            # in funtion to a slice of form (a,b,1) it is not the same.
            # For compatability I've gone for "the same"
            def convert_slice(aslice):
                if aslice.step is None:
                    return slice(aslice.start, aslice.stop, 1)
                return aslice
            return tuple([convert_slice(a) for a in tuple_of_slices])
    
        array = ZarrArrayStub(self.shape, self.chunks)
        indexer = OrthogonalIndexer(args, array) 
        for _, _, out_selection in indexer:
            yield convert_selection(out_selection)

    ##### The following property is made available to support ActiveStorage
    ##### and to help those who may want to generate kerchunk indices and
    ##### bypass the iterator methods.
    @property
    def index(self):
        """ Direct access to the chunk index, if there is one."""
        if self._index is None:
            raise ValueError('No chunk index available for HDF layout class {self.layout}')
        else:
            return self._index
        
    ######
    # The following DatasetID methods are used by PyFive and you wouldn't expect
    # third parties to use them. They are not H5Py methods.
    ######

    def _build_index(self, dataobject):
        """ 
        Build the chunk index if it doesn't exist. This is only 
        called for chunk data, and only when the variable is accessed.
        That is, it is not called when we an open a file, or when
        we list the variables in a file, but only when we do
        v = open_file['var_name'] where 'var_name' is chunked.

        """
        
        if self._index is not None: 
            return
        
        # look out for an empty dataset, which will have no btree
        if np.prod(self.shape) == 0:
            self._index = {}
            return
        
        logging.info(f'Building chunk index in pyfive {version("pyfive")}')
       
        chunk_btree = BTreeV1RawDataChunks(
                dataobject.fh, dataobject._chunk_address, dataobject._chunk_dims)
        
        self._index = {}
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
                self._index[key] = StoreInfo(key, filter_mask, addr, size)

    def _get_contiguous_data(self, args):
    
        if self.data_offset == UNDEFINED_ADDRESS:
            # no storage is backing array, return all zeros
            return np.zeros(self.shape, dtype=self.dtype)[args]

        if not isinstance(self.dtype, tuple):
            if self.avoid_mmap:
                return self._get_direct_from_contiguous(args)
            else:
                try:
                    with open(self._filename,'rb') as open_file:
                        # return a memory-map to the stored array
                        # I think this would mean that we only move the sub-array corresponding to result!
                        view =  np.memmap(open_file, dtype=self.dtype, mode='c',
                                    offset=self.data_offset, shape=self.shape, order=self._order)
                    result = view[args]
                    return result
                except UnsupportedOperation:
                    return self._get_direct_from_contiguous(args)
        else:
            dtype_class = self.dtype[0]
            if dtype_class == 'REFERENCE':
                size = self.dtype[1]
                if size != 8:
                    raise NotImplementedError('Unsupported Reference type - size {size}')
                with open(self._filename,'rb') as open_file:
                    ref_addresses = np.memmap(
                        open_file, dtype=('<u8'), mode='c', offset=self.data_offset,
                        shape=self.shape, order=self._order)
                    return np.array([Reference(addr) for addr in ref_addresses])[args]
            else:
                raise NotImplementedError('datatype not implemented - {dtype_class}')


    def _get_direct_from_contiguous(self, args=None):
        """
        We read the entire contiguous array, and pull out the selection (args) from that.
        This is a fallback situation if we can't use a memory map which would otherwise be lazy.
        This will normally be when we don't have a true Posix file.
        """
    
        itemsize = np.dtype(self.dtype).itemsize
        num_elements = np.prod(self.shape)
        num_bytes = num_elements*itemsize
       
        # we need it all, let's get it all (i.e. this really does read the lot)
        with open(self._filename,'rb') as open_file:
            open_file.seek(self.data_offset)
            chunk_buffer = open_file.read(num_bytes)
        chunk_data = np.frombuffer(chunk_buffer, dtype=self.dtype)
        chunk_data = chunk_data.reshape(self.shape, order=self.order)
        return chunk_data[args]

    
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
            if np.prod(self.shape) == 0:
                return np.zeros(self.shape)

        array = ZarrArrayStub(self.shape, self.chunks)
        indexer = OrthogonalIndexer(args, array) 
        out_shape = indexer.shape
        out = np.empty(out_shape, dtype=dtype, order=self._order)

        for chunk_coords, chunk_selection, out_selection in indexer:
            chunk_coords = tuple(map(mul, chunk_coords, self.chunks))
            filter_mask, chunk_buffer = self.read_direct_chunk(chunk_coords)
            if self.filter_pipeline is not None:
                chunk_buffer = BTreeV1RawDataChunks._filter_chunk(chunk_buffer, filter_mask, self.filter_pipeline, self.dtype.itemsize)
            chunk_data = np.frombuffer(chunk_buffer, dtype=dtype)
            out[out_selection] = chunk_data.reshape(self.chunks, order=self._order)[chunk_selection]
       
        if true_dtype is not None:
            # no idea if this is going to work!
            if dtype_class == 'REFERENCE':
                to_reference = np.vectorize(Reference)
                out = to_reference(out)
            else:
                raise NotImplementedError('datatype not implemented')

        return out
    

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

        #horrible kludge for now, this isn't really the same sort of thing
        #https://github.com/NCAS-CMS/pyfive/issues/13#issuecomment-2557121461
        # this is used directly in the Dataset init method.
        self.offset = dataobject.offset
