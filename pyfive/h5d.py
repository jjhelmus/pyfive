import numpy as np
from collections import namedtuple
from operator import mul
from pyfive.indexing import OrthogonalIndexer, ZarrArrayStub
from pyfive.btree import BTreeV1RawDataChunks
from pyfive.core import Reference, UNDEFINED_ADDRESS
from pyfive.misc_low_level import get_vlen_string_data
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
    def __init__(self, dataobject, pseudo_chunking_size_MB=4):
        """ 
        Instantiated with the pyfive datasetdataobject, we copy and cache everything 
        we want so that the only file operations are now data accesses.
        
        if pseudo_chunking_size_MB is set to a value greater than zero, and
        if the storage is not local posix (and hence np.mmap is not available) then 
        when accessing contiguous variables, we attempt to find a suitable
        chunk shape to approximate that volume and read the contigous variable
        as if were chunked. This is to facilitate lazy loading of partial data
        from contiguous storage.

        (Currently the only way to change this value is by explicitly using
        the set_pseudo_chunk_size method. Most users will not need to change 
        it.)

        """

        self._order = dataobject.order
        fh = dataobject.fh
        
        try:
            # See if 'fh' is an underlying file descriptor
            fh.fileno()
        except (AttributeError, OSError):
            #  No file descriptor => Not Posix
            self.posix = False
            self.__fh = fh
            self.pseudo_chunking_size = pseudo_chunking_size_MB*1024*1024
            try:
                # maybe this is an S3File instance?
                self._filename = getattr(fh,'path')
            except:
                # maybe a remote https file opened as bytes?
                # failing that, maybe a memory file, return as None
                self._filename = getattr(fh,'full_name','None')
        else:
            # Has a file descriptor => Posix
            self.posix = True
            self._filename = fh.name
            self.pseudo_chunking_size = 0 

        self.filter_pipeline = dataobject.filter_pipeline
        self.shape = dataobject.shape
        self.rank = len(self.shape)
        self.chunks = dataobject.chunks

        # experimental code. We need to find out whether or not this
        # is unnecessary duplication. At the moment it seems best for
        # each variable to have it's own copy of those needed for 
        # data access. Though that's clearly not optimal if they include
        # other data. To be determined.
        self._global_heaps={} 
        
        self._msg_offset, self.layout_class,self.property_offset = dataobject.get_id_storage_params()
        self._unique = (self._filename, self.shape, self._msg_offset)

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
        """
        if not self._index:
            return None
        else:
            return self._index[self._nthindex[index]]

    def get_chunk_info_by_coord(self, coordinate_index):
        """
        Retrieve information about a chunk specified by the array address of the chunk’s 
        first element in each dimension.
        """
        if not self._index:
            return None
        else:
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
        if not self.index:
            return None
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
                if self.data_offset == UNDEFINED_ADDRESS:
                    # no storage is backing array, return all zeros
                    return np.zeros(self.shape, dtype=self.dtype)[args]
                else:
                    return self._get_contiguous_data(args)
            case 2:  # chunked storage
                if not self._index:
                    return np.zeros(self.shape, dtype=self.dtype)[args]
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
        intersection of the given chunk with the selection area. 
        This can be used to read data in that chunk.
        """
        if self.chunks is None:
            raise TypeError('Dataset is not chunked')
        
        def convert_selection(tuple_of_slices):
            # while a slice of the form slice(a,b,None) is equivalent
            # in function to a slice of form (a,b,1) it is not the same.
            # For compatability I've gone for "the same"
            def convert_slice(aslice):
                if aslice.step is None:
                    return slice(aslice.start, aslice.stop, 1)
                return aslice
            return tuple([convert_slice(a) for a in tuple_of_slices])
    
        array = ZarrArrayStub(self.shape, self.chunks)

        if args:
            # convert to getitem type args
            converted = []
            for s in args:
                if isinstance(s, slice) and (s.stop - s.start) == 1:
                    converted.append(s.start)
                else:
                    converted.append(s)
            args = tuple(converted)
            indexer = OrthogonalIndexer(*args, array) 
        else:
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
    #### The following method can be used to set pseudo chunking size after the 
    #### file has been closed and before data transactions. This is pyfive specific
    def set_psuedo_chunk_size(self, newsize_MB):
        """ Set pseudo chunking size for contiguous variables. The default
        value is 4 MB which should be suitable for most applications. For
        arrays smaller than this value, no pseudo chunking is used. 
        Larger arrays will be accessed in in roughly newsize_MB reads. """
        if self.layout_class == 1:
            if not self.posix:
                self.pseudo_chunking_size = newsize_MB*1024*1024
            else:
                pass  # silently ignore it, we'll be using a np.memmap
        else:
            raise ValueError('Attempt to set pseudo chunking on non-contigous variable')

    def get_chunk_info_from_chunk_coord(self, chunk_coords):
        """
        Retrieve storage information about a chunk specified by its index.
        This index is in chunk space (as used by zarr) and needs to be converted
        to hdf5 coordinate space.  Additionaly, if this file is not chunked, the storeinfo 
        is returned for the contiguous data as if it were one chunk.
        """
        if not self._index:
            dummy =  StoreInfo(None, None, self.data_offset, self.dtype.itemsize*np.prod(self.shape))
            return dummy
        else:
            coord_index = tuple(map(mul, chunk_coords, self.chunks))
            return self.get_chunk_info_by_coord(coord_index)
        
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
        if np.prod(self.shape) == 0 or dataobject._chunk_address == UNDEFINED_ADDRESS:
            self._index = {}
            return
        
        logging.info(f'Building chunk index in pyfive {version("pyfive")}')
       
        chunk_btree = BTreeV1RawDataChunks(
                dataobject.fh, dataobject._chunk_address, dataobject._chunk_dims)
        
        self._index = {}
        self._nthindex = []
        
        for node in chunk_btree.all_nodes[0]:
            for node_key, addr in zip(node['keys'], node['addresses']):
                start = node_key['chunk_offset'][:-1]
                key = start
                size = node_key['chunk_size']
                filter_mask = node_key['filter_mask']
                self._nthindex.append(key)
                self._index[key] = StoreInfo(key, filter_mask, addr, size)

    def _get_contiguous_data(self, args):

        if not isinstance(self.dtype, tuple):
            if not self.posix:
                # Not posix
                return self._get_direct_from_contiguous(args)
            else:
                # posix
                try:
                    # Create a memory-map to the stored array, which
                    # means that we will end up only copying the
                    # sub-array into in memory.
                    fh = self._fh
                    view =  np.memmap(
                        fh,
                        dtype=self.dtype,
                        mode='c',
                        offset=self.data_offset,
                        shape=self.shape,
                        order=self._order
                    )
                    # Create the sub-array
                    result = view[args]
                    # Copy the data from disk to physical memory
                    result = result.view(type=np.ndarray)
                    fh.close()
                    return result
                except UnsupportedOperation:
                    return self._get_direct_from_contiguous(args)
        else:
            dtype_class = self.dtype[0]
            if dtype_class == 'REFERENCE':
                size = self.dtype[1]
                if size != 8:
                    raise NotImplementedError('Unsupported Reference type - size {size}')

                fh = self._fh
                ref_addresses = np.memmap(
                    fh, dtype=('<u8'), mode='c', offset=self.data_offset,
                    shape=self.shape, order=self._order)
                result = np.array([Reference(addr) for addr in ref_addresses])[args]
                if self.posix:
                    fh.close()

                return result
            elif dtype_class == 'VLEN_STRING':
                fh = self._fh
                array = get_vlen_string_data(fh, self.data_offset, self._global_heaps, self.shape, self.dtype)
                return array.reshape(self.shape, order=self._order)
            else:
                raise NotImplementedError(f'datatype not implemented - {dtype_class}')


    def _get_direct_from_contiguous(self, args=None):
        """
        This is a fallback situation if we can't use a memory map which would otherwise be lazy.
        If pseudo_chunking_size is set, we attempt to read the contiguous data in chunks
        otherwise we have to read the entire array. This is a fallback situation if we 
        can't use a memory map which would otherwise be lazy. This will normally be when 
        we don't have a true Posix file. We should never end up here with compressed
        data.
        """
        def __get_pseudo_shape():
            """ Determine an appropriate chunk and stride for a given pseudo chunk size """
            element_size = self.dtype.itemsize
            chunk_shape = np.copy(self.shape)
            while True:
                chunk_size = np.prod(chunk_shape) * element_size
                if chunk_size < self.pseudo_chunking_size:
                    break
                for i in range(len(chunk_shape)):  
                    if chunk_shape[i] > 1:
                        chunk_shape[i] //= 2
                        break
            return chunk_shape, chunk_size

        class LocalOffset:
            def __init__(self, shape, chunk_shape, stride):
                chunks_per_dim = [int(np.ceil(a / c)) for a, c in zip(shape, chunk_shape)]
                self.chunk_strides = np.cumprod([1] + chunks_per_dim[::-1])[:-1][::-1]
                self.stride = stride
            def coord_to_offset(self,chunk_coords):
                linear_offset = sum(idx * stride for idx, stride in zip(chunk_coords, self.chunk_strides))
                return linear_offset*self.stride
              
        fh = self._fh
        if self.pseudo_chunking_size:
            chunk_shape, stride = __get_pseudo_shape()
            offset_finder = LocalOffset(self.shape,chunk_shape,stride)
            array = ZarrArrayStub(self.shape, chunk_shape)
            indexer = OrthogonalIndexer(args, array)
            out_shape = indexer.shape
            out = np.empty(out_shape, dtype=self.dtype, order=self._order)
            chunk_size = np.prod(chunk_shape)

            for chunk_coords, chunk_selection, out_selection in indexer:
                index = self.data_offset + offset_finder.coord_to_offset(chunk_coords)
                fh.seek(index)
                chunk_buffer = fh.read(stride)
                chunk_data = np.frombuffer(chunk_buffer, dtype=self.dtype).copy()
                if len(chunk_data) < chunk_size:
                    # last chunk over end of file
                    padded_chunk_data = np.zeros(chunk_size, dtype=self.dtype)
                    padded_chunk_data[:len(chunk_data)] = chunk_data
                    chunk_data = padded_chunk_data
                out[out_selection] = chunk_data.reshape(chunk_shape, order=self._order)[chunk_selection]
    
            if self.posix:
                fh.close()

            return out

        else:
            itemsize = np.dtype(self.dtype).itemsize
            num_elements = np.prod(self.shape, dtype=int)
            num_bytes = num_elements*itemsize

            # we need it all, let's get it all (i.e. this really does
            # read the lot)
            fh.seek(self.data_offset)
            chunk_buffer = fh.read(num_bytes) 
            chunk_data = np.frombuffer(chunk_buffer, dtype=self.dtype).copy()
            chunk_data = chunk_data.reshape(self.shape, order=self._order)
            chunk_data = chunk_data[args]
            if self.posix:
                fh.close()

            return chunk_data
    
    def _get_raw_chunk(self, storeinfo):
        """ 
        Obtain the bytes associated with a chunk.
        """
        fh = self._fh
        fh.seek(storeinfo.byte_offset)
        out = fh.read(storeinfo.size)
        if self.posix:
            fh.close()

        return out

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
            # map from chunk coordinate space to array space which is how hdf5 keeps the index
            chunk_coords = tuple(map(mul, chunk_coords, self.chunks))
            filter_mask, chunk_buffer = self.read_direct_chunk(chunk_coords)
            if self.filter_pipeline is not None:
                # we are only using the class method here, future filter pipelines may need their own function
                chunk_buffer = BTreeV1RawDataChunks._filter_chunk(chunk_buffer, filter_mask, self.filter_pipeline, self.dtype.itemsize)
            chunk_data = np.frombuffer(chunk_buffer, dtype=dtype).copy()
            out[out_selection] = chunk_data.reshape(self.chunks, order=self._order)[chunk_selection]
       
        if true_dtype is not None:
            
            if dtype_class == 'REFERENCE':
                to_reference = np.vectorize(Reference)
                out = to_reference(out)
            else:
                raise NotImplementedError('datatype not implemented')

        return out

    @property
    def _fh(self):
        """Return an open file handle to the parent file.

        When the parent file has been closed, we will need to reopen it
        to continue to access data. This facility is provided to support
        thread safe data access. However, now the file is open outside
        a context manager, the user is responsible for closing it,
        though it should get closed when the variable instance is
        garbage collected.

        """

        if self.posix:
            # Posix: Open the file, without caching it.
            return open(self._filename, 'rb')

        # Not posix: Use the cached file if it's open, otherwise open
        #            the file and cache it.
        fh = self.__fh
        if fh.closed:
            fh = open(self._filename, 'rb')
            self.__fh = fh

        return fh



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
