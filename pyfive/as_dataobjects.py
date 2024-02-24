from .dataobjects import DataObjects, DATA_STORAGE_MSG_TYPE
from .datatype_msg import DatatypeMessage
import numpy as np
from .btree import BTreeV1RawDataChunks
from .indexing import OrthogonalIndexer


class ZarrArrayStub:
    """ 
    This mimics the funcationality of the zarr array produced by kerchunk,
    but with only what is needed for indexing
    """
    def __init__(self, shape, chunks):
        self._chunks = list(chunks)
        self._shape = list(shape)


class ADataObjects(DataObjects):
    """ 
    Subclass of DataObjets which access the chunk addresses for a given slice of data
    """
    def __init__(self,*args,**kwargs):
        """
        Initialise via super class
        """
        super().__init__(*args,**kwargs)

        #  Need our own copy for now to utilise the zarr indexer.
        #  An optimisation could be to modify what is returned from OrthogonalIndexer
        self._zchunk_index={}

        self.order='C'

    def get_offset_addresses(self):
        """ 
        Get the offset addresses for the data requested
        """

        # offset and size from data storage message
        msg = self.find_msg_type(DATA_STORAGE_MSG_TYPE)[0]
        msg_offset = msg['offset_to_message']
        version, dims, layout_class, property_offset = (
            self._get_data_message_properties(msg_offset))

        if layout_class == 0:  # compact storage
            raise NotImplementedError("Compact storage")
        elif layout_class == 1:  # contiguous storage
            return NotImplementedError("Contiguous storage")
        if layout_class == 2:  # chunked storage
            return self._as_get_chunk_addresses()
    

    def _as_get_chunk_addresses(self):
        """ 
        Get the offset addresses associated with all the chunks 
        known to the b-tree of this object
        """
        if self._zchunk_index == {}:

            self._get_chunk_params()

            chunk_btree = BTreeV1RawDataChunks(
                self.fh, self._chunk_address, self._chunk_dims)

            count = np.prod(self.shape)
            itemsize = np.dtype(self.dtype).itemsize
            chunk_buffer_size = count * itemsize

            # The zarr orthogonal indexer returns the position in chunk
            # space, whereas pyfive wants the position in array space.
            # Here we index the pyfive chunk_index in zarr index space.
        
            ichunks = [1/c for c in self.chunks]
            
            for node in chunk_btree.all_nodes[0]:
                for node_key, addr in zip(node['keys'], node['addresses']):
                    size = chunk_buffer_size
                    if self.filter_pipeline:
                        size = node_key['chunk_size']
                    start = node_key['chunk_offset'][:-1]
                    key = tuple([int(i*d) for i,d in zip(list(start),ichunks)])
                    self._zchunk_index[key] = (addr,size)

    def __getitem__(self, args):

        if self._zchunk_index == {}:
            self._as_get_chunk_addresses()
            print("Loaded addresses for ", len(self._zchunk_index),' chunks')

        array = ZarrArrayStub(self.shape, self.chunks)

        indexer = OrthogonalIndexer(args, array)
        stripped_indexer = [(a, b, c) for a,b,c in indexer]

        filter_pipeline=None #FIXME, needs to be an argument or grabbed from somewhere
        count = np.prod(self.chunks)
        itemsize = np.dtype(self.dtype).itemsize
        default_chunk_buffer_size = itemsize*count
    
        out_shape = indexer.shape
        out = np.empty(out_shape, dtype=self.dtype, order=self.order)

        for chunk_coords, chunk_selection, out_selection in stripped_indexer:
            addr, chunk_buffer_size = self._zchunk_index[chunk_coords] 
            self.fh.seek(addr)
            if filter_pipeline is None:
                chunk_buffer = self.fh.read(default_chunk_buffer_size)
            else:
                raise NotImplementedError
                # The plan here would be to take the _filter_chunk method from BTree1RawDataChunks
                # pop it out on it's own and make it a class method here as well as wherever else it needs to be
            chunk_data = np.frombuffer(chunk_buffer, dtype=self.dtype)
            out[out_selection] = chunk_data.reshape(self.chunks, order=self.order)[chunk_selection]

        return out
        



