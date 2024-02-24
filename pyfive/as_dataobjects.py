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

            self.chunk_btree = BTreeV1RawDataChunks(
                self.fh, self._chunk_address, self._chunk_dims)

            count = np.prod(self.shape)
            itemsize = np.dtype(self.dtype).itemsize
            chunk_buffer_size = count * itemsize

            # The zarr orthogonal indexer returns the position in chunk
            # space, whereas pyfive wants the position in array space.
            # Here we index the pyfive chunk_index in zarr index space.
        
            ichunks = [1/c for c in self.chunks]
            
            for node in self.chunk_btree.all_nodes[0]:
                for node_key, addr in zip(node['keys'], node['addresses']):
                    size = node_key['chunk_size']
                    if self._filter_pipeline:
                        filter_mask = node_key['filter_mask']
                    else:
                        filter_mask=None
                    start = node_key['chunk_offset'][:-1]
                    key = tuple([int(i*d) for i,d in zip(list(start),ichunks)])
                    self._zchunk_index[key] = (addr,size,filter_mask)

    def __getitem__(self, args):

        if self._zchunk_index == {}:
            self._as_get_chunk_addresses()

        array = ZarrArrayStub(self.shape, self.chunks)

        indexer = OrthogonalIndexer(args, array)
        stripped_indexer = [(a, b, c) for a,b,c in indexer]

        itemsize = np.dtype(self.dtype).itemsize    
        out_shape = indexer.shape
        out = np.empty(out_shape, dtype=self.dtype, order=self.order)

        for chunk_coords, chunk_selection, out_selection in stripped_indexer:
            addr, chunk_buffer_size, filter_mask = self._zchunk_index[chunk_coords] 
            chunk_buffer = self.chunk_btree.get_one_chunk_buffer(
                addr, chunk_buffer_size, itemsize,self._filter_pipeline, filter_mask)
            chunk_data = np.frombuffer(chunk_buffer, dtype=self.dtype)
            out[out_selection] = chunk_data.reshape(self.chunks, order=self.order)[chunk_selection]

        return out
        



