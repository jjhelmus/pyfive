from .dataobjects import DataObjects, DATA_STORAGE_MSG_TYPE
from .datatype_msg import DatatypeMessage
import numpy as np
from .btree import BTreeV1RawDataChunks

class ADataObjects(DataObjects):
    """ 
    Subclass of DataObjets which access the chunk addresses for a given slice of data
    """
    def __init__(self,*args,**kwargs):
        """
        Initialise via super class
        """
        super().__init__(*args,**kwargs)

        # not yet sure we need our own copy
        self._as_chunk_index=[]

    def get_offset_addresses(self, args=None):
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
            return self._as_get_chunk_addresses(args)
    

    def _as_get_chunk_addresses(self, args):
        """ 
        Get the offset addresses associated with all the chunks 
        known to the b-tree of this object
        """
        self._get_chunk_params()
        
        if self._as_chunk_index == []:
            chunk_btree = BTreeV1RawDataChunks(
                self.fh, self._chunk_address, self._chunk_dims)

            count = np.prod(self.shape)
            itemsize = np.dtype(self.dtype).itemsize
            chunk_buffer_size = count * itemsize
            
            for node in chunk_btree.all_nodes[0]:
                for node_key, addr in zip(node['keys'], node['addresses']):
                    size = chunk_buffer_size
                    if self.filter_pipeline:
                        size = node_key['chunk_size']
                    start = node_key['chunk_offset'][:-1]
                    region = [slice(i, i+j) for i, j in zip(start, self.shape)]
                    self._as_chunk_index.append([region, start, size])

        if args is not None:
            return NotImplementedError
        return self._as_chunk_index



