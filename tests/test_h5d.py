import h5py
import pyfive
from pathlib import Path
import pytest

mypath = Path(__file__).parent

filename = 'compressed.hdf5'
variable_name = 'dataset3'

def chunk_down(ff, vv):
    """ 
    Test the chunking stuff
    """
    var = ff[vv]
    varid = var.id
    n = varid.get_num_chunks()
    c = varid.get_chunk_info(4)
    with pytest.raises(OSError):
        # This isn't on the chunk boundary, so should fail
        address = (2,0)
        d = varid.read_direct_chunk(address)
    address = c.chunk_offset
    d = varid.read_direct_chunk(address)
    return n, c.chunk_offset, c.filter_mask, c.byte_offset, c.size, d


def get_chunks(ff, vv, view=3):
    var = ff[vv]
    x = var[:,2]
    y = var[:,:]
    chunks = list(var.iter_chunks())
    for i in range(view):
        print('Chunk ',i)
        print(chunks[i])
    return list(var.iter_chunks())


def test_h5d_chunking_details():

    with h5py.File(mypath/filename) as f:
        h5detail = chunk_down(f, variable_name)

    with pyfive.File(mypath/filename) as g:
        p5detail = chunk_down(g, variable_name)

    assert h5detail == p5detail 

def test_iter_chunks():

    with h5py.File(mypath/filename) as f:
        h5chunks = get_chunks(f, variable_name)

    with pyfive.File(mypath/filename) as g:
        p5chunks = get_chunks(f, variable_name)

    assert (h5chunks == p5chunks).all()


if __name__ == "__main__":
    test_h5d_chunking_details()

