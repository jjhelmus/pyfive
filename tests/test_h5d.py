import h5py
import pyfive
from pathlib import Path
import pytest

mypath = Path(__file__).parent
filename = 'compressed.hdf5'
variable_name = 'dataset3'
breaking_address=(2,0)

#mypath = mypath.parent/'bnl/'
#filename = 'da193o_25_day__grid_T_198807-198807.nc'
#variable_name = 'tos'
# breaking_address=(2,0,3)

def chunk_down(ff, vv):
    """ 
    Test the chunking stuff
    """
    var = ff[vv]
    v = var[2,2]
    varid = var.id
    n = varid.get_num_chunks()
    c = varid.get_chunk_info(4)
    with pytest.raises(OSError):
        # This isn't on the chunk boundary, so should fail
        address = breaking_address
        d = varid.read_direct_chunk(address)
    address = c.chunk_offset
    d = varid.read_direct_chunk(address)
    dd = varid.get_chunk_info_by_coord(address)

    return n, c.chunk_offset, c.filter_mask, c.byte_offset, c.size, d, v


def get_chunks(ff, vv, view=0):
    var = ff[vv]
    chunks = list(var.iter_chunks())
    for i in range(view):
        print('Chunk ',i)
        print(chunks[i])
    return chunks


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
        p5chunks = get_chunks(g, variable_name)

    assert h5chunks == p5chunks 

