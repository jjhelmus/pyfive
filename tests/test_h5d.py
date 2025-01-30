import h5py
import pyfive
from pathlib import Path
import pytest
from operator import mul

mypath = Path(__file__).parent
filename = 'compressed.hdf5'
variable_name = 'dataset3'
breaking_address=(2,0)

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


def get_chunks(ff, vv):
    var = ff[vv]
    chunks = list(var.iter_chunks())
    return chunks

def get_slices(var, using_py5):
    """ Return suitlable test slice from var """
    rank = len(var.shape)
    assert rank == 2
    slice1 = slice(8,15)
    slice2 = slice(8,15)
    return (slice1, slice2)

def test_h5d_chunking_details():

    with h5py.File(mypath/filename) as f:
        h5detail = chunk_down(f, variable_name)

    with pyfive.File(mypath/filename) as g:
        p5detail = chunk_down(g, variable_name)

    assert h5detail == p5detail 

def test_iter_chunks_all():

    with h5py.File(mypath/filename) as f:
        h5chunks = get_chunks(f, variable_name)

    with pyfive.File(mypath/filename) as g:
        p5chunks = get_chunks(g, variable_name)

    assert h5chunks == p5chunks 

def test_iter_chunks_sel():
    """ I don't really understand what h5py is doing here, so 
    obviously I don't have the right method in pyfive and/
    or the right test #FIXME """

    with h5py.File(mypath/filename) as f:
        var = f[variable_name]
        slices = get_slices(var, False)
        h5chunks = list(var.iter_chunks(slices))
        #print(h5chunks,var.shape, var.chunks)

    with pytest.raises(NotImplementedError):
        with pyfive.File(mypath/filename) as g:
            var = g[variable_name]
            slices = get_slices(var, True)
            p5chunks = list(var.iter_chunks(slices))
            #print(p5chunks,var.shape, var.chunks)

        assert h5chunks == p5chunks 