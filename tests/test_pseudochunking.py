import pytest
import numpy as np
import pyfive 
import h5py
import io
from pathlib import Path

DIRNAME = Path(__file__).parent
DATASET_MULTIDIM_HDF5_FILE = DIRNAME/'data/dataset_multidim.hdf5'

@pytest.fixture
def setup_data():
    
    file_like = io.BytesIO()
    f = h5py.File(file_like, 'w', libver='earliest')
    nx, ny = 600,400
    data = np.arange(nx*ny).reshape(nx,ny)
    f.create_dataset('var1', shape=(nx,ny), dtype='<i4',data=data, track_times=False)
    f.close()

    with pyfive.File(file_like,'r') as f:
        var1 = f['var1']
        # use 100 KB as the chunk size
        var1.id.set_psuedo_chunk_size(0.1)

    return var1, data


def test_get_direct_from_contiguous(setup_data):
    
    var1, data = setup_data
    result = var1[...]
    np.testing.assert_array_equal(result,data)


def test_get_direct_from_contiguous_with_slice(setup_data):

    var1, data = setup_data

    via_file = var1[10:50,2:5]
    via_data = data[10:50,2:5]

    np.testing.assert_array_equal(via_file, via_data)
    
  