import h5py
import pyfive
import io

def make_file(file_like, _vlen_string):
    with h5py.File(file_like,'w') as hfile:
        
        dt = h5py.special_dtype(vlen=str)
        v = hfile.create_dataset("var_len_str", (1,), dtype=dt)
        v[0] = _vlen_string
    

def test_vlen_string():

    tfile = io.BytesIO()
    _vlen_string = "foo"
    make_file(tfile, _vlen_string)
    with pyfive.File(tfile) as hfile:
        print(hfile)
        ds1 = hfile['var_len_str']
        assert _vlen_string == ds1[0]
      