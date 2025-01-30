import pyfive
import h5py
import numpy as np 
from pathlib import Path 

## Test files provided as part of https://github.com/NCAS-CMS/pyfive/issues/23

HERE = Path(__file__).parent

def _compare_var_attrs(p5file, h5file):
    """ Compare attributes, which ought to bet he same thing except for the 
    dimension lists which have package dependent attributes """
    assert p5file.keys() == h5file.keys()
    for k,v in p5file.items():
        vv = h5file[k]
        if k == 'DIMENSION_LIST':
            assert len(v) == len(vv)
        else:
            if isinstance(v, np.ndarray):
                assert np.all(v == vv)
            else:
                assert v == vv

def test_A_works():
    """ This file behaves."""
    with pyfive.File(HERE/'data/issue23_A.nc') as hfile:

        file_attrs = hfile.attrs
        q_attrs = hfile['q'].attrs


    with h5py.File(HERE/'data/issue23_A.nc') as hfile:

        file_attrs2 = dict(hfile.attrs)
        q_attrs2 = dict(hfile['q'].attrs)
        # note that unless we explicitly copy these to dicts,
        # they cannot be referenced after the file is closed.

    assert file_attrs == file_attrs2
    _compare_var_attrs(q_attrs, q_attrs2)


def test_B_fails():
    """ This file fails """

    with pyfive.File(HERE/'data/issue23_B.nc') as hfile:

        file_attrs = hfile.attrs
        t_attrs = hfile['tas'].attrs

    with h5py.File(HERE/'data/issue23_B.nc') as hfile:

        file_attrs2 = dict(hfile.attrs)
        t_attrs2 = dict(hfile['tas'].attrs)
        # note that unless we explicitly copy these to dicts,
        # they cannot be referenced after the file is closed.

    assert file_attrs == file_attrs2
    _compare_var_attrs(t_attrs, t_attrs2)


        

