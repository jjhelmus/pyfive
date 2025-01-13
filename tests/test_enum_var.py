""" Unit tests for pyfive dealing with an enum variable """

import os
import pytest

import pyfive

DIRNAME = os.path.dirname(__file__)
ENUMVAR_HDF5_FILE = os.path.join(DIRNAME, 'enum_variable.hdf5')

@pytest.mark.xfail(reason="FileNotFoundError: [Errno 2] No such file or directory: '/home/.../pyfive/pyfive/tests/enum_variable.hdf5")
def test_read_enum_variable():

    with pyfive.File(ENUMVAR_HDF5_FILE) as hfile:

        for x in hfile: 
            if x == 'enum_t':
                with pytest.warns(UserWarning,match='^Found '):
                    print(x, hfile[x])
            elif x == 'enum_var':
                with pytest.raises(NotImplementedError):
                    print(x, hfile[x])
            else: 
                print(x, hfile[x])