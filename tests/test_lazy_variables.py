
import pyfive
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent


def test_file_laziness():    
    """ Pyfive should not load the data until it is requested. """
    with pyfive.File(HERE/'data/issue23_A.nc') as hfile:
        variables = [v for v in hfile]
        # we do not want to have executed a _getitem__ on any of the variables yet
    
    # check each member of that list is a string
    # it is, it's just the keys of the Mapping superclass of Group
    assert all(isinstance(v, str) for v in variables)







def test_attributes_outside_context():
    """ Pyfive should be able to access attributes outside the context manager. """
    with pyfive.File(HERE/'data/issue23_A.nc') as hfile:

        file_attrs = hfile.attrs
        fdict = dict(file_attrs)
        q_attrs = hfile['q'].attrs
        qdict = dict(q_attrs)
        
    fdict1 = dict(file_attrs)
    qdict1 = dict(q_attrs)

    assert fdict1 == fdict
    assert qdict1 == qdict

def test_file_data_oustside_context():
    """ Pyfive should be able to access data outside the context manager. 
    The data variable should be capable of reopening a closed file when
    it needs access to the data. This mode should support thread 
    parallelism without the need for a lock.
    """

    with pyfive.File(HERE/'data/issue23_A.nc') as hfile:

        qdata = hfile['q']
        qdata1 = qdata[...]
        qsum1 = np.sum(qdata1)

    qdata2 = qdata[...]
    qsum2 =  np.sum(qdata2)

    assert qsum1 == qsum2


def test_numpy_array_type():
    """Pyfive slices should always return a np.ndarray, not a np.memmap.

    """
    # Get data from contiguous file
    with pyfive.File(HERE/'data/issue23_A_contiguous.nc') as hfile:
        qdata = hfile['q']
        qdata1 = qdata[...]
        assert isinstance(qdata1, np.ndarray)
        assert not isinstance(qdata1, np.memmap)

    # Get data from chunked file
    with pyfive.File(HERE/'data/issue23_A.nc') as hfile:
        qdata = hfile['q']
        qdata2 = qdata[...]
        assert isinstance(qdata2, np.ndarray)
        assert not isinstance(qdata2, np.memmap)

    # Check that the data are equal in both cases
    assert (qdata1 == qdata2).all()


    


