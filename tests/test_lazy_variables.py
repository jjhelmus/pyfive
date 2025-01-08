
import pyfive
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent

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
    qsum2 =  sum(qdata2)

    assert qsum1 == qsum2




    


