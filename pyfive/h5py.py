### This file contains H5Py classes which are not used by
### pyfive, but which are included in the public API for
### htnetcdf which expects to see these H5PY classes.


from pyfive.datatype_msg import DatatypeMessage
import numpy as np

class Datatype:
    """ 
    Class provided for compatability with the H5PY API,
    to allow applications such as h5netcdf to import it,
    but not use it.
    """
    def __init__(self,*args,**kw):
        raise NotImplementedError

class Empty:

    """
    Proxy object to represent empty/null dataspaces (a.k.a H5S_NULL).
    This can have an associated dtype, but has no shape or data. This is not
    the same as an array with shape (0,). This class provided for compatibility
    with the H5Py API to support h5netcdf. It is not used by pyfive.
    """
    shape = None
    size = None

    def __init__(self, dtype):
        self.dtype = np.dtype(dtype)

    def __eq__(self, other):
        if isinstance(other, Empty) and self.dtype == other.dtype:
            return True
        return False

    def __repr__(self):
        return "Empty(dtype={0!r})".format(self.dtype)