#
# These are provided to support h5netcdf, and are not used
# by the pyfive package itself. 
#


def check_enum_dtype(dt):
    """
    If the dtype represents an HDF5 enumerated type, returns the dictionary
    mapping string names to integer values.
    Returns None if the dtype does not represent an HDF5 enumerated type.
    ---
    Note that currently pyfive does not support HDF5 enumerated types,
    so this will always return None (see datatype_msg), and AFIK, should
    never get called in anger. It is only included so h5netcdf wont
    barf at its absence when pyfive is used as a backend.
    """
    try:
        return dt.metadata.get('enum', None)
    except AttributeError:
        return None
    

def check_string_dtype(dt):
    """
    If the dtype represents an HDF5 string, returns a string_info object.
    The returned string_info object holds the encoding and the length.
    The encoding can only be 'utf-8' or 'ascii'. The length may be None
    for a variable-length string, or a fixed length in bytes.
    Returns None if the dtype does not represent an HDF5 string.
    ---
    It's not obvious what this is used for yet, so we just return None 
    for now.
    """
    #vlen_kind = check_vlen_dtype(dt)
    #    return string_info('utf-8', None)
    ##if vlen_kind is unicode:
    #elif vlen_kind is bytes:
    #    return string_info('ascii', None)
    #elif dt.kind == 'S':
    #    enc = (dt.metadata or {}).get('h5py_encoding', 'ascii')
    #    return string_info(enc, dt.itemsize)
    #else:
    #    return None
    return None
def check_dtype(**kwds):
    """ Check a dtype for h5py special type "hint" information.  Only one
    keyword may be given.

    vlen = dtype
        If the dtype represents an HDF5 vlen, returns the Python base class.
        Currently only built-in string vlens (str) are supported.  Returns
        None if the dtype does not represent an HDF5 vlen.

    enum = dtype
        If the dtype represents an HDF5 enumerated type, returns the dictionary
        mapping string names to integer values.  Returns None if the dtype does
        not represent an HDF5 enumerated type.

    ref = dtype
        If the dtype represents an HDF5 reference type, returns the reference
        class (either Reference or RegionReference).  Returns None if the dtype
        does not represent an HDF5 reference type.
    """

    if len(kwds) != 1:
        raise TypeError("Exactly one keyword may be provided")

    name, dt = kwds.popitem()

    if name not in ('vlen', 'enum', 'ref'):
        raise TypeError('Unknown special type "%s"' % name)

    try:
        return dt.metadata[name]
    except TypeError:
        return None
    except KeyError:
        return None