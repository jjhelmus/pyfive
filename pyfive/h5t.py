def check_enum_dtype(dt):
    """
    If the dtype represents an HDF5 enumerated type, returns the dictionary
    mapping string names to integer values.
    Returns None if the dtype does not represent an HDF5 enumerated type.
    
    Note that currently pyfive does not support HDF5 enumerated types,
    so this will always return None (see datatype_msg), and AFIK, should
    never get called in anger. It is only included so h5netcdf wont
    barf at its absence when pyfive is used as a backend.
    """
    try:
        return dt.metadata.get('enum', None)
    except AttributeError:
        return None