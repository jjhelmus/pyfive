""" Create an HDF5 files with an enum datatype using the netcdf interface """
from netCDF4 import Dataset
import numpy as np

ncd = Dataset('enum_variable.hdf5','w')
enum_dict = dict(stratus=1, cumulus=2, nimbus=3, missing=255)
enum_type = ncd.createEnumType(np.uint8,'enum_t', enum_dict)

dim = ncd.createDimension('axis',5) 
enum_var = ncd.createVariable('enum_var',enum_type,'axis',
                                fill_value=enum_dict['missing'])
enum_var[:] = [enum_dict[k] for k in ['stratus','stratus','missing','nimbus','cumulus']]
ncd.close()
