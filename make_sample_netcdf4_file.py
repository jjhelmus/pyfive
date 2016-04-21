import numpy as np
import netCDF4

dset = netCDF4.Dataset('basic_example.nc', 'w', format='NETCDF4_CLASSIC')
dset.title = 'Basic example'
dset.createDimension('dim1', 100)
example = dset.createVariable('example', 'i4', ('dim1', ))
example[:] = np.arange(100, dtype='int32')
example.foo = 99.5
example.bar = 42
dset.close()



