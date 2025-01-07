#! /usr/bin/env python
""" Create a netcdf file with an unlimited dimension, but no data """

import netCDF4
import numpy as np

f = netCDF4.Dataset('netcdf4_empty_unlimited.nc', 'w')
f.createDimension('x', 4)
f.createDimension('unlimited', None)  # Unlimited dimension
v = f.createVariable("foo_unlimited", float, ("x", "unlimited"))
f.close()

f = netCDF4.Dataset('netcdf4_unlimited.nc', 'w')
f.createDimension('x', 4)
f.createDimension('unlimited', None)  # Unlimited dimension
v = f.createVariable("foo_unlimited", float, ("x", "unlimited"))
v[:] = np.ones((4,1))
f.close()
