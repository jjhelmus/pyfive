#! /usr/bin/env python
""" Create a NetCDF4 classic file. """

import netCDF4
import numpy as np

f = netCDF4.Dataset('netcdf4_classic.nc', 'w', format='NETCDF4_CLASSIC')
f.createDimension('x', 4)
f.attr1 = -123
f.attr2 = 130

var = f.createVariable('var1', 'i4', ('x', ))
var[:] = np.arange(4)
var.attr3 = 12.34
var.attr4 = 'Hi'

f.close()
