#! /usr/bin/env python
""" Create a HDF5 file with filter pipeline description v2. """

import h5py
import numpy as np

f = h5py.File('filter_pipeline_v2.hdf5', 'w', libver='v108')

f.create_dataset('data', data=np.ones((10,10,10)), chunks=True, compression=9)

f.close()
