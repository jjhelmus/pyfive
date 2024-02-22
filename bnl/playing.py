import pyfive
from pathlib import Path
from pyfive.as_dataobjects import ADataObjects

MYFILE = 'da193o_25_day__grid_T_198807-198807.nc'
MYFILE = '../tests/chunked.hdf5'
MYPATH = Path(__file__).parent

#f = h5py.File(MYPATH/MYFILE,'r')
f2 = pyfive.File(MYPATH/MYFILE)
path = 'dataset1'
link_target = f2._links[path]
dsref = ADataObjects(f2.file._fh, link_target)
chunk_index = dsref.get_offset_addresses()
print(chunk_index)


#v='tos'
#tos =f2[v]
#v='dataset1'
#print(tos)
#x = tos[2,:]
#print(x)
#print(tos.shape)



