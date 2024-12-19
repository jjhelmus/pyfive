import h5py
import pyfive
from pathlib import Path
import time

mypath = Path(__file__).parent
fname = 'da193o_25_day__grid_T_198807-198807.nc'
p = mypath/fname

h1 = time.time()
f1 = h5py.File(p,'r')
h2 = time.time()
f2 = pyfive.File(p)
h3 = time.time()



v = f2['tos']
d = v._dataobjects
h4 = time.time()
d._get_chunk_addresses()
h5 = time.time()


print("File Opening Time Comparison")
print(f'h5py:   {h2-h1:9.6f}')
print(f'pyfive: {h3-h2:9.6f}')
print(f'Additional times: {h4-h3:9.6f}, {h5-h4:9.6f}')
print(f'Total times: H5 {h4-h3:9.6f}, P5 {h5-h4:9.6f}')

