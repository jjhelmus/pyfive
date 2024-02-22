import pyfive
from pathlib import Path
from pyfive.as_dataobjects import ADataObjects
import s3fs 
import h5py
import time

MYFILE = 'da193o_25_day__grid_T_198807-198807.nc'
#MYFILE = '../tests/chunked.hdf5'
MYFILE =  'da193a_25_day__198807-198807.nc'

INPUT_OPTIONS = [
    ('da193o_25_day__grid_T_198807-198807.nc','tos','s3'),
    ('da193a_25_day__198807-198807.nc','m01s06i247_4','s3'),
    ('../tests/chunked.hdf5','dataset1','local'),
    ('CMIP6-test.nc','tas', 's3')
]

MYPATH = Path(__file__).parent

option = 1
location = INPUT_OPTIONS[option][2]
MYFILE = INPUT_OPTIONS[option][0]

if location == 's3':

    S3_URL = 'https://uor-aces-o.s3-ext.jc.rl.ac.uk/'
    fs = s3fs.S3FileSystem(anon=True, client_kwargs={'endpoint_url': S3_URL})
    uri = 'bnl/'+MYFILE

    t1 = time.time()
    with fs.open(uri,'rb') as s3file2:
        f2 = pyfive.File(s3file2)
        #f2 = pyfive.File(MYPATH/MYFILE)
        path = INPUT_OPTIONS[option][1]
        link_target = f2._links[path]
        dsref = ADataObjects(f2.file._fh, link_target)
        chunk_index = dsref.get_offset_addresses()
        t2 = time.time()
        print(f'Chunk index timer  {t2-t1:.2}s')
        for e in chunk_index:
            print(e)


    with fs.open(uri,'rb') as s3file2:
        f3 = h5py.File(s3file2,'r')
        print(f3[path])


#v='tos'
#tos =f2[v]
#v='dataset1'
#print(tos)
#x = tos[2,:]
#print(x)
#print(tos.shape)



