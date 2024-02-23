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

option = 2
location = INPUT_OPTIONS[option][2]
MYFILE = INPUT_OPTIONS[option][0]
path = INPUT_OPTIONS[option][1]


def working(f2, path, printindex=True):
    link_target = f2._links[path]
    t1 = time.time()
    dsref = ADataObjects(f2.file._fh, link_target)
    chunk_index = dsref.get_offset_addresses()
    t2 = time.time()
    print(f'Chunk index timer  {t2-t1:.2}s')
    if printindex:
        for e in chunk_index:
            print(e)
   
    return t2

if location == 's3':

    S3_URL = 'https://uor-aces-o.s3-ext.jc.rl.ac.uk/'
    fs = s3fs.S3FileSystem(anon=True, client_kwargs={'endpoint_url': S3_URL})
    uri = 'bnl/'+MYFILE

    t1 = time.time()
    with fs.open(uri,'rb') as s3file2:
        f2 = pyfive.File(s3file2)
        t2 = working(f2, path)
        print(f'Complete chunking timer {t2-t1:.2}s')

    with fs.open(uri,'rb') as s3file2:
        f3 = h5py.File(s3file2,'r')
        print(f3[path])

elif location == 'local':

   
    f2 = pyfive.File(MYPATH/MYFILE)
    x = f2[path]
    y = x[2,:]
    print(x.shape)
    print(y)
    t1 = time.time()
    t2 = working(f2, path, printindex=False)
    d = ADataObjects(f2.file._fh, f2._links[path])
    r = d[2:]
    if len(r) >= len(y):
        print(f"yeah, well, it's not working (returning {len(r)} items instead of {len(y)})")
        # as it's stands, r should be a set of indices for chunks containing y, which should have 
        # length less than or equal to length (y). At the moment it's too long, so that's clearly
        # broken
        print(r)
        raise ValueError('Busted')
    raise ValueError('Busted, but in a better way')


else:
    raise ValueError('You stuffed up')

