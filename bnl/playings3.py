import h5py
import pyfive
import s3fs

MYFILE = 'da193o_25_day__grid_T_198807-198807.nc'
S3_URL = 'https://uor-aces-o.s3-ext.jc.rl.ac.uk/'


uri = 'bnl/'+MYFILE
fs = s3fs.S3FileSystem(anon=True, client_kwargs={'endpoint_url': S3_URL})


with fs.open(uri,'rb') as s3file:
    f = h5py.File(s3file,'r')
    tos =f['tos']
    print(tos)
with fs.open(uri,'rb') as s3file2:
    f2 = pyfive.File(s3file2)
    tos2 = f2['tos']
    print(tos2)

