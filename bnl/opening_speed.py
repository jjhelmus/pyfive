import h5py
import pyfive
from pathlib import Path
import time
import s3fs

S3_URL = 'https://uor-aces-o.s3-ext.jc.rl.ac.uk/'
S3_BUCKET = 'bnl'

def test_speed(s3=False):

    mypath = Path(__file__).parent
    fname1 = 'da193o_25_day__grid_T_198807-198807.nc'
    vname1 = 'tos'
    p1 = mypath/fname1

    fname2 = 'ch330a.pc19790301-def-short.nc'
    vname2 = 'UM_m01s16i202_vn1106'
    p2 = Path.home()/'Repositories/h5netcdf/h5netcdf/tests/'/fname2

    do_run(p1, fname1, vname1, s3)

    do_run(p2, fname2, vname2, s3)


def do_s3(package, fname, vname): 

    fs = s3fs.S3FileSystem(anon=True, client_kwargs={'endpoint_url': S3_URL})
    uri = S3_BUCKET + '/' + fname
    with fs.open(uri,'rb') as p:    
        t_opening, t_var, t_calc, t_tot = do_inner(package, p, vname)

    return t_opening, t_var, t_calc, t_tot

def do_inner(package, p, vname, withdask=False):
    h0 = time.time()
    pf1 = package.File(p)
    h3 = time.time()
    t_opening = 1000* (h3-h0)

    h5a = time.time()
    vp = pf1[vname]
    h5 = time.time()
    t_var = 1000* (h5-h5a)

    h6a = time.time()
    sh = sum(vp)
    h6 = time.time()
    t_calc = 1000* (h6-h6a)

    t_tot = t_calc+t_var+t_opening

    pf1.close()
    return t_opening, t_var, t_calc, t_tot



def do_run(p, fname, vname, s3):

    if s3:
        import s3fs
      

    # for posix force this to be a comparison from memory
    # by ensuring file is in disk cache and ignore first access
    # but we then do an even number of accesses to make sure we are not 
    # biased by caching. 
    n = 0
    datanames = ['h_opening','p_opening','h_var','p_var','h_calc','p_calc','h_tot','p_tot']
    results = {x:0.0 for x in datanames}
    while n <2:
        n+=1

        if s3:
            h_opening, h_var, h_calc, h_tot = do_s3(h5py, fname, vname)
            p_opening, p_var, p_calc, p_tot = do_s3(pyfive, fname, vname)

        else:
            h_opening, h_var, h_calc, h_tot = do_inner(h5py, p, vname)
            p_opening, p_var, p_calc, p_tot = do_inner(pyfive, p, vname)

        if n>1:
            for x,r  in zip(datanames,[h_opening,p_opening,h_var,p_var,h_calc,p_calc,h_tot,p_tot]):
                results[x] += r

    for v in results.values():
        v = v/(n-1)


    print("File Opening Time Comparison ", fname, f' (ms, S3={s3})')
    print(f"h5py:   {results['h_opening']:9.6f}")
    print(f"pyfive: {results['p_opening']:9.6f}")

    print(f'Variable instantiation for [{vname}]')
    print(f"h5py:   {results['h_var']:9.6f}")
    print(f"pyfive: {results['p_var']:9.6f}")

    print('Access and calculation time for summation')
    print(f"h5py:   {results['h_calc']:9.6f}")
    print(f"pyfive: {results['p_calc']:9.6f}")

    print('Total times')
    print(f"h5py:   {results['h_tot']:9.6f}")
    print(f"pyfive: {results['p_tot']:9.6f}")

if __name__=="__main__":
    test_speed()
    test_speed(s3=True)




