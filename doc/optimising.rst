Optimising speed of data access
******************************* 

HDF5 files can be large and complicated, with complex internal structures which can introduce signficant overheads when accessing the data.

These complexities (and the overheads they introduce) can be mitigated by optimising how you access the data, but this requires an understanding of 
how the data is stored in the file and how the data access library (in this case ``pyfive``) works.

A record of the performance work done in ``pyfive`` itself, and what was measured, is in :doc:`optimisation_changes`.

The data storage complexities arise from two main factors: the use of chunking, and the way attributes are stored in the files.

**Chunking**: HDF5 files can store data in chunks, which allows for more efficient access to large datasets. 
However, this also means that the library needs to maintain an index (`b-tree`) which relates the position in coordinate space to where each chunk is stored in the file.
There is a `b-tree` index for each chunked variable, and this index can be scattered across the file, which can introduce overheads when accessing the data.

**Attributes**: HDF5 files can store attributes (`metadata`) associated with datasets and groups, and these attributes are stored in a separate section of the file.
Again, these can be scattered across the files.


Optimising the files themselves
-------------------------------

Optimal access to data occurs when the data is chunked in a way that matches the access patterns of your application, and when the
`b-tree` indexes and attributes are stored contiguously in the file.  

Users of ``pyfive`` will always confront data files which have been  created by other software, but if possible, it is worth exploring whether 
the `h5repack <https://docs.h5py.org/en/stable/special.html#h5repack>`_ tool can 
be used to make a copy of the file which is optimised for access by using sensible chunks and to store the attributes and `b-tree` indexes contiguously.
If that is possible, then all access will benefit from fewer calls to storage to get the necessary metadata, and the data access will be faster.


Avoiding Loading Information You Don't Need
-------------------------------------------

In general, the more information you load from the file, the slower the access will be. If you know the variables you need, then don't iterate
over the variables, instantiate them directly.

For example, instead of doing:

.. code-block:: python      

    import pyfive

    with pyfive.File("data.h5", "r") as f:
        variables = [var for var in f]
        print("Variables in file:", variables)
        temp = variables['temp']

You can do:

.. code-block:: python

    import pyfive
    with pyfive.File("data.h5", "r") as f:
        temp = f['temp']            

You might do the first when finding out what is in the file, but once you know what you need, it is much more efficient to access the variables directly.
That avoids a lot of loading of metadata and attributes that you don't need, and speeds up the access to the data.


Parallel Data Access
--------------------

Unlike ``h5py``, ``pyfive`` is designed to be thread-safe, and it is possible to access the same file from multiple threads without contention.
This is particularly useful when working with large datasets, as it allows you to read data in parallel without blocking other threads.

For example, you can use the `concurrent.futures` module to read data from multiple variables in parallel:

.. code-block:: python

    import pyfive
    from concurrent.futures import ThreadPoolExecutor

    variable_names = ["var1", "var2", "var3"]

    with pyfive.File("data.h5", "r") as f:

        def get_min_of_variable(var_name):
            dset = f[var_name]
            data = dset[...]  # Read the entire variable
            return data.min()
            
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(get_min_of_variable, variable_names))

    print("Results:", results)

You can do the same thing to parallelise manipulations within the variables, by for example using, ``dask``, but that is beyond the scope of this document.

Reading and decoding the chunks of a *single* variable can also be spread over threads inside ``pyfive``. Decompressing and unshuffling chunks
is often most of the time spent reading a compressed variable, and it releases Python's global interpreter lock, so it does scale
with the number of threads. This is off by default, because if you are already calling ``pyfive`` from your own threads (as above, or from ``dask``)
the two layers of threads multiply. To use it, set the number of worker threads on the dataset identifier before reading:

.. code-block:: python

    import pyfive

    with pyfive.File("data.h5", "r") as f:
        dset = f['temp']
        dset.id.set_parallelism(thread_count=4)
        data = dset[...]  # chunks are read and decoded on four threads

This applies to files on a local file system and to remote files read via ``fsspec``. It does not change the result of any read, only how quickly it completes;
the benefit is greatest for large, compressed variables made up of many chunks.

In addition, you can use the attributes ``max_request_block`` and ``batch_request_size`` to customise how requests are made to remote servers in the pyfive backend.

.. code-block:: python

    import pyfive

    var_name = "var1"
    max_request_block = 10e6 # 10 MB
    batch_request_size = 80

    with pyfive.File("data.h5", "r", 
            max_request_block=max_request_block, 
            batch_request_size=batch_request_size) as f:

        dset = f[var_name]
        data = dset[...]  # Read the entire variable
        print('Min: ',data.min())

The ``batch_request_size`` specifies the number of individual requests that can be made asynchronously in parallel by the backend session. Some remote services providing data access have limits applied to the rate of requests, such that if you attempt to make more than a certain number of requests simultaneously (or if you have more than that number of open/pending requests) you may be denied further requests with a ``HTTP 429: Too Many Requests`` error. If you encounter this problem, it may be effective to simply limit the requests your own client can make concurrently.

In addition to the above limit, it may be more efficient for you to make fewer requests in general as each request will require some time for handling on the server end. If you use the ``max_request_block`` parameter, the backend part of the pyfive client will attempt to combine requests that represent a continuous block of memory, so instead of making many requests you can just make one larger request and reduce the overheads. Range-requests will be combined up to the limit you set here (10 Megabytes in the example above). For maximum efficiency, consider the bandwidth available to you, in relation to the maximum request size and the number of concurrent requests. The above example of 80 requests at 10MB per request, when combined with a typical non-fibre bandwidth of 30 Mbps would mean the whole batch would take 200 seconds which may be more than the timeout limit for the session, so you may want to reduce the number of requests or the block size for your situation.


Using pyfive with S3
--------------------

HDF5 was designed for usage on POSIX file systems where it makes sense to get specific ranges of bytes from files as they are needed.
For example, the extraction of a specific range of bytes from a variable with a statement like ``x=myvar[10:1]`` would require
first the calculation of where that selection of data (10:12) sits in storage, and then the extraction (and perhaps decompression) 
of just the chunks of data needed to get that data.  If the index needed to work that location wasn't in memory, that would need to
be read first.  In practice with ``pyfive`` we try and preload the index, but the net effect of all these operations are a lot of 
small reads from storage. Across a network, using S3, this would be prohibitive, so the ``s3fs`` middleware (used to make the remote
file, which for HDF5 will be stored as one object, look like it is on a file system) tries to make fewer reads and cache those in
memory so repeated reads can be more efficient.  The optimal caching strategy is dependent on the file layout
and the expected access pattern, so ``s3fs`` provides a lot of flexibility as to how to configure that caching strategy.

For ``pyfive`` the three most important variables to consider altering are the 
``default_block_size`` number, the ``default_cache_type`` option and the ``default_fill_cache`` boolean.

- **default_block_size**  
    This is the size (in bytes) of the blocks that ``s3fs`` will read in one transaction.  
    The bigger this is, the fewer reads that are undertaken, but the more memory and bandwidth are used.  
    The default is 50 MB, which is a poor choice for most HDF5 files where the metadata may be scattered across the files.  
    In practice, a value of a small number of MB could be a good compromise for files which have not been repacked to store the metadata contiguously and/or where the data access pattern will be small random chunks.

- **default_cache_type**  
    This is the type of caching that ``s3fs`` will use.  
    Details of the available options for S3 are formally in the `fsspec documentation <https://filesystem-spec.readthedocs.io/en/latest/api.html#read-buffering>`_.  
    Often the default of ``readahead`` is a good choice.

- **default_fill_cache**  
    This is a boolean which determines whether ``s3fs`` will persistently cache the data that it reads.  
    If this is set to ``True``, then the blocks are cached persistently in memory, but if set to ``False``, then it only makes sense in conjunction with ``default_cache_type`` set to ``readahead`` or ``bytes`` to support streaming access to the data.

.. note::

    Even with these strategies, it is possible that the file layout itself is such that access will be slow.  
    See the next section for more details of how to optimise your hDF5 files for cloud acccess.


