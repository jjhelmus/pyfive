p5dump
------

``pyfive`` includes a command line tool ``p5dump`` which can be used to dump the contents of an HDF5 file to the 
terminal (e.g. ``p5dump myfile.hdf5``). This is similar to the ``ncdump`` tool included with the NetCDF library, or the ``h5dump`` tool included 
with the HDF5 library, but like the rest of ``pyfive``, is implemented in pure Python without any dependencies on the HDF5 C library.

``p5dump`` is not identical to either of these tools, though the default output is very close to that of ``ncdump``.
When called with the ``-s`` flag (e.g. ``p5dump -s myfile.hdf5``) the output provides extra information for chunked
datasets, including the locations of the start and end of the chunk index `b-tree` 
and the location of the first data chunk for that variable. This extra information is useful for understanding
the performance of data access for chunked variables, particularly when accessing data in object stores such as
`S3`. In general, if one finds that the `b-tree` index continues past the first data chunk, access 
performance may be sub-optimal - in this situation, if you have control over the data, you might well
consider using the ``h5repack`` tool from the standard HDF5 distribution to make a copy of the file with the 
chunk index and attributes stored contiguously. All tools which read HDF5 files will benefit from this.

A ``p5dump`` example:

.. code-block:: console

   $ p5dump myfile.hdf5
   File: myfile.hdf5 {
   dimensions:
           lon = 8;
           bounds2 = 2;
           lat = 5;
   variables:
           float64 lat_bnds(lat, bounds2) ;
           float32 bounds2(bounds2) ;
           float64 lat(lat) ;
                   lat:units = "degrees_north" ;
                   lat:standard_name = "latitude" ;
                   lat:bounds = "lat_bnds" ;
           float64 lon_bnds(lon, bounds2) ;
           float64 lon(lon) ;
                   lon:units = "degrees_east" ;
                   lon:standard_name = "longitude" ;
                   lon:bounds = "lon_bnds" ;
           float64 time ;
                   time:units = "days since 2018-12-01" ;
                   time:standard_name = "time" ;
           float64 q(lat, lon) ;
                   q:project = "research" ;
                   q:standard_name = "specific_humidity" ;
                   q:units = "1" ;
                   q:coordinates = "time" ;
                   q:cell_methods = "area: mean" ;
   // global attributes:
                   q:Conventions = "CF-1.12" ;
   }

With the ``-s`` option, extra attributes are displayed: *_Storage*,
*_n_chunks*, *_chunk_shape*, *_btree_range*, *_first_chunk*, and
*_compression*:

.. code-block:: console

   $ p5dump -s myfile.hdf5
   File: example_field_0.nc {
   dimensions:
           lat = 5;
           lon = 8;
           bounds2 = 2;
   variables:
           float64 lat_bnds(lat, bounds2) ;
                   lat_bnds:_Storage = "Chunked" ;
                   lat_bnds:_n_chunks = 1 ;
                   lat_bnds:_chunk_shape = (5, 2) ;
                   lat_bnds:_btree_range = (6144, 6144) ;
                   lat_bnds:_first_chunk = 10808 ;
                   lat_bnds:_compression = "gzip(4)" ;
           float32 bounds2(bounds2) ;
                   bounds2:_Storage = "Contiguous" ;
           float64 lat(lat) ;
                   lat:units = "degrees_north" ;
                   lat:standard_name = "latitude" ;
                   lat:bounds = "lat_bnds" ;
                   lat:_Storage = "Chunked" ;
                   lat:_n_chunks = 1 ;
                   lat:_chunk_shape = (5,) ;
                   lat:_btree_range = (10836, 10836) ;
                   lat:_first_chunk = 12932 ;
                   lat:_compression = "gzip(4)" ;
           float64 lon_bnds(lon, bounds2) ;
                   lon_bnds:_Storage = "Chunked" ;
                   lon_bnds:_n_chunks = 1 ;
                   lon_bnds:_chunk_shape = (8, 2) ;
                   lon_bnds:_btree_range = (12959, 12959) ;
                   lon_bnds:_first_chunk = 15575 ;
                   lon_bnds:_compression = "gzip(4)" ;
           float64 lon(lon) ;
                   lon:units = "degrees_east" ;
                   lon:standard_name = "longitude" ;
                   lon:bounds = "lon_bnds" ;
                   lon:_Storage = "Chunked" ;
                   lon:_n_chunks = 1 ;
                   lon:_chunk_shape = (8,) ;
                   lon:_btree_range = (15621, 15621) ;
                   lon:_first_chunk = 17717 ;
                   lon:_compression = "gzip(4)" ;
           float64 time ;
                   time:units = "days since 2018-12-01" ;
                   time:standard_name = "time" ;
                   time:_Storage = "Contiguous" ;
           float64 q(lat, lon) ;
                   q:project = "research" ;
                   q:standard_name = "specific_humidity" ;
                   q:units = "1" ;
                   q:coordinates = "time" ;
                   q:cell_methods = "area: mean" ;
                   q:_Storage = "Chunked" ;
                   q:_n_chunks = 4 ;
                   q:_chunk_shape = (3, 4) ;
                   q:_btree_range = (20663, 20663) ;
                   q:_first_chunk = 17755 ;
                   q:_compression = "gzip(4)" ;
   // global attributes:
                   q:Conventions = "CF-1.12" ;
   }

p5check
-------

``p5check`` is a lightweight companion to ``p5dump -s`` for deciding, without reading any data, whether a file (local, or a
remote ``https://`` or ``s3://`` URL) will be slow to read remotely. It reports two things for chunked variables:

* *fragmented metadata*: the chunk index (`b-tree`) of a variable lies beyond the start of the first chunk of data in the file, 
  so reading the index needs many small requests scattered through the file.
* *unit chunks*: a dimension (of size greater than one) with a chunk size of one, which gives very many chunks and a large chunk index.

.. code-block:: console

   $ p5check https://example.org/data/myfile.nc
   File: https://example.org/data/myfile.nc
     1 chunked variable(s), 0 other dataset(s) (not chunked, or empty)
     /q: shape=(8640, 324, 432) chunks=(1, 324, 432) n_chunks=8640
         fragmented metadata: b-tree range (895303, 3829225883) extends past the first chunk of data in the file
         chunk size one in dimension(s) [0]
   ...

By default only the upper levels of each chunk index and its first and last leaf node are read, which is enough to show that the metadata is
fragmented. If none is found every leaf is then read to confirm it, which is cheap as an unfragmented index is contiguous. 
``--full`` (``check_layout(source, full=True)``) always reads every leaf, which makes the list of affected variables exact too.
The same check provides the ``File.consolidated_metadata`` flag.
Use ``-v`` to list all chunked variables, and ``--anon`` for anonymous access to ``s3://`` URLs. The same checks are available from 
Python as ``pyfive.p5check.check_layout(source)``, which returns a report object. As with ``p5dump -s``, the usual remedy is to 
copy the file with ``h5repack`` using consolidated metadata and larger chunks.
