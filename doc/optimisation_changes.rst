.. _optimisation_changes:

Performance improvements: what changed and why
**********************************************

This page records a set of performance changes made to the chunked-data read path, why each was made, what was measured, and
what remains. It is written for users who want to know what to expect, and for developers who may want to reproduce or extend the measurements.

Background
==========

A large remote file with ``shuffle`` and ``fletcher32`` in its filter chain was reported to take of order ten minutes to read with ``pyfive``, against
about one minute for ``zarr``, while files without those filters took similar times. The remote I/O was an unlikely cause, since both
libraries used ``fsspec``. Profiling showed several independent CPU-side costs, none of which was the network, and which are described below.

After these changes, reading the whole of one such file (a 4.8 GB remote HDF5 file with 1,825 chunks and the filter chain shuffle, gzip and Fletcher32)
took **2 min 8 s with** ``pyfive`` **and 2 min 2 s with** ``zarr`` (one run each, as reported from another machine, against the same server). That is about 36 MiB/s for
``pyfive``. On this development machine, reading a slice of the same file with ``pyfive`` also ran at 35 MiB/s, 
of which under 10 percent of the time was decoding, and plain ``curl`` range requests reached
11 to 16 MiB/s with one stream and about 33 MiB/s in total with eight parallel streams, so adding concurrency did not help. 


Summary
=======

.. list-table::
   :header-rows: 1
   :widths: 24 38 38

   * - Change
     - Problem
     - Measured effect
   * - Fletcher32 checksum via ``numcodecs``
     - The checksum was verified by a Python loop over every 16-bit word.
     - Verifying a 1 MiB chunk: 76 ms to 0.2 ms. Reading a 4 MiB random-data variable (four 1 MiB chunks) with shuffle, gzip and Fletcher32 was about 25 times faster (234 ms to 9 ms; measured with the ``numcodecs`` checksum before it was adopted).
   * - Linear-time matching of fetched ranges
     - Each chunk scanned all fetched ranges: quadratic in the number of chunks.
     - 64,000 chunks: 57.8 s to 0.35 s. 16,000 chunks: 3.4 s to 0.07 s.
   * - ``thread_count`` also threads decoding
     - Decoding (decompress, unshuffle, checksum) was always serial, even when reading was parallel.
     - Gzip data, 160 chunks, 8 threads: about 5.6 times faster. Full filter chain, 4 threads: about 2.8 times faster.
   * - Unshuffle releases the GIL
     - The ``bytearray`` implementation held the GIL, so threads could not speed it up.
     - Serial: 0.76 times (4-byte) and 0.55 times (8-byte) of the old time. Four threads: 1.3 to 1.5 times faster, against none before.
   * - ``batch_request_size`` honoured once
     - A bulk read made an unthrottled fetch of all ranges, and then fetched them all again with the limit applied.
     - One request, carrying the limit. Backends that reject ``batch_size`` no longer fail.
   * - Fetched chunks released as decoded
     - Every fetched chunk was held until the whole read finished.
     - Peak resident memory for a 1 GiB read: 1,743 to 1,501 MiB; with ``max_request_block``: 2,515 to 1,774 MiB. Speed is unchanged.

Changes in detail
=================

Fletcher32 verification
-----------------------

The checksum is now computed with the native ``numcodecs`` implementation instead of a Python loop. Checksums are still always verified, and
an invalid checksum still raises ``ValueError("fletcher32 checksum invalid")``. Two details preserve the previous behaviour exactly: an empty
payload (which ``numcodecs`` cannot encode) has a zero checksum, and the two checksum words are compared modulo 65535, so that the two
representations of zero (0 and 0xffff) are both accepted.

``numcodecs`` (version 0.16.5 or later) is therefore now a runtime dependency of ``pyfive``.

Range matching in bulk remote reads
-----------------------------------

When reading from an ``fsspec`` file system, all required byte ranges are fetched together, and each chunk then has to be found in the returned
data. This search is now a bisection over the sorted ranges, so the cost grows as K log K for K chunks rather than K squared. This
is independent of filters and of the network. It only matters for variables with very many chunks: it was a negligible cost for the 1,825 chunks of the real file above, but dominated reads of tens of thousands of chunks.

Threaded decoding
-----------------

``dataset.id.set_parallelism(thread_count=N)`` now decodes chunks on ``N`` worker threads, for local files (which were already read in
parallel using ``os.pread``) and for remote files read in bulk through ``fsspec``. Each worker handles a different chunk and writes to a
different part of the output array, so no locking is needed, and an exception in any worker is re-raised to the caller.
For local files each worker now reads, decodes and stores one chunk at a time, so reading overlaps with decoding and only about ``N`` compressed
chunks are held at once.

**The default is unchanged**: ``thread_count`` is 0, meaning that chunks are decoded one at a time on the calling thread. This is deliberate. ``pyfive``
is often called from threads or task schedulers of the caller's own, such as ``dask``, and adding a second layer of threads can slow things down.
The documentation of ``set_parallelism`` previously said "Default 4"; this was incorrect and has been corrected. Anyone who already set
``thread_count`` above zero for a local file now also gets threaded decoding. The serial read strategy for other file-like objects, such as
``BytesIO``, is not threaded, as they share a single handle.

Measured speed-ups (a 160-chunk variable, 0.85 MiB chunks, best of three, ten-core machine):

.. list-table::
   :header-rows: 1

   * - Filters
     - Backend
     - 0 threads
     - 2
     - 4
     - 8
   * - gzip
     - local
     - 1.07 s
     - 0.55 s
     - 0.31 s
     - 0.19 s
   * - gzip
     - fsspec
     - 1.06 s
     - 0.58 s
     - 0.35 s
     - 0.25 s
   * - shuffle, gzip, Fletcher32 (after the unshuffle change)
     - local
     - 0.70 s
     - 0.39 s
     - 0.25 s
     - 0.24 s
   * - shuffle, gzip, Fletcher32 (after the unshuffle change)
     - fsspec
     - 0.70 s
     - 0.41 s
     - 0.27 s
     - 0.26 s

Unshuffling
-----------

The shuffle filter is reversed with NumPy, copying each byte lane into a preallocated array, instead of by ``bytearray`` slice assignment.
NumPy releases the GIL while it copies, so unshuffling on several threads now helps. Several alternatives were measured first. A single NumPy transpose scaled slightly better under threads, but
was up to 1.7 times slower than the old code when run serially, which is the default, so it was not used. ``numcodecs``' own shuffle codec was about the same speed as the old code and
did not scale with threads. The output is still ``bytes``, and any trailing bytes that are not a whole element (for example a checksum when Fletcher32 precedes shuffle in the
filter pipeline) are still copied through unchanged.

Batch request size
------------------

``max_request_block`` and ``batch_request_size`` are described in :doc:`optimising`. Reading with ``batch_request_size`` set previously
issued one fetch without the limit, discarded its result, and then fetched everything again with the limit. This doubled the traffic and defeated the purpose of the
option, which is to avoid ``HTTP 429`` (too many requests) errors. The limit is now passed on the single request. If a backend does not accept ``batch_size`` (or ``on_error``)
that argument is dropped and the read still happens only once.

Memory held during bulk reads
-----------------------------

In a bulk remote read the fetched chunks were all kept until every chunk had been decoded, so peak memory was the output array plus the whole stored size
(plus the stored size again when ``max_request_block`` merges ranges). Each chunk is now located within its fetched block and sliced when it is decoded,
and a block is released as soon as its last chunk is done. Measured on a 1 GiB read (resident memory growth, macOS, sampled every 5 ms):

.. list-table::
   :header-rows: 1

   * - Case
     - Before
     - After
   * - Default
     - 1,743 MiB
     - 1,501 MiB
   * - ``max_request_block`` set
     - 2,515 MiB
     - 1,774 MiB
   * - 4 threads
     - 1,810 MiB
     - 1,624 MiB

The saving is smaller than the stored size because the operating system keeps some freed memory resident, and it may differ on other platforms; it was only measured on macOS. Read time is unchanged.

Investigated, not adopted
=========================

* **Overlapping fetching with decoding.** A bulk request returns only when every range has arrived, so nothing is decoded until the last byte is in. Fetching in groups,
  several at a time, and decoding each group while the next was in flight was tried against a simulated remote store with a 551 MiB variable. It was between 1.1 and 1.5
  times faster than decoding after a single fetch, but it was never faster than the time spent decoding, and it added a lot of code and tuning (a naive version
  fetching groups one after another was several times slower, because each group costs a round trip). On the real file above, fetching was 91 percent of the time and
  decoding under 10 percent, so the gain would have been small. It was reverted.
* **Removing remaining copies in the filter chain.** For 2.5 MiB chunks, the whole chain takes about 6.5 ms, of which gzip is about 4.9 ms. Removing the extra copies in checksum
  handling and unshuffling would save about 0.5 ms (roughly 8 percent) at the cost of a second code path, so it was not done.

What was not changed, and what remains
======================================

* **Reading the chunk index** (the b-tree) is unchanged. For the real file it took 0.5 s, because its index lies in the first MiB, which ``pyfive``
  reads and buffers. In files whose index is scattered between the data, reading it can need many small reads, which is costly on a high-latency link. ``btree_parallel`` exists for
  that case, but it only has an effect if set before the index is built, which happens when a dataset is first accessed. It was not measured on a real remote file.
* **Merging ranges with** ``max_request_block`` uses ``fsspec.utils.merge_offset_ranges``, whose cost grows quadratically with the number of ranges. It only
  affects reads that set that option, and the code is in ``fsspec``.
* **Network speed** cannot be improved by ``pyfive``. When reading is limited by the link, as it was for the file above, none of these changes can help.

How the measurements were made, and their limits
================================================

Most timings use an in-memory ``fsspec`` file system, optionally with simulated latency and bandwidth, so that CPU costs can be separated from the network. 
They were taken on a ten-core Apple M4 with Python 3.12, NumPy 2.4, ``h5py`` 3.16, ``fsspec`` 2026.2 and ``numcodecs`` 0.16.5, and speed-ups will differ on other hardware. 
The real-file results are single runs, and the ``zarr`` store compared against was not inspected for its chunking or compression. 
Speed-ups from threading are limited by how much of the work releases the GIL, and by memory bandwidth.

The tests added with these changes are:

* ``tests/test_fletcher32.py``: compatibility with checksums written by HDF5, corruption and edge cases.
* ``tests/test_bulk_read_scaling.py``: bulk reads scale linearly with the number of chunks.
* ``tests/test_threaded_decode.py``: threaded decoding gives identical results, only runs on worker threads when asked, propagates errors, and is faster.
* ``tests/test_unshuffle.py``: every itemsize and remainder, against chunks shuffled by the HDF5 library itself, and scaling with threads.
* ``tests/test_batch_request_size.py``: one throttled request, and backends that reject ``batch_size``.
* ``tests/test_bulk_read_memory.py``: fetched chunks are released as they are decoded.
* ``tests/benchmark_filter_chain.py`` and ``tests/test_filter_benchmark.py``: an opt-in harness that times each filter stage separately, with local and in-memory ``fsspec`` files, for different chunk sizes
  and filter combinations, and can also profile an existing ``fsspec`` URL. It is run with ``PYTHONPATH=. python tests/benchmark_filter_chain.py --output results.json``, and its timings are not pass or fail checks.

The timing tests compare an operation against a baseline in the same run rather than against fixed times, so that they hold on machines of different speeds, and several of them
were checked to fail against the previous implementation. Even so, run times on shared machines are noisy enough to fail a correct implementation occasionally, so these
tests are marked ``timing_sensitive``: they run locally, but are skipped on GitHub Actions unless ``PYFIVE_RUN_TIMING_TESTS=1`` is set.
