"""Bulk fsspec chunk reads must scale roughly linearly with the chunk count."""

import time

import fsspec
import h5py
import numpy as np
import pytest

import pyfive

CHUNK_VALUES = 64


@pytest.fixture
def memory_fs():
    fs = fsspec.filesystem("memory")
    created = []

    def put(tmp_path, nchunks, name):
        data = np.arange(nchunks * CHUNK_VALUES, dtype="f4").reshape(
            nchunks, CHUNK_VALUES
        )
        local = tmp_path / name
        with h5py.File(local, "w") as f:
            f.create_dataset("d", data=data, chunks=(1, CHUNK_VALUES))
        remote = f"/{name}"
        fs.pipe_file(remote, local.read_bytes())
        created.append(remote)
        return remote, data

    yield fs, put
    for path in created:
        fs.rm(path)


def _read(fs, path, selection=Ellipsis, **file_kwargs):
    with fs.open(path, "rb") as fh:
        with pyfive.File(fh, **file_kwargs) as hfile:
            dataset = hfile["d"]
            # CPU time, not wall time: the read is single-threaded and CPU-bound,
            # and CPU time is not inflated by other work sharing the machine.
            start = time.process_time()
            result = dataset[selection]
            return result, time.process_time() - start


@pytest.mark.parametrize("file_kwargs", [{}, {"max_request_block": 1 << 20}])
@pytest.mark.parametrize(
    "selection",
    [Ellipsis, np.s_[::-1], np.s_[::3], np.s_[100:900:7]],
    ids=["all", "reversed", "strided", "strided-subset"],
)
def test_bulk_read_matches_for_any_chunk_order(
    memory_fs, tmp_path, selection, file_kwargs
):
    # Reversed and strided selections request chunks in non-increasing or
    # sparse byte order, so ranges cannot be assumed to be sorted or adjacent.
    fs, put = memory_fs
    path, data = put(tmp_path, 1500, "order.h5")
    result, _ = _read(fs, path, selection, **file_kwargs)
    np.testing.assert_array_equal(result, data[selection])


@pytest.mark.timing_sensitive
def test_bulk_read_time_is_linear_in_chunk_count(memory_fs, tmp_path):
    """Matching fetched ranges to chunks was quadratic in the number of chunks.

    Reading 16x as many chunks should take about 16x as long if the cost is
    linear, and about 256x as long if it is quadratic (the old code measured
    over 100x). The threshold of 64 leaves a factor of 4 either side, so that
    timing noise on a busy machine cannot decide the outcome. Comparing two
    reads in the same run cancels out the speed of the machine.
    """
    fs, put = memory_fs
    small_n, large_n = 2000, 32000
    small_path, small_data = put(tmp_path, small_n, "small.h5")
    large_path, large_data = put(tmp_path, large_n, "large.h5")

    # Interleave the runs so that a slow period affects both sizes alike.
    small = []
    large = []
    for _ in range(5):
        small.append(_read(fs, small_path)[1])
        result, elapsed = _read(fs, large_path)
        large.append(elapsed)
    np.testing.assert_array_equal(result, large_data)

    ratio = min(large) / min(small)
    assert ratio < 64, (
        f"{large_n} chunks took {min(large):.3f}s vs {min(small):.3f}s for {small_n}: "
        f"x{ratio:.0f} for {large_n // small_n}x the chunks suggests superlinear scaling"
    )
