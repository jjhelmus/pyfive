"""The HDF5 shuffle filter is reversed correctly, and fast, including in threads."""

import os
import time
from concurrent.futures import ThreadPoolExecutor

import h5py
import numpy as np
import pytest

from pyfive.btree import BTreeV1RawDataChunks, SHUFFLE_FILTER

PIPELINE = [{"filter_id": SHUFFLE_FILTER}]


def unshuffle(buffer, itemsize, filter_mask=0):
    return BTreeV1RawDataChunks._filter_chunk(buffer, filter_mask, PIPELINE, itemsize)


def reference_shuffle(buffer, itemsize):
    """Forward shuffle: byte lane j of every element, then the unshuffled tail."""
    main = len(buffer) - len(buffer) % itemsize
    lanes = [buffer[lane:main:itemsize] for lane in range(itemsize)]
    return b"".join(lanes) + buffer[main:]


def reference_unshuffle(buffer, itemsize):
    """The bytearray algorithm used before NumPy was adopted."""
    main = len(buffer) - len(buffer) % itemsize
    out = bytearray(main)
    step = main // itemsize
    for lane in range(itemsize):
        out[lane::itemsize] = buffer[lane * step : (lane + 1) * step]
    return bytes(out) + buffer[main:]


@pytest.mark.parametrize("itemsize", [1, 2, 3, 4, 5, 8, 12, 16, 64])
def test_unshuffle_every_remainder(itemsize):
    rng = np.random.default_rng(itemsize)
    for elements in (0, 1, 2, 257):
        for remainder in range(itemsize):
            size = elements * itemsize + remainder
            original = rng.integers(0, 256, size, dtype="u1").tobytes()
            shuffled = reference_shuffle(original, itemsize)
            result = unshuffle(shuffled, itemsize)
            assert result == original
            assert result == reference_unshuffle(shuffled, itemsize)
            assert type(result) is bytes


@pytest.mark.parametrize("wrapper", [bytes, bytearray, memoryview])
@pytest.mark.parametrize("itemsize", [1, 4])
def test_unshuffle_accepts_any_buffer_and_leaves_it_unchanged(wrapper, itemsize):
    original = bytes(range(100)) * 3 + b"xy"
    shuffled = reference_shuffle(original, itemsize)
    source = (
        wrapper(bytearray(shuffled)) if wrapper is memoryview else wrapper(shuffled)
    )
    assert unshuffle(source, itemsize) == original
    assert bytes(source) == shuffled
    assert type(unshuffle(source, itemsize)) is bytes


def test_unshuffle_skipped_when_masked():
    shuffled = reference_shuffle(bytes(range(64)), 4)
    assert unshuffle(shuffled, 4, filter_mask=1) == shuffled


@pytest.mark.parametrize("dtype", ["<f4", ">f4", "<f8", ">i2", "u1", "<c16", "S7"])
def test_unshuffle_matches_hdf5_library(tmp_path, dtype):
    """Compare with a chunk shuffled by the HDF5 library itself."""
    data = np.arange(1000).astype(dtype)
    path = tmp_path / "shuffle.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("d", data=data, chunks=(1000,), shuffle=True)
    with h5py.File(path, "r") as f:
        mask, raw = f["d"].id.read_direct_chunk((0,))
    assert raw != data.tobytes() or data.dtype.itemsize == 1
    assert unshuffle(raw, data.dtype.itemsize, mask) == data.tobytes()


@pytest.mark.timing_sensitive
@pytest.mark.skipif((os.cpu_count() or 1) < 4, reason="needs at least 4 CPUs")
def test_unshuffle_scales_with_threads():
    """Unshuffling must release the GIL: 4 threads should beat 1 thread.

    The previous bytearray implementation held the GIL for its whole run and
    gained nothing from threads (1.00x); this one gains roughly 1.3-2x on
    4 threads, depending on the machine's memory bandwidth.
    """
    itemsize, nbuffers = 4, 64
    original = np.random.default_rng(0).integers(0, 256, 2**21, dtype="u1").tobytes()
    shuffled = reference_shuffle(original, itemsize)

    def run_all(executor=None):
        if executor is None:
            return [unshuffle(shuffled, itemsize) for _ in range(nbuffers)]
        return list(
            executor.map(lambda _: unshuffle(shuffled, itemsize), range(nbuffers))
        )

    def best(function, repeats=5):
        times = []
        for _ in range(repeats):
            start = time.perf_counter()
            function()
            times.append(time.perf_counter() - start)
        return min(times)

    serial = best(run_all)
    with ThreadPoolExecutor(4) as executor:
        assert run_all(executor)[0] == original
        threaded = best(lambda: run_all(executor))
    assert serial / threaded > 1.15, (
        f"4 threads took {threaded:.3f}s vs {serial:.3f}s serial "
        f"(x{serial / threaded:.2f}); unshuffle is not releasing the GIL"
    )


@pytest.mark.timing_sensitive
def test_unshuffle_is_not_slower_than_bytearray_loop_when_serial():
    itemsize, repeats = 8, 64
    original = np.random.default_rng(1).integers(0, 256, 2**21, dtype="u1").tobytes()
    shuffled = reference_shuffle(original, itemsize)

    def best(function):
        times = []
        for _ in range(5):
            start = time.perf_counter()
            for _ in range(repeats):
                function(shuffled, itemsize)
            times.append(time.perf_counter() - start)
        return min(times)

    new, old = best(unshuffle), best(reference_unshuffle)
    assert new < 1.2 * old, f"unshuffle {new:.3f}s vs bytearray loop {old:.3f}s"
