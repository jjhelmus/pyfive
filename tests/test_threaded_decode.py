"""``thread_count`` should thread chunk decoding as well as chunk reads."""

import os
import threading
import time

import fsspec
import h5py
import numpy as np
import pytest

import pyfive
from pyfive import h5d

CHUNK = (1, 4, 192, 288)
NCHUNKS = 96


@pytest.fixture(scope="module")
def compressed_file(tmp_path_factory):
    """A gzip-compressed, multi-chunk file, plus its bytes on a memory filesystem."""
    rng = np.random.default_rng(7)
    base = np.cumsum(rng.standard_normal(int(np.prod(CHUNK))) * 1e-2)
    base = base.astype("f4").reshape(CHUNK)
    data = np.concatenate([base + i for i in range(NCHUNKS)])
    path = tmp_path_factory.mktemp("threaded") / "gzip.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset(
            "d", data=data, chunks=CHUNK, compression="gzip", compression_opts=4
        )
    fs = fsspec.filesystem("memory")
    fs.pipe_file("/threaded-gzip.h5", path.read_bytes())
    yield path, fs, "/threaded-gzip.h5", data
    fs.rm("/threaded-gzip.h5")


def _open(source, backend, thread_count):
    if backend == "posix":
        hfile = pyfive.File(source[0])
    else:
        hfile = pyfive.File(source[1].open(source[2], "rb"))
    dataset = hfile["d"]
    dataset.id.set_parallelism(
        thread_count=thread_count, cat_range_allowed=backend == "fsspec"
    )
    return hfile, dataset


def _timed_read(source, backend, thread_count, selection=Ellipsis):
    hfile, dataset = _open(source, backend, thread_count)
    with hfile:
        start = time.perf_counter()
        result = dataset[selection]
        return result, time.perf_counter() - start


@pytest.mark.parametrize("backend", ["posix", "fsspec"])
@pytest.mark.parametrize("thread_count", [0, 1, 4])
@pytest.mark.parametrize(
    "selection",
    [Ellipsis, np.s_[::-1], np.s_[5:60:3], np.s_[10:11], np.s_[:, 1:3, ::2, 7]],
    ids=["all", "reversed", "strided", "one-chunk", "partial-chunks"],
)
def test_threaded_decode_matches_serial(
    compressed_file, backend, thread_count, selection
):
    result, _ = _timed_read(compressed_file, backend, thread_count, selection)
    np.testing.assert_array_equal(result, compressed_file[3][selection])


@pytest.mark.parametrize("backend", ["posix", "fsspec"])
@pytest.mark.parametrize(
    "thread_count, expect_workers",
    [(0, False), (3, True)],
    ids=["default-off", "opt-in"],
)
def test_decode_runs_on_worker_threads_only_when_requested(
    compressed_file, monkeypatch, backend, thread_count, expect_workers
):
    deciders = set()
    original = h5d.ChunkRead._decode_chunk

    def spy(self, *args, **kwargs):
        deciders.add(threading.get_ident())
        return original(self, *args, **kwargs)

    monkeypatch.setattr(h5d.ChunkRead, "_decode_chunk", spy)
    result, _ = _timed_read(compressed_file, backend, thread_count)
    np.testing.assert_array_equal(result, compressed_file[3])

    main = threading.get_ident()
    if expect_workers:
        assert main not in deciders and 1 <= len(deciders) <= thread_count
    else:
        assert deciders == {main}


@pytest.mark.parametrize("backend", ["posix", "fsspec"])
def test_worker_errors_propagate(compressed_file, monkeypatch, backend):
    def broken(self, *args, **kwargs):
        raise ValueError("corrupt chunk")

    monkeypatch.setattr(h5d.ChunkRead, "_decode_chunk", broken)
    hfile, dataset = _open(compressed_file, backend, 4)
    with hfile, pytest.raises(ValueError, match="corrupt chunk"):
        dataset[...]


@pytest.mark.timing_sensitive
@pytest.mark.skipif((os.cpu_count() or 1) < 2, reason="needs more than one CPU")
@pytest.mark.parametrize("backend", ["posix", "fsspec"])
def test_threaded_decode_is_faster(compressed_file, backend):
    """gzip inflation releases the GIL, so decoding in threads must beat serial."""
    workers = min(4, os.cpu_count() or 1)
    serial = min(_timed_read(compressed_file, backend, 0)[1] for _ in range(3))
    threaded = min(_timed_read(compressed_file, backend, workers)[1] for _ in range(3))
    assert threaded < 0.8 * serial, (
        f"{backend}: {workers} threads took {threaded:.3f}s vs {serial:.3f}s serial "
        f"(x{serial / threaded:.2f} speedup)"
    )
