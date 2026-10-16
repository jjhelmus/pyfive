"""Bulk fsspec reads must release each fetched chunk once it has been decoded."""

import tracemalloc

import fsspec
import h5py
import numpy as np
import pytest

import pyfive
from pyfive.h5d import ChunkRead

CHUNK = (2, 8, 64, 96)
NCHUNKS = 80


@pytest.fixture(scope="module")
def remote(tmp_path_factory):
    rng = np.random.default_rng(5)
    noise = rng.standard_normal(CHUNK).astype("f4")
    smooth = np.cumsum(rng.standard_normal(CHUNK), axis=-1).astype("f4")
    data = np.concatenate([smooth * 0.01 + noise * (1 + i % 3) for i in range(NCHUNKS)])
    local = tmp_path_factory.mktemp("mem") / "mem.h5"
    with h5py.File(local, "w") as f:
        f.create_dataset(
            "d",
            data=data,
            chunks=CHUNK,
            shuffle=True,
            compression="gzip",
            compression_opts=4,
            fletcher32=True,
        )
    fs = fsspec.filesystem("memory")
    fs.pipe_file("/memory-release.h5", local.read_bytes())
    yield fs, data, local.stat().st_size
    fs.rm("/memory-release.h5")


def live_memory_at_each_decode(monkeypatch):
    live = []
    original = ChunkRead._decode_chunk

    def spy(self, *args, **kwargs):
        live.append(tracemalloc.get_traced_memory()[0])
        return original(self, *args, **kwargs)

    monkeypatch.setattr(ChunkRead, "_decode_chunk", spy)
    return live


@pytest.mark.parametrize("thread_count", [0, 4])
@pytest.mark.parametrize("max_request_block", [None, 1 << 19])
def test_fetched_chunks_are_released_as_they_are_decoded(
    remote, monkeypatch, thread_count, max_request_block
):
    fs, data, stored = remote
    with fs.open("/memory-release.h5", "rb") as fh:
        with pyfive.File(fh, max_request_block=max_request_block) as hfile:
            dataset = hfile["d"]
            dataset.id.set_parallelism(thread_count=thread_count)
            live = live_memory_at_each_decode(monkeypatch)
            tracemalloc.start()
            try:
                result = dataset[...]
            finally:
                tracemalloc.stop()

    np.testing.assert_array_equal(result, data)
    assert len(live) == NCHUNKS
    # Everything fetched is alive at the first decode (the output array is
    # allocated up front, so the difference is the fetched, undecoded chunks).
    held_at_start = live[0] - data.nbytes
    held_at_end = live[-1] - data.nbytes
    assert held_at_start > 0.8 * stored
    # A few chunks are legitimately in flight on the worker threads (stored,
    # inflated and unshuffled copies), so allow for those; unreleased was ~1.0x.
    assert held_at_end < 0.4 * stored, (
        f"{held_at_end / stored:.2f}x of the stored size was still held at the "
        f"last decode (started at {held_at_start / stored:.2f}x)"
    )


def test_memory_release_with_a_selection_that_skips_chunks(remote):
    fs, data, _ = remote
    with fs.open("/memory-release.h5", "rb") as fh:
        with pyfive.File(fh, max_request_block=1 << 19) as hfile:
            dataset = hfile["d"]
            dataset.id.set_parallelism(thread_count=2)
            for selection in (np.s_[::-1], np.s_[10:90:7], np.s_[40:42]):
                np.testing.assert_array_equal(dataset[selection], data[selection])
