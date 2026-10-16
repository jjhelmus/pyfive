"""``batch_request_size`` must limit a bulk read, not follow an unthrottled one."""

import fsspec
import h5py
import numpy as np
import pytest

import pyfive

NCHUNKS = 40


@pytest.fixture
def remote(tmp_path):
    fs = fsspec.filesystem("memory")
    data = np.arange(NCHUNKS * 16, dtype="f4").reshape(NCHUNKS, 16)
    local = tmp_path / "batch.h5"
    with h5py.File(local, "w") as f:
        f.create_dataset("d", data=data, chunks=(1, 16))
    fs.pipe_file("/batch.h5", local.read_bytes())
    yield fs, data
    fs.rm("/batch.h5")


def record_cat_ranges(fs, monkeypatch):
    calls = []
    original = fs.cat_ranges

    def spy(paths, starts, stops, *args, **kwargs):
        calls.append({"ranges": len(paths), "kwargs": kwargs})
        return original(paths, starts, stops, *args, **kwargs)

    monkeypatch.setattr(fs, "cat_ranges", spy)
    return calls


@pytest.mark.parametrize("thread_count", [0, 4])
@pytest.mark.parametrize("max_request_block", [None, 1 << 20])
def test_batch_size_read_makes_one_throttled_request(
    remote, monkeypatch, thread_count, max_request_block
):
    fs, data = remote
    with fs.open("/batch.h5", "rb") as fh:
        with pyfive.File(
            fh, batch_request_size=7, max_request_block=max_request_block
        ) as hfile:
            dataset = hfile["d"]
            dataset.id.set_parallelism(thread_count=thread_count)
            calls = record_cat_ranges(fs, monkeypatch)
            np.testing.assert_array_equal(dataset[...], data)

    assert len(calls) == 1, "chunk data must be fetched once, not twice"
    assert calls[0]["kwargs"].get("batch_size") == 7


def test_without_batch_size_nothing_extra_is_passed(remote, monkeypatch):
    fs, data = remote
    with fs.open("/batch.h5", "rb") as fh:
        with pyfive.File(fh) as hfile:
            dataset = hfile["d"]
            calls = record_cat_ranges(fs, monkeypatch)
            np.testing.assert_array_equal(dataset[...], data)

    assert len(calls) == 1
    assert "batch_size" not in calls[0]["kwargs"]


def test_backend_without_batch_size_support_still_reads(remote, monkeypatch):
    """A backend whose cat_ranges rejects ``batch_size`` falls back to no limit."""
    fs, data = remote
    original = fs.cat_ranges

    def strict(paths, starts, stops, on_error="return"):
        return original(paths, starts, stops, on_error=on_error)

    with fs.open("/batch.h5", "rb") as fh:
        with pyfive.File(fh, batch_request_size=7) as hfile:
            dataset = hfile["d"]
            monkeypatch.setattr(fs, "cat_ranges", strict)
            np.testing.assert_array_equal(dataset[...], data)
