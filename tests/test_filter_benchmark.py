"""Correctness contracts for the opt-in performance investigation."""

import struct

import h5py
import numpy as np
import pytest

from benchmark_filter_chain import (
    VARIANTS,
    benchmark_file,
    filter_trial,
    numpy_unshuffle,
    numpy_verify_fletcher32,
)
from pyfive.btree import BTreeV1RawDataChunks


def checked_buffer(payload):
    """Independent scalar reference, including odd-byte zero padding."""
    padded = payload + (b"\0" if len(payload) % 2 else b"")
    sum1 = sum2 = 0
    for (word,) in struct.iter_unpack("<H", padded):
        sum1 = (sum1 + word) % 65535
        sum2 = (sum2 + sum1) % 65535
    return payload + struct.pack(">HH", sum1, sum2)


@pytest.mark.parametrize("size", [0, 1, 2, 3, 4097, 131071, 131072, 131073, 262147])
@pytest.mark.parametrize("pattern", ["random", "maximal"])
def test_numpy_checksum_reference_and_corruption(size, pattern):
    payload = (
        np.random.default_rng(42).integers(0, 256, size, dtype="u1").tobytes()
        if pattern == "random"
        else b"\xff" * size
    )
    raw = checked_buffer(payload)
    assert numpy_verify_fletcher32(raw) is True
    assert BTreeV1RawDataChunks._verify_fletcher32(raw) is True
    corrupt_checksum = raw[:-1] + bytes([raw[-1] ^ 1])
    with pytest.raises(ValueError, match="fletcher32 checksum invalid"):
        numpy_verify_fletcher32(corrupt_checksum)
    if payload:
        corrupt_data = bytes([raw[0] ^ 1]) + raw[1:]
        with pytest.raises(ValueError, match="fletcher32 checksum invalid"):
            numpy_verify_fletcher32(corrupt_data)


@pytest.mark.parametrize("itemsize", [1, 2, 3, 4, 8, 16, 64])
def test_numpy_unshuffle_all_remainders(itemsize):
    for elements in (0, 1, 257):
        for remainder in range(itemsize):
            size = elements * itemsize + remainder
            payload = bytes(i % 256 for i in range(size))
            main = size - remainder
            shuffled = (
                b"".join(payload[lane:main:itemsize] for lane in range(itemsize))
                + payload[main:]
            )
            assert numpy_unshuffle(shuffled, itemsize) == payload


@pytest.mark.parametrize("variant", (*VARIANTS, "native-fletcher"))
@pytest.mark.parametrize("swift_order", [False, True])
@pytest.mark.parametrize("dtype", ["<f4", ">f8", "u1"])
def test_trial_real_hdf5_order_masks_and_restoration(
    tmp_path, variant, swift_order, dtype
):
    if variant == "native-fletcher":
        pytest.importorskip("numcodecs")
    path = tmp_path / "filters.h5"
    data = np.arange(256, dtype=dtype)
    with h5py.File(path, "w") as file:
        dcpl = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
        dcpl.set_chunk(data.shape)
        if swift_order:
            dcpl.set_fletcher32()
            dcpl.set_shuffle()
        else:
            dcpl.set_shuffle()
            dcpl.set_fletcher32()
        dcpl.set_deflate(4)
        dataset = h5py.h5d.create(
            file.id,
            b"data",
            h5py.h5t.py_create(data.dtype),
            h5py.h5s.create_simple(data.shape),
            dcpl=dcpl,
        )
        dataset.write(h5py.h5s.ALL, h5py.h5s.ALL, data)
        dataset.close()
    with h5py.File(path, "r") as file:
        mask, raw = file["data"].id.read_direct_chunk((0,))
        plist = file["data"].id.get_create_plist()
        pipeline = [
            {"filter_id": plist.get_filter(i)[0]} for i in range(plist.get_nfilters())
        ]
    original = BTreeV1RawDataChunks._filter_chunk
    verify = BTreeV1RawDataChunks._verify_fletcher32
    with filter_trial(variant, instrument=True) as stats:
        assert (
            BTreeV1RawDataChunks._filter_chunk(
                raw,
                mask,
                pipeline,
                data.itemsize,
            )
            == data.tobytes()
        )
        assert stats["fletcher32_calls"] == 1
        # Exercise each original mask index, not just the first filter bit.
        for index, entry in enumerate(pipeline):
            skip_all = (1 << len(pipeline)) - 1
            assert (
                BTreeV1RawDataChunks._filter_chunk(
                    raw,
                    skip_all,
                    pipeline,
                    data.itemsize,
                )
                == raw
            )
            single_mask = skip_all ^ (1 << index)
            stage_input = data.tobytes()
            if entry["filter_id"] == 3:
                stage_input = checked_buffer(stage_input)
            elif entry["filter_id"] == 1:
                import zlib

                stage_input = zlib.compress(stage_input)
            assert BTreeV1RawDataChunks._filter_chunk(
                stage_input,
                single_mask,
                pipeline,
                data.itemsize,
            ) == original(stage_input, 0, [entry], data.itemsize)
    assert BTreeV1RawDataChunks._filter_chunk == original
    assert BTreeV1RawDataChunks._verify_fletcher32 == verify


@pytest.mark.parametrize("selection", ["full", "chunk", "narrow"])
@pytest.mark.parametrize("serial", [False, True])
def test_benchmark_instrumentation(tmp_path, selection, serial):
    import fsspec

    data = np.arange(8192, dtype="<f4")
    path = tmp_path / "benchmark.h5"
    with h5py.File(path, "w") as file:
        file.create_dataset(
            "data",
            data=data,
            chunks=(4096,),
            shuffle=True,
            fletcher32=True,
            compression="gzip",
        )
    fs = fsspec.filesystem("memory")
    memory_path = f"/filter-test/{tmp_path.name}"
    fs.pipe_file(memory_path, path.read_bytes())
    try:
        report = benchmark_file(
            lambda: fs.open(memory_path, "rb"),
            "data",
            selection,
            list(VARIANTS),
            2,
            data,
            serial,
        )
    finally:
        fs.rm(memory_path)
    for result in report.values():
        assert len(result["samples"]) == len(result["profiles"]) == 2
        assert result["read_wall_min_s"] <= result["read_wall_median_s"]
        assert result["read_wall_median_s"] <= result["read_wall_max_s"]
        for profile in result["profiles"]:
            chunks = 2 if selection == "full" else 1
            assert profile["chunks"] == chunks
            assert profile["filters"] == [2, 1, 3]
            assert profile["cat_ranges_calls"] == (0 if serial else 1)
            assert profile["decoded_bytes"] == chunks * 4096 * data.itemsize
            if serial:
                # The metadata buffer can satisfy entire small stored chunks.
                assert profile["wrapper_read_bytes"] > 0
                assert profile["read_strategy"] == "serial"
            else:
                assert profile["cat_ranges_bytes"] > 0
                assert profile["read_strategy"] == "fsspec-cat-ranges"
            for name in ("inflate", "shuffle", "fletcher32"):
                assert profile[name + "_calls"] == chunks
                assert profile[name + "_input_bytes"] > 0
                assert 0 <= profile[name + "_s"] <= profile["decode_s"]


@pytest.mark.parametrize("variant", VARIANTS)
def test_trial_restores_after_checksum_failure(variant):
    original = BTreeV1RawDataChunks._filter_chunk
    verify = BTreeV1RawDataChunks._verify_fletcher32
    with pytest.raises(ValueError, match="fletcher32 checksum invalid"):
        with filter_trial(variant, instrument=True):
            BTreeV1RawDataChunks._filter_chunk(
                b"\0\0\0\1",
                0,
                [{"filter_id": 3}],
                4,
            )
    assert BTreeV1RawDataChunks._filter_chunk == original
    assert BTreeV1RawDataChunks._verify_fletcher32 == verify
