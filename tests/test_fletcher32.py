"""Test pyfive's abililty to read datasets with a fletcher32 filter."""

import os
import struct
import unittest

import h5py
import numpy as np
from numpy.testing import assert_array_equal
from numcodecs import Fletcher32
import pytest

import pyfive
from pyfive.btree import BTreeV1RawDataChunks, FLETCH32_FILTER

DIRNAME = os.path.dirname(__file__)
DATASET_FLETCHER_HDF5_FILE = os.path.join(DIRNAME, "data", "fletcher32.hdf5")


def test_fletcher32_datasets():
    with pyfive.File(DATASET_FLETCHER_HDF5_FILE) as hfile:
        # check data
        dset1 = hfile["dataset1"]
        assert_array_equal(dset1[:], np.arange(4 * 4).reshape((4, 4)))
        assert dset1.chunks == (2, 2)

        # check data
        dset2 = hfile["dataset2"]
        assert_array_equal(dset2[:], np.arange(3))
        assert dset2.chunks == (3,)

        # check attribute
        assert dset1.fletcher32


@pytest.mark.parametrize("size", [1, 2, 3, 4097, 131073])
def test_numcodecs_hdf5_fletcher32_compatibility(tmp_path, size):
    data = np.random.default_rng(42).integers(0, 256, size, dtype="u1")
    path = tmp_path / "fletcher32.h5"
    with h5py.File(path, "w") as file:
        dataset = file.create_dataset(
            "data",
            data=data,
            chunks=(size,),
            fletcher32=True,
        )
        mask, raw = dataset.id.read_direct_chunk((0,))

    assert bytes(Fletcher32().decode(raw)) == data.tobytes()
    assert BTreeV1RawDataChunks._verify_fletcher32(raw) is True
    pipeline = [{"filter_id": FLETCH32_FILTER}]
    assert BTreeV1RawDataChunks._filter_chunk(raw, mask, pipeline, 1) == data.tobytes()
    assert BTreeV1RawDataChunks._filter_chunk(raw, 1, pipeline, 1) == raw
    for offset in (0, len(raw) - 1):
        corrupt = bytearray(raw)
        corrupt[offset] ^= 1
        with pytest.raises(ValueError, match="fletcher32 checksum invalid"):
            BTreeV1RawDataChunks._filter_chunk(bytes(corrupt), mask, pipeline, 1)


@pytest.mark.parametrize("size", [0, 1, 2, 513])
@pytest.mark.parametrize("checksum_word", [0, 65535])
def test_fletcher32_zero_checksum(size, checksum_word):
    # Maximal words and an empty payload both have zero modulo-65535 sums.
    payload = b"\xff" * (size * 2)
    raw = payload + struct.pack(">HH", checksum_word, checksum_word)
    assert BTreeV1RawDataChunks._verify_fletcher32(raw) is True
    with pytest.raises(ValueError, match="fletcher32 checksum invalid"):
        BTreeV1RawDataChunks._verify_fletcher32(raw[:-1] + b"\x01")


class TestChunkFletcher32(unittest.TestCase):
    def test_fletcher32_invalid(self):
        bad_chunk = b"\x00\x00\x00\x01"
        with self.assertRaises(ValueError) as context:
            BTreeV1RawDataChunks._verify_fletcher32(bad_chunk)
