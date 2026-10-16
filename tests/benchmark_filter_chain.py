"""Opt-in filter benchmark; timings are evidence, not CI pass/fail thresholds.

Run from the checkout with PYTHONPATH=. python tests/benchmark_filter_chain.py.
All replacements and instrumentation are scoped to this single-process harness.
"""

import argparse
from collections import defaultdict
from contextlib import contextmanager, ExitStack
import importlib.metadata
import json
from pathlib import Path
import platform
import statistics
import sys
import tempfile
import time
from unittest.mock import patch

import fsspec
import h5py
import numpy as np

import pyfive
from pyfive.btree import (
    BTreeV1RawDataChunks,
    FLETCH32_FILTER,
    GZIP_DEFLATE_FILTER,
    SHUFFLE_FILTER,
)
from pyfive.h5d import ChunkRead
from pyfive.utilities import MetadataBufferingWrapper


FILTER_NAMES = {
    GZIP_DEFLATE_FILTER: "inflate",
    SHUFFLE_FILTER: "shuffle",
    FLETCH32_FILTER: "fletcher32",
}
VARIANTS = ("current", "numpy-fletcher", "numpy-shuffle", "numpy-both")


def numpy_unshuffle(buffer, itemsize):
    if itemsize == 1:
        return buffer
    main_size = len(buffer) - len(buffer) % itemsize
    lanes = np.frombuffer(buffer, dtype="u1", count=main_size)
    return (
        lanes.reshape(itemsize, main_size // itemsize).T.tobytes() + buffer[main_size:]
    )


def numpy_verify_fletcher32(buffer):
    """Experimental bounded reductions; retain HDF5 byte order and odd padding."""
    payload = buffer[:-4]
    if len(payload) % 2:
        payload += b"\0"
    words = np.frombuffer(payload, dtype="<u2")
    sum1 = sum2 = 0
    # Bound each reduction: even maximal words cannot overflow uint64.
    for offset in range(0, words.size, 65536):
        block = words[offset : offset + 65536]
        prefix = np.cumsum(block, dtype=np.uint64)
        sum2 = (sum2 + block.size * sum1 + int(prefix.sum())) % 65535
        sum1 = (sum1 + int(prefix[-1])) % 65535
    ref1, ref2 = np.frombuffer(buffer[-4:], dtype=">u2")
    if sum1 != ref1 or sum2 != ref2:
        raise ValueError("fletcher32 checksum invalid")
    return True


@contextmanager
def filter_trial(variant="current", instrument=False):
    """Run real filters one stage at a time, preserving order and mask indices."""
    original = BTreeV1RawDataChunks._filter_chunk
    stats = defaultdict(float)
    with ExitStack() as stack:
        if variant in ("numpy-fletcher", "numpy-both"):
            stack.enter_context(
                patch.object(
                    BTreeV1RawDataChunks,
                    "_verify_fletcher32",
                    staticmethod(numpy_verify_fletcher32),
                )
            )
        elif variant == "native-fletcher":
            from numcodecs import Fletcher32

            codec = Fletcher32()

            def native_verify(buffer):
                codec.decode(buffer)
                return True

            stack.enter_context(
                patch.object(
                    BTreeV1RawDataChunks,
                    "_verify_fletcher32",
                    staticmethod(native_verify),
                )
            )
        elif variant not in VARIANTS:
            raise ValueError(f"Unknown benchmark variant: {variant}")

        def decode(cls, buffer, mask, pipeline, itemsize):
            for index in range(len(pipeline) - 1, -1, -1):
                if mask & (1 << index):
                    continue
                entry = pipeline[index]
                filter_id = entry["filter_id"]
                input_bytes = len(buffer)
                start = time.perf_counter() if instrument else 0
                if filter_id == SHUFFLE_FILTER and variant in (
                    "numpy-shuffle",
                    "numpy-both",
                ):
                    buffer = numpy_unshuffle(buffer, itemsize)
                else:
                    buffer = original(buffer, 0, [entry], itemsize)
                if instrument:
                    name = FILTER_NAMES.get(filter_id, f"filter_{filter_id}")
                    stats[name + "_s"] += time.perf_counter() - start
                    stats[name + "_calls"] += 1
                    stats[name + "_input_bytes"] += input_bytes
            return buffer

        # Uninstrumented current is exactly the production pipeline.
        if instrument or variant != "current":
            stack.enter_context(
                patch.object(
                    BTreeV1RawDataChunks,
                    "_filter_chunk",
                    classmethod(decode),
                )
            )
        yield stats


def timed_read(opener, selection, variant, instrument, expected=None, serial=False):
    """Measure open/lookup/read separately, including process CPU and fetch work."""
    start = time.perf_counter()
    handle = opener()
    handle_open_s = time.perf_counter() - start
    with handle, ExitStack() as stack:
        with filter_trial(variant, instrument) as stats:
            stats["handle_open_s"] = handle_open_s

            def measured(function, seconds_key, calls_key, bytes_key):
                def call(*args, **kwargs):
                    start = time.perf_counter()
                    result = function(*args, **kwargs)
                    stats[seconds_key] += time.perf_counter() - start
                    stats[calls_key] += 1
                    buffers = result if isinstance(result, list) else [result]
                    stats[bytes_key] += sum(len(b) for b in buffers)
                    return result

                return call

            if instrument:
                stack.enter_context(
                    patch.object(
                        handle,
                        "read",
                        measured(
                            handle.read,
                            "handle_read_s",
                            "handle_read_calls",
                            "handle_read_bytes",
                        ),
                    )
                )
                wrapper_read = MetadataBufferingWrapper.read

                def measured_wrapper_read(self, *args, **kwargs):
                    return measured(
                        lambda: wrapper_read(self, *args, **kwargs),
                        "wrapper_read_s",
                        "wrapper_read_calls",
                        "wrapper_read_bytes",
                    )()

                stack.enter_context(
                    patch.object(
                        MetadataBufferingWrapper,
                        "read",
                        measured_wrapper_read,
                    )
                )
                if hasattr(handle, "fs"):
                    stack.enter_context(
                        patch.object(
                            handle.fs,
                            "cat_ranges",
                            measured(
                                handle.fs.cat_ranges,
                                "cat_ranges_s",
                                "cat_ranges_calls",
                                "cat_ranges_bytes",
                            ),
                        )
                    )
                decode = ChunkRead._decode_chunk

                def measured_decode(self, *args, **kwargs):
                    start = time.perf_counter()
                    result = decode(self, *args, **kwargs)
                    stats["decode_s"] += time.perf_counter() - start
                    stats["chunks"] += 1
                    stats["decoded_bytes"] += result.nbytes
                    return result

                stack.enter_context(
                    patch.object(
                        ChunkRead,
                        "_decode_chunk",
                        measured_decode,
                    )
                )
            start = time.perf_counter()
            with pyfive.File(handle) as file:
                stats["open_s"] = time.perf_counter() - start
                start = time.perf_counter()
                dataset = file[selection[0]]
                stats["lookup_s"] = time.perf_counter() - start
                stats["dtype"] = str(dataset.dtype)
                stats["chunk_shape"] = dataset.chunks
                stats["shape"] = dataset.shape
                stats["filters"] = [
                    entry["filter_id"] for entry in (dataset.id.filter_pipeline or [])
                ]
                dataset.id.set_parallelism(
                    thread_count=0,
                    cat_range_allowed=not serial,
                )
                stats["read_strategy"] = (
                    "serial" if serial or dataset.id.posix else "fsspec-cat-ranges"
                )
                if selection[1] == "chunk":
                    if dataset.chunks is None:
                        raise ValueError("chunk selection requires chunked storage")
                    index = tuple(slice(0, n) for n in dataset.chunks)
                elif selection[1] == "narrow":
                    index = tuple(slice(0, 1) for _ in dataset.shape)
                else:
                    index = Ellipsis
                before = dict(stats)
                cpu = time.process_time()
                start = time.perf_counter()
                result = dataset[index]
                stats["read_wall_s"] = time.perf_counter() - start
                stats["read_cpu_s"] = time.process_time() - cpu
                stats["logical_bytes"] = result.nbytes
                for key in (
                    "handle_read_s",
                    "handle_read_calls",
                    "handle_read_bytes",
                    "cat_ranges_s",
                    "cat_ranges_calls",
                    "cat_ranges_bytes",
                    "wrapper_read_s",
                    "wrapper_read_calls",
                    "wrapper_read_bytes",
                ):
                    stats[key] -= before.get(key, 0)
                if expected is not None:
                    np.testing.assert_array_equal(result, expected[index])
                stats["read_other_s"] = (
                    (
                        stats["read_wall_s"]
                        - stats["decode_s"]
                        - (
                            stats["wrapper_read_s"]
                            if not dataset.id.posix
                            else stats["handle_read_s"]
                        )
                        - stats["cat_ranges_s"]
                    )
                    if instrument
                    else None
                )
                return dict(stats), result


def benchmark_file(
    opener, dataset, selection, variants, repeats, expected=None, serial=False
):
    samples = {variant: [] for variant in variants}
    profiles = {variant: [] for variant in variants}
    reference = expected
    for repeat in range(repeats):
        # Alternate order to reduce systematic cache/thermal ordering bias.
        order = variants if repeat % 2 == 0 else variants[::-1]
        for variant in order:
            plain, result = timed_read(
                opener,
                (dataset, selection),
                variant,
                False,
                expected,
                serial,
            )
            if reference is None:
                reference = result.copy()
            if expected is None:
                np.testing.assert_array_equal(result, reference)
            profiled, result = timed_read(
                opener,
                (dataset, selection),
                variant,
                True,
                expected,
                serial,
            )
            if expected is None:
                np.testing.assert_array_equal(result, reference)
            samples[variant].append(plain)
            profiles[variant].append(profiled)
    return {
        variant: {
            "samples": samples[variant],
            "profiles": profiles[variant],
            "read_wall_median_s": statistics.median(
                s["read_wall_s"] for s in samples[variant]
            ),
            "read_wall_min_s": min(s["read_wall_s"] for s in samples[variant]),
            "read_wall_max_s": max(s["read_wall_s"] for s in samples[variant]),
        }
        for variant in variants
    }


def environment():
    versions = {}
    for name in ("pyfive", "numpy", "h5py", "fsspec", "s3fs", "zarr", "numcodecs"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "versions": versions,
        "pyfive_source": pyfive.__file__,
        "cache_note": "Warm process/OS; fresh handles do not imply cold caches.",
        "io_note": "Returned range bytes/read calls are not HTTP wire metrics.",
    }


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mib", type=positive_int, default=8)
    parser.add_argument("--chunk-kib", type=positive_int, nargs="+", default=[64, 1024])
    parser.add_argument("--repeats", type=positive_int, default=3)
    parser.add_argument("--itemsize", type=int, choices=(4, 8), default=4)
    parser.add_argument(
        "--native",
        action="store_true",
        help="also trial numcodecs Fletcher32 (optional dependency)",
    )
    parser.add_argument(
        "--serial",
        action="store_true",
        help="disable fsspec bulk ranges for a strategy control",
    )
    parser.add_argument("--url", help="instead read an existing fsspec URL")
    parser.add_argument("--dataset", default="data")
    parser.add_argument(
        "--selection", choices=("full", "chunk", "narrow"), default="full"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    variants = list(VARIANTS) + (["native-fletcher"] if args.native else [])
    report = {"environment": environment(), "arguments": vars(args).copy(), "cases": []}
    report["arguments"]["output"] = str(args.output)
    if args.url:
        fs, path = fsspec.core.url_to_fs(args.url)
        report["cases"].append(
            {
                "source": "fsspec-url",
                "results": benchmark_file(
                    lambda: fs.open(path, "rb", cache_type="none"),
                    args.dataset,
                    args.selection,
                    variants,
                    args.repeats,
                    serial=args.serial,
                ),
            }
        )
    else:
        rng = np.random.default_rng(42)
        count = args.mib * 1024 * 1024 // args.itemsize
        dtype = np.dtype(f"<f{args.itemsize}")
        with tempfile.TemporaryDirectory(prefix="pyfive-filters-") as directory:
            memory = fsspec.filesystem("memory")
            for pattern in ("structured", "random"):
                data = (
                    np.arange(count, dtype=dtype) % 1024
                    if pattern == "structured"
                    else rng.random(count).astype(dtype)
                )
                for kib in args.chunk_kib:
                    if count % (kib * 1024 // args.itemsize):
                        parser.error("dataset bytes must be a multiple of chunk bytes")
                    for gzip in (False, True):
                        for shuffle, fletcher in (
                            (False, False),
                            (True, False),
                            (False, True),
                            (True, True),
                        ):
                            path = Path(directory) / "case.h5"
                            with h5py.File(path, "w") as file:
                                dataset = file.create_dataset(
                                    "data",
                                    data=data,
                                    chunks=(kib * 1024 // args.itemsize,),
                                    compression="gzip" if gzip else None,
                                    compression_opts=4 if gzip else None,
                                    shuffle=shuffle,
                                    fletcher32=fletcher,
                                )
                                plist = dataset.id.get_create_plist()
                                pipeline = [
                                    plist.get_filter(i)[0]
                                    for i in range(plist.get_nfilters())
                                ]
                                stored = dataset.id.get_storage_size()
                            memory_path = f"/pyfive-filter-benchmark/{path.parent.name}"
                            memory.pipe_file(memory_path, path.read_bytes())
                            try:
                                for source, opener in (
                                    ("local", lambda: path.open("rb")),
                                    (
                                        "fsspec-memory",
                                        lambda: memory.open(
                                            memory_path,
                                            "rb",
                                        ),
                                    ),
                                ):
                                    case = {
                                        "source": source,
                                        "pattern": pattern,
                                        "chunk_bytes": kib * 1024,
                                        "filters": pipeline,
                                        "stored_bytes": stored,
                                        "results": benchmark_file(
                                            opener,
                                            "data",
                                            args.selection,
                                            variants,
                                            args.repeats,
                                            data,
                                            args.serial,
                                        ),
                                    }
                                    report["cases"].append(case)
                            finally:
                                memory.rm(memory_path)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {len(report['cases'])} cases to {args.output}")


if __name__ == "__main__":
    main()
