"""
Lightweight check of the layout of chunked data in an HDF5 file, local or remote.

Two characteristics make reads of chunked data from remote storage (HTTP, S3)
slow, and both can be found without reading any actual data:

- *fragmented metadata*: the chunk index (b-tree) of a variable lies beyond the
  start of the first chunk of data in the file, so the index can only be read with many small requests
  scattered through the file.
- *unit chunks*: a dimension (of size greater than one) with a chunk size of one,
  which means a very large number of chunks, and hence a very large index.

Both can usually be fixed (by the owner of the data) with ``h5repack``.
"""

import sys
import signal

from pyfive import File
from pyfive.layout import check_datasets


def _check_open_file(f, source, full):
    return check_datasets(f._lazy_datasets(), source, full)


def check_layout(source, full=False, **storage_options):
    """
    Check the layout of the chunked variables in an HDF5 file.

    ``source`` is a local path, a URL (``https://...`` or ``s3://...``, opened with
    ``fsspec``; ``storage_options`` are passed to the ``fsspec`` filesystem), or an
    already open file-like object. Only metadata is read.

    By default only the internal b-tree nodes, and the first and last leaf of each
    index, are read. That is enough to show that metadata is fragmented, so
    ``fragmented_metadata`` is exact, but in that case the per-variable
    ``fragmented`` flags may miss variables whose index is only beyond a chunk that
    was not sampled. If no fragmentation is seen, every leaf is read to confirm it,
    which is cheap as the index is then contiguous. ``full=True`` always reads every
    leaf, so the per-variable flags are exact too.

    Returns a :class:`LayoutReport`.
    """
    if hasattr(source, "read"):
        with File(source) as f:
            return _check_open_file(f, str(getattr(source, "path", source)), full)

    source = str(source)
    if "://" not in source:
        with File(source) as f:
            return _check_open_file(f, source, full)

    import fsspec

    fs, path = fsspec.core.url_to_fs(source, **storage_options)
    with fs.open(path, "rb") as fh, File(fh) as f:
        return _check_open_file(f, source, full)


def format_report(report, verbose=False):
    """Return a human readable version of a :class:`LayoutReport`."""
    lines = [f"File: {report.source}"]
    lines.append(
        f"  {len(report.variables)} chunked variable(s), "
        f"{report.n_other} other dataset(s) (not chunked, or empty)"
    )
    for v in report.variables:
        if not (verbose or v.has_issue):
            continue
        lines.append(
            f"  {v.name}: shape={v.shape} chunks={v.chunks} n_chunks={v.n_chunks}"
        )
        if v.fragmented:
            lines.append(
                f"      fragmented metadata: b-tree range {v.btree_range} "
                f"extends past the first chunk of data in the file"
            )
        if v.unit_chunk_axes:
            lines.append(
                f"      chunk size one in dimension(s) {list(v.unit_chunk_axes)}"
            )
        if verbose and not v.has_issue:
            lines.append("      ok")
    if report.fragmented_metadata:
        lines.append(
            "Fragmented metadata: some chunk indexes lie beyond the start of the data; "
            "reading these remotely needs many small requests."
        )
    if report.unit_chunks:
        lines.append(
            "Unit chunks: some dimensions have chunk size one, giving very many chunks "
            "(and a large chunk index)."
        )
    if report.has_issue:
        lines.append(
            "Consider repacking (e.g. h5repack) with consolidated metadata and larger chunks."
        )
    else:
        lines.append("No layout problems found.")
    return "\n".join(lines)


def main(argv=None):
    """
    Check an HDF5 file (local path, https:// or s3:// URL) for chunk layouts which are slow to
    read remotely: fragmented metadata (chunk indexes which continue past the start of the data)
    and dimensions with a chunk size of one. Only metadata is read.

    Usage: p5check [-v] [--full] [--anon] filename
    - v will list all chunked variables, not just those with problems
    - full will always read every chunk index in full, which makes the list of
      variables with fragmented metadata exact, but can be slow for files with
      fragmented metadata (by default, the upper levels and two leaves of each index
      are read, and every leaf only if no fragmentation is found)
    - anon will use anonymous access for s3:// URLs
    """
    if argv is None:
        argv = sys.argv[1:]

    verbose = anon = full = False
    args = []
    for arg in argv:
        if arg == "-h":
            print(main.__doc__)
            return 0
        elif arg == "-v":
            verbose = True
        elif arg == "--full":
            full = True
        elif arg == "--anon":
            anon = True
        elif arg.startswith("-"):
            raise ValueError(f"Invalid arguments: {argv}")
        else:
            args.append(arg)

    match args:
        case []:
            raise ValueError("No filename provided")
        case [source]:
            options = {"anon": True} if anon and source.startswith("s3://") else {}
            print(format_report(check_layout(source, full=full, **options), verbose))
            return 0
        case _:
            raise ValueError(f"Invalid arguments: {argv}")


if __name__ == "__main__":
    try:
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    except (AttributeError, ValueError):
        pass

    try:
        sys.exit(main())
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
