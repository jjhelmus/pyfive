"""
Assess the layout of chunked data in an HDF5 file, without reading any data.

Two characteristics make reads of chunked data from remote storage (HTTP, S3) slow:

- *fragmented metadata*: the chunk index (b-tree) of a variable lies beyond the
  start of the first chunk of data in the file, so the index can only be read with many
  small requests scattered through the file.
- *unit chunks*: a dimension (of size greater than one) with a chunk size of one,
  which means a very large number of chunks, and hence a very large index.

This module is used by ``File.consolidated_metadata`` and ``p5check``, and must not import
from ``pyfive.high_level``.
"""

import math
from dataclasses import dataclass, field


@dataclass
class VariableLayout:
    """Layout information for one chunked variable."""

    name: str
    shape: tuple
    chunks: tuple
    n_chunks: int
    btree_range: tuple
    first_chunk: int
    unit_chunk_axes: tuple
    fragmented: bool = False  # index extends past the first chunk of data in the file

    @property
    def has_issue(self):
        return self.fragmented or bool(self.unit_chunk_axes)


@dataclass
class LayoutReport:
    """The result of :func:`check_datasets`."""

    source: str
    variables: list = field(default_factory=list)
    n_other: int = 0  # compact, contiguous or empty datasets (nothing to check)

    @property
    def fragmented_metadata(self):
        """True if the chunk index of any variable extends past the first chunk of data in the file."""
        return any(v.fragmented for v in self.variables)

    @property
    def unit_chunks(self):
        """True if any variable has a dimension (of size > 1) with a chunk size of one."""
        return any(v.unit_chunk_axes for v in self.variables)

    @property
    def has_issue(self):
        return self.fragmented_metadata or self.unit_chunks


def _check_variable(path, dataset, full):
    dsid = dataset.id
    if dsid.layout_class != 2:
        return None
    if full:
        # There is no point reading a remote index serially if it can be avoided
        dsid.set_parallelism(btree_parallel=True)
        dsid._build_index()
        if dsid.get_num_chunks() == 0:
            return None
        btree_range, first_chunk = dsid.btree_range, dsid.first_chunk
    else:
        scan = dsid._scan_index()
        if scan is None:
            return None
        btree_range, first_chunk = scan[:2], scan[2]
    shape, chunks = tuple(dataset.shape), tuple(dataset.chunks)
    # the most chunks there could be; a sparse dataset may have fewer
    n_chunks = math.prod(-(-s // c) for s, c in zip(shape, chunks))
    return VariableLayout(
        name=path,
        shape=shape,
        chunks=chunks,
        n_chunks=n_chunks,
        btree_range=btree_range,
        first_chunk=first_chunk,
        unit_chunk_axes=tuple(
            i for i, (s, c) in enumerate(zip(shape, chunks)) if c == 1 and s > 1
        ),
    )


def _scan(datasets, source, full):
    report = LayoutReport(source=source)
    for path, dataset in datasets:
        variable = _check_variable(path, dataset, full)
        if variable is None:
            report.n_other += 1
        else:
            report.variables.append(variable)
    if report.variables:
        first_data = min(v.first_chunk for v in report.variables)
        for v in report.variables:
            v.fragmented = v.btree_range[1] > first_data
    return report


def check_datasets(datasets, source, full=False):
    """
    Check the layout of ``datasets``, a list of ``(path, dataset)`` pairs for all the datasets
    in one file, as given by ``Group._lazy_datasets``. See :func:`pyfive.p5check.check_layout`
    for the meaning of ``full``.
    """
    report = _scan(datasets, source, full)
    if not full and report.variables and not report.fragmented_metadata:
        # Sampled chunks can only prove fragmentation, not rule it out. Confirm by
        # reading every leaf, which is cheap here as an unfragmented index is contiguous.
        report = _scan(datasets, source, True)
    return report
