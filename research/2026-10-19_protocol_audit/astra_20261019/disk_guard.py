"""Pure disk-budget checks; no filesystem, subprocess, network, or signal calls.

The caller obtains a *complete, checksum-verified* archive's ``7zz l -slt``
listing and a fresh filesystem snapshot. Never list a sparse partial archive.
Existing archives/pins already reduce free_bytes; do not subtract them twice.
The remaining-download estimate is about additional allocated bytes, not download
completion: sparse .part files often have their final logical size from the start.
"""
from __future__ import annotations

import re
from pathlib import PurePosixPath
from typing import Mapping

RESERVE_BYTES = 60_000_000_000
MARGIN_BYTES = 2_000_000_000


def _integer(value, label: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f'{label} is unknown or is not an integer')
    if value < (1 if positive else 0):
        raise ValueError(f'{label} must be {"positive" if positive else "nonnegative"}')
    return value


def _member_path(value: str) -> str:
    normalized = value.replace('\\', '/')
    if (not normalized or '\x00' in normalized or normalized.startswith('/')
            or re.match(r'^[A-Za-z]:', normalized)):
        raise ValueError(f'absolute or invalid member path: {value!r}')
    if any(part in ('.', '..') for part in normalized.split('/')):
        raise ValueError(f'traversal or dot member path: {value!r}')
    return str(PurePosixPath(normalized))


def parse_slt_listing(text: str) -> dict[str, int]:
    """Sum whole_N.csv and beta_N.csv bytes per full folder, failing closed.

    The archive-header Path may be absolute and is intentionally ignored. All
    member paths after the SLT separator must be safe and unique. Selected
    whole/beta records must be positive-sized, paired, and canonically numbered.
    Nonselected safe members are ignored for the extraction budget.
    """
    if not isinstance(text, str):
        raise ValueError('listing text is unknown')
    lines = text.splitlines()
    try:
        start = lines.index('----------') + 1
    except ValueError as error:
        raise ValueError('missing 7zz SLT member separator') from error
    entries: list[dict[str, str]] = []
    current: dict[str, str] | None = None
    for line in lines[start:]:
        if line.startswith('Path = '):
            if current is not None:
                entries.append(current)
            current = {'Path': line[7:]}
        elif current is not None and line.startswith('Size = '):
            if 'Size' in current:
                raise ValueError('duplicate Size field in member record')
            current['Size'] = line[7:]
    if current is not None:
        entries.append(current)

    seen: set[str] = set()
    pairs: dict[tuple[str, int], dict[str, int]] = {}
    for entry in entries:
        path = _member_path(entry['Path'])
        if path in seen:
            raise ValueError(f'duplicate member path: {path}')
        seen.add(path)
        match = re.fullmatch(r'(.+)/(whole|beta)_([1-9][0-9]*)\.csv', path)
        if not match:
            if re.search(r'(^|/)(whole|beta)_[^/]*\.csv$', path):
                raise ValueError(f'unsupported selected member identity: {path}')
            continue
        size_text = entry.get('Size')
        if size_text is None or not re.fullmatch(r'[0-9]+', size_text):
            raise ValueError(f'missing or invalid member Size: {path}')
        size = _integer(int(size_text), 'selected member size', positive=True)
        folder, kind, trial = match[1], match[2], int(match[3])
        pair = pairs.setdefault((folder, trial), {})
        if kind in pair:
            raise ValueError(f'duplicate selected member identity: {path}')
        pair[kind] = size
    if not pairs:
        raise ValueError('no supported whole/beta members; budget is unknown')
    totals: dict[str, int] = {}
    for (folder, trial), pair in pairs.items():
        if set(pair) != {'whole', 'beta'}:
            raise ValueError(f'missing whole/beta pair: {folder} trial {trial}')
        totals[folder] = totals.get(folder, 0) + sum(pair.values())
    return dict(sorted(totals.items()))


def exact_catalog_entry(manifest: Mapping, archive_name: str) -> dict:
    """Select exactly one matching file entry from the local catalog manifest."""
    if (not isinstance(archive_name, str) or not archive_name.endswith('.7z')
            or '/' in archive_name or '\\' in archive_name):
        raise ValueError('expected an exact archive basename')
    if not isinstance(manifest, Mapping):
        raise ValueError('manifest is unknown')
    record = manifest.get('record', manifest)
    if not isinstance(record, Mapping) or not isinstance(record.get('files'), list):
        raise ValueError('manifest file list is unknown')
    matches = [entry for entry in record['files']
               if isinstance(entry, Mapping) and entry.get('key') == archive_name]
    if len(matches) != 1:
        raise ValueError('exact archive metadata missing or duplicated')
    entry = dict(matches[0])
    _integer(entry.get('size'), 'catalog archive size', positive=True)
    return entry


def remaining_download_bytes(manifest: Mapping, archive_name: str, *,
                             allocated_bytes: int | None,
                             logical_bytes: int | None) -> int:
    """Estimate additional allocation from exact size and stat allocation.

    Pass st_blocks * 512, not st_size, for allocated_bytes. For a verified absent
    partial file, pass explicit zeros; None is unknown and is rejected. Filesystem
    blocks may round allocation above logical size. This does not verify MD5 or
    completed download chunks and must never authorize extraction by itself.
    """
    size = exact_catalog_entry(manifest, archive_name)['size']
    allocated = _integer(allocated_bytes, 'partial allocated bytes')
    logical = _integer(logical_bytes, 'partial logical bytes')
    if logical > size:
        raise ValueError('partial logical size exceeds exact catalog size')
    if logical == 0 and allocated != 0:
        raise ValueError('zero logical size with nonzero allocation needs inspection')
    return max(0, size - allocated)


def assess_budget(free_bytes: int, folder_bytes: Mapping[str, int], *,
                  remaining_bytes: int | None,
                  reserve_bytes: int = RESERVE_BYTES,
                  margin_bytes: int = MARGIN_BYTES) -> dict:
    """Budget one sequential extraction folder and remaining download writes.

    Call again with a fresh free-space snapshot at every extraction boundary.
    This assumes no other large download/extraction starts concurrently. Current
    pin allocations are already included in free_bytes. Future archives need a
    separate budget before starting; success here is not approval of the queue.
    """
    free = _integer(free_bytes, 'free bytes')
    remaining = _integer(remaining_bytes, 'remaining download bytes')
    reserve = _integer(reserve_bytes, 'reserve bytes')
    margin = _integer(margin_bytes, 'margin bytes')
    if not isinstance(folder_bytes, Mapping) or not folder_bytes:
        raise ValueError('extraction folder sizes are unknown')
    sizes: dict[str, int] = {}
    for folder, size in folder_bytes.items():
        if not isinstance(folder, str) or _member_path(folder) != folder:
            raise ValueError('folder key must be a normalized safe relative path')
        sizes[folder] = _integer(size, 'folder bytes', positive=True)
    largest = max(sizes, key=sizes.get)
    maximum = sizes[largest]
    required = remaining + maximum + reserve + margin
    return {'ok': free >= required, 'free_bytes': free,
            'remaining_download_bytes': remaining, 'largest_folder': largest,
            'largest_folder_bytes': maximum, 'reserve_bytes': reserve,
            'margin_bytes': margin, 'required_free_bytes': required,
            'headroom_bytes': free - required,
            'projected_free_after_download_and_extraction': free - remaining - maximum}
