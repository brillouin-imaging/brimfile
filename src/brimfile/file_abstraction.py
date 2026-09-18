from __future__ import annotations

import sys
import io
import warnings
from abc import ABC, abstractmethod
from enum import Enum
import numpy as np
import asyncio
import importlib.util

import zarr
import numcodecs  # noqa: F401  (transitive dependency of zarr's numcodecs-based codecs)
import zarr.api.asynchronous as zarr_async
from zarr.abc.store import (
    Store,
    ByteRequest,
    RangeByteRequest,
    OffsetByteRequest,
    SuffixByteRequest,
)

from typing import TypeAlias
Version: TypeAlias = tuple[int, int, int] | None


class FileAbstraction(ABC):
    """
    Abstract base class that rapresents a general interface to work with brim files.

    This class defines a common interface for file operations, such as creating attributes,
    retrieving attributes, and managing groups and datasets. It is designed to be extended
    by specific file implementations, such as HDF5 or Zarr.

    All the methods which require a path to an exixsting object in the file, will accept
    either the object itself (as defined by the specific implementation) or its path as a string.
    """

    # -------------------- Public attributes --------------------
    version: Version = None

    # -------------------- Attribute Management --------------------

    @abstractmethod
    async def create_attr(self, obj, name: str, data, **kwargs):
        """
        Create an attribute in the file.

        Args:
            obj: object that supports the creation of an attribute (e.g. group or dataset) or its path as a string.
            name (str): Name of the attribute.
            data: Data for the attribute.
            **kwargs: Additional arguments for attribute creation.
        """
        pass

    @abstractmethod
    async def get_attr(self, obj, name: str):
        """
        Return the data of an attribute in the file.

        Args:
            obj: object that supports the creation of an attribute (e.g. group or dataset) or its path as a string.
            name (str): Name of the attribute.
        Raises:
            KeyError: If the attribute does not exist.
        """
        pass

    # -------------------- Group Management --------------------

    @abstractmethod
    async def open_group(self, full_path: str, **kwargs):
        """
        Open a group in the file.

        Args:
            full_path (str): Path to the group.
            **kwargs: Additional arguments for opening the group.
        """
        pass

    @abstractmethod
    async def create_group(self, full_path: str, **kwargs):
        """
        Create a group in the file.

        Args:
            full_path (str): Path to the group.
            **kwargs: Additional arguments for creating the group.
        """
        pass

    # -------------------- Dataset Management --------------------

    class Compression:
        """
        Compression options for datasets.

        `BLOSC`, `GZIP`, and `ZSTD` map to the bytes-to-bytes codecs natively
        bundled with zarr (see https://zarr.readthedocs.io/en/stable/api/zarr/codecs/).
        `ZLIB` is kept for backward compatibility and is equivalent to `BLOSC`
        with `cname='zlib'`.
        """
        NONE = None
        DEFAULT = 1
        ZLIB = 2
        BLOSC = 3
        GZIP = 4
        ZSTD = 5

        def __init__(self, type=DEFAULT, level=None, **kwargs):
            self.type = type
            self.level = level
            # extra codec-specific keyword arguments (e.g. `cname`/`shuffle` for
            # BLOSC, `checksum` for ZSTD), forwarded as-is to the zarr codec
            self.kwargs = kwargs

        def to_zarr_compressor(self):
            """
            Convert the compression options to a zarr-compatible compression object.

            Returns:
                A zarr-compatible compression object or None if no compression is specified.
            """
            kwargs = dict(self.kwargs)
            match self.type:
                case FileAbstraction.Compression.DEFAULT:
                    # see https://zarr.readthedocs.io/en/stable/api/zarr/index.html#zarr.create_array
                    compressor = 'auto'
                case FileAbstraction.Compression.ZLIB:
                    if self.level is not None:
                        kwargs.setdefault('clevel', self.level)
                    compressor = zarr.codecs.BloscCodec(
                        cname='zlib', **kwargs)
                case FileAbstraction.Compression.BLOSC:
                    if self.level is not None:
                                        kwargs.setdefault('clevel', self.level)
                    compressor = zarr.codecs.BloscCodec(**kwargs)
                case FileAbstraction.Compression.GZIP:
                    if self.level is not None:
                        kwargs.setdefault('level', self.level)
                    compressor = zarr.codecs.GzipCodec(**kwargs)
                case FileAbstraction.Compression.ZSTD:
                    if self.level is not None:
                        kwargs.setdefault('level', self.level)
                    compressor = zarr.codecs.ZstdCodec(**kwargs)
                case _:
                    warnings.warn(
                        f"Compression type '{self.type}' not supported by zarr. Using no compression.")
                    compressor = None
            return compressor

    @abstractmethod
    async def open_dataset(self, full_path: str):
        """
        Open a dataset in the file.

        Args:
            full_path (str): Path to the dataset.

        Returns:
            Dataset object which must support numpy indexing and slicing.
        """
        pass

    @abstractmethod
    async def create_dataset(self, parent_group, name: str, data, chunk_size=None, compression: 'FileAbstraction.Compression' = None):
        """
        Create a dataset in the file.

        Args:
            parent_group: Group in which to create the dataset or its path as a string.
            name (str): Name of the dataset.
            data: Data for the dataset.
            chunk_size (tuple, optional): Chunk size for the dataset. If None the automatically computed size will be used.
            compression (FileAbstraction.Compression, optional): Compression options for the dataset.
        """
        pass

    # -------------------- Listing --------------------

    @abstractmethod
    async def list_objects(self, obj) -> list:
        """
        Lists the objects (groups or datasets) contained within one hierarchical level below the given object.

        Args:
            obj: parent object or its path as a string.

        Returns:
            list: List of strings representing the names of the objects.
        """
        pass

    @abstractmethod
    async def object_exists(self, full_path) -> bool:
        """
        Check if an object exists in the file.

        Args:
            full_path (str): Path to the object.

        Returns:
            bool: True if the object exists, False otherwise.
        """
        pass

    @abstractmethod
    async def list_attributes(self, obj) -> list:
        """
        Lists the attributes attached to the specified object.

        Args:
            obj: object or its path as a string.

        Returns:
            list: List of strings representing the names of the attributes.
        """
        pass

    # -------------------- File Management --------------------

    def close(self):
        """
        Close the file.
        """
        pass

    # -------------------- Properties --------------------

    async def is_read_only(self) -> bool:
        """
        Check if the file is read-only.

        Returns:
            bool: True if the file is read-only, False otherwise.
        """
        return True


class StoreType(Enum):
    """
    Enum to represent the type of store used by the Zarr file.
    """
    ZIP = 'zip'
    """We recommend using zip only for reading files. Writing will work, but at the cost of duplicating entries
    inside the archive (see [GitHub issue](https://github.com/zarr-developers/zarr-python/issues/1695)).
    Consider using zarr store instead and zipping it at the end of writing."""
    ZARR = 'zarr'
    S3 = 'S3'
    FOLDER = 'folder'
    """A browser directory picker's file list (Pyodide only) - see `_zarrFile.__init__`."""
    AUTO = 'auto'
    """Automatically determine the store type based on the filename (i.e. extension or url schema).
    Only applies when the source is a string; a browser `File`/file-list source (Pyodide) always
    requires an explicit `ZIP` or `FOLDER` store_type, since there is no filename to inspect."""


# used by _async_getitem: real zarr's own async/sync array classes, on both platforms
_ZarrAsyncArray = zarr.AsyncArray
_ZarrArray = zarr.Array


async def _async_getitem(obj, indices: tuple):
    """
    Asynchronously get a slice of an object that supports indexing and slicing.

    Args:
        obj: Object that supports indexing and slicing (e.g., zarr.AsyncArray).
        indices (tuple): Tuple of indices or slices to retrieve.
    Returns:
        The sliced data from the object.

    N.B. this function is a quick workaround to transition from the sync to async paradigm.
         Consider rethinking the whole structure it in the future!
    """
    if isinstance(indices, list):
        indices = tuple(indices)
    elif not isinstance(indices, tuple):
        indices = (indices,)

    if isinstance(obj, _ZarrAsyncArray):
        # N.B. it is important to check first if obj is a _ZarrAsyncArray,
        # since the async call should have priority
        return await obj.getitem(indices)
    elif isinstance(obj, np.ndarray) or isinstance(obj, _ZarrArray):
        return obj[indices]
    else:
        raise ValueError(f"Object of type '{type(obj)}' does not support indexing and slicing.")


def _gather_sync(*aws, return_exceptions: bool = False):
    """
    Sync version of asyncio.gather.
    Args: same as asyncio.gather
    """
    async def _f():
        return await asyncio.gather(*aws, return_exceptions=return_exceptions)
    return sync(_f())


def _parse_storage_url(url):
    from urllib.parse import urlparse

    parsed = urlparse(url)
    scheme = parsed.scheme
    netloc = parsed.netloc
    path = parsed.path.lstrip('/')

    # Case 1: Amazon S3 (virtual-hosted-style or path-style)
    if netloc == "amazonaws.com" or netloc.endswith(".amazonaws.com"):
        parts = netloc.split('.')
        if parts[0] != 's3':  # virtual-hosted-style
            bucket = parts[0]
            endpoint = '.'.join(parts[1:])
            object_path = path
        else:  # path-style
            path_parts = path.split('/', 1)
            bucket = path_parts[0]
            endpoint = netloc
            object_path = path_parts[1] if len(path_parts) > 1 else ''
    # Case 2: Google Cloud Storage
    elif netloc == "storage.googleapis.com" or netloc.endswith(".storage.googleapis.com"):
        if netloc == "storage.googleapis.com":
            # path-style: https://storage.googleapis.com/bucket-name/object
            path_parts = path.split('/', 1)
            bucket = path_parts[0]
            endpoint = netloc
            object_path = path_parts[1] if len(path_parts) > 1 else ''
        else:
            # virtual-hosted-style: https://bucket-name.storage.googleapis.com/object
            bucket = netloc.split('.')[0]
            endpoint = '.'.join(netloc.split('.')[1:])
            object_path = path
    # Case 3: Custom endpoint or S3-compatible storage (MinIO, etc.)
    else:
        path_parts = path.split('/', 1)
        bucket = path_parts[0]
        endpoint = netloc
        object_path = path_parts[1] if len(path_parts) > 1 else ''

    return {
        'protocol': scheme,
        'bucket': bucket,
        'endpoint': endpoint,
        'object_path': object_path
    }


def _resolve_byte_range(byte_range: ByteRequest | None, total_size: int | None) -> tuple[int, int | None]:
    """
    Convert a zarr ByteRequest into an explicit (start, stop) pair, in the same spirit as
    zarr's own (private) `_normalize_byte_range_index` helper.

    `stop` may be `None` if `total_size` isn't known and the request is open-ended
    (no byte_range, or an OffsetByteRequest) - callers that need a concrete stop
    (e.g. to build an HTTP Range header) must supply `total_size` in that case.
    """
    if byte_range is None:
        return 0, total_size
    elif isinstance(byte_range, RangeByteRequest):
        return byte_range.start, byte_range.end
    elif isinstance(byte_range, OffsetByteRequest):
        return byte_range.offset, total_size
    elif isinstance(byte_range, SuffixByteRequest):
        if total_size is None:
            raise ValueError(
                "A SuffixByteRequest requires a known total size.")
        return max(0, total_size - byte_range.suffix), total_size
    else:
        raise ValueError(f"Unexpected byte_range, got {byte_range!r}.")


# ---------------------------------------------------------------------------
# sync(): the one platform-specific piece of plumbing.
#
# Everything in this module (and everywhere else in brimfile, via
# `from .file_abstraction import sync`) works exclusively against zarr's async
# API (`zarr.api.asynchronous`, `zarr.AsyncArray`, `zarr.AsyncGroup`) and
# bridges to a synchronous call with this single `sync()` function - never
# through zarr's own synchronous facade (`zarr.Array`/`zarr.Group`), whose
# built-in `sync()` (zarr.core.sync.sync) spins up a real OS thread to host a
# second event loop. That works fine on CPython but Pyodide can't create real
# threads, so the Pyodide branch below uses a busy-loop that repeatedly ticks
# the *same* event loop instead - which only works because the surrounding
# Pyodide runtime supports JSPI (WebAssembly.Suspending), letting it actually
# suspend/resume while an underlying JS Promise (e.g. a `fetch()` or
# `Blob.arrayBuffer()`) resolves, rather than spinning uselessly.
# ---------------------------------------------------------------------------
if "pyodide" in sys.modules:
    import pyodide  # noqa: F401
    import js  # noqa: F401

    async def _awaitable_wrapper(coro):
        return await coro

    def sync(coro):
        """
        Synchronously run an asynchronous coroutine.
        """
        loop = asyncio.get_event_loop()
        task = loop.create_task(_awaitable_wrapper(coro))
        # In Pyodide, we can't block the event loop, so instead we yield back
        # control until the task is done.
        while not task.done():
            loop.run_until_complete(asyncio.sleep(0))
        return task.result()
else:
    def sync(coro):
        """
        Synchronously run an asynchronous coroutine.
        """
        return zarr.core.sync.sync(coro)


# ---------------------------------------------------------------------------
# Browser-backed zarr Stores (Pyodide only).
#
# These are the *only* new pieces of code needed to read a browser-picked
# file/folder or a remote URL: everything else (attributes, groups, array
# shape/dtype/indexing, codec decoding) is the real `zarr` package, shared
# with the CPython branch. None of these classes import `js`/`pyodide` at
# module scope, so they're importable and independently testable on CPython
# too, given a JsProxy-shaped stand-in (see tests/test_browser_stores.py).
# ---------------------------------------------------------------------------

async def _read_whole_js_file(js_file) -> bytes:
    """
    Read an entire browser `File`/`Blob` into memory as `bytes`, via a single
    JS call.

    Used for the ZIP case below: `zarr.storage.ZipStore` needs a real,
    synchronously-seekable stream to hand to stdlib `zipfile`, and an earlier
    version of this code provided that lazily (via a `_JsFileIO` class doing
    small `sync()`-bridged reads on demand). That doesn't actually work on
    *either* platform: `zipfile` performs those reads from deep inside
    `ZipStore._sync_open()`, itself already running inside a coroutine that a
    prior `sync()` call is in the middle of resolving, and re-entering the
    same bridge from there is exactly the "call sync() from within a running
    loop" case both platforms' `sync()` correctly refuse (CPython's
    thread-based one raises `zarr.core.sync.SyncError` for it explicitly;
    Pyodide's busy-loop one would deadlock/raise `RuntimeError` the same way,
    since a loop can't `run_until_complete()` while already running).
    Eagerly buffering the whole file up front - a single `sync()` call made
    directly from `_zarrFile.__init__`, i.e. from ordinary, non-async calling
    code rather than from inside a running coroutine - sidesteps the problem
    entirely, at the cost of holding the whole ZIP in memory for the lifetime
    of the file (a similar trade-off to today's JS `ZipStore.fromBlob`, and
    one `_zarrFile`'s own docstring already steers people away from for large
    files). A genuinely lazy version is possible on Pyodide using
    `pyodide.ffi.run_sync` (a JSPI-backed primitive built for exactly this
    "call sync, mid-stack, from already-async code" case, available since
    Pyodide 0.27.7) if that turns out to matter in practice.
    """
    size = int(js_file.size)
    buf = await js_file.slice(0, size).arrayBuffer()
    return buf.to_bytes()


class _BrowserFolderStore(Store):
    """
    Read-only zarr Store backed by a JS Array of `File` objects picked via a
    browser directory input (`<input type="file" webkitdirectory>`).

    The caller is expected to have already normalized the browser `FileList`
    into a plain JS Array (e.g. `Array.from(fileList)`) before it reaches
    Python, for reliable indexing/length access across pyodide versions.
    Each file's `webkitRelativePath` (falling back to `relativePath`, then
    `name`) is used to reconstruct the folder layout; the top-level folder
    name itself is stripped so the resulting keys match what zarr expects
    relative to the dataset root.
    """
    supports_writes = False
    supports_deletes = False
    supports_listing = True
    supports_partial_writes = False

    def __init__(self, files, *, read_only: bool = True, sync_fn=None):
        super().__init__(read_only=read_only)
        self._sync = sync_fn or sync
        self._by_key: dict[str, object] = {}
        root_name = None
        n = int(files.length)
        for i in range(n):
            f = files[i]
            raw_path = (
                getattr(f, "webkitRelativePath", None)
                or getattr(f, "relativePath", None)
                or f.name
            )
            raw_path = str(raw_path).replace("\\", "/").lstrip("/")
            parts = raw_path.split("/", 1)
            if root_name is None and len(parts) > 1:
                root_name = parts[0]
            key = parts[1] if len(parts) > 1 else parts[0]
            if key:
                self._by_key[key] = f
        self.root_name = root_name or "zarr_folder"

    def __eq__(self, other):
        return isinstance(other, _BrowserFolderStore) and other._by_key is self._by_key

    async def get(self, key, prototype, byte_range=None):
        f = self._by_key.get(key)
        if f is None:
            return None
        size = int(f.size)
        start, stop = _resolve_byte_range(byte_range, size)
        if stop is None:
            stop = size
        buf = await f.slice(start, stop).arrayBuffer()
        return prototype.buffer.from_bytes(buf.to_bytes())

    async def get_partial_values(self, prototype, key_ranges):
        return [await self.get(key, prototype, rng) for key, rng in key_ranges]

    async def exists(self, key) -> bool:
        return key in self._by_key

    async def set(self, key, value):
        raise NotImplementedError(
            "_BrowserFolderStore is read-only")

    async def delete(self, key):
        raise NotImplementedError(
            "_BrowserFolderStore is read-only")

    async def list(self):
        for k in self._by_key:
            yield k

    async def list_prefix(self, prefix: str):
        for k in self._by_key:
            if k.startswith(prefix):
                yield k

    async def list_dir(self, prefix: str):
        p = prefix
        if p and not p.endswith("/"):
            p += "/"
        plen = len(p)
        seen = set()
        for k in self._by_key:
            if not k.startswith(p):
                continue
            rest = k[plen:]
            if not rest:
                continue
            top = rest.split("/", 1)[0]
            if top not in seen:
                seen.add(top)
                yield top


class _BrowserFetchStore(Store):
    """
    Read-only zarr Store that fetches byte ranges over HTTP - for URLs
    (including public, anonymously-readable S3-compatible buckets) reachable
    directly from the browser.

    Byte fetching and bucket listing are injected as plain async callables
    (`fetch_bytes`/`list_keys`) so this class is independently testable on
    CPython with a stand-in; production leaves them as `None` and gets the
    real, pyodide-only implementations (`pyodide.http.pyfetch`-based) below.
    """
    supports_writes = False
    supports_deletes = False
    supports_listing = True
    supports_partial_writes = False

    def __init__(self, url: str, *, read_only: bool = True, fetch_bytes=None, list_keys=None):
        super().__init__(read_only=read_only)
        self._base = url.rstrip("/") + "/"
        self._fetch_bytes = fetch_bytes or _pyodide_fetch_bytes
        self._list_keys = list_keys or _pyodide_s3_list_keys

    def __eq__(self, other):
        return isinstance(other, _BrowserFetchStore) and other._base == self._base

    async def get(self, key, prototype, byte_range=None):
        start, stop = _resolve_byte_range(byte_range, None)
        headers = {}
        if start or stop is not None:
            end_part = "" if stop is None else str(stop - 1)
            headers["Range"] = f"bytes={start}-{end_part}"
        data = await self._fetch_bytes(self._base + key, headers)
        if data is None:
            return None
        return prototype.buffer.from_bytes(data)

    async def get_partial_values(self, prototype, key_ranges):
        return [await self.get(key, prototype, rng) for key, rng in key_ranges]

    async def exists(self, key) -> bool:
        return await self.get(key, _default_prototype()) is not None

    async def set(self, key, value):
        raise NotImplementedError(
            "_BrowserFetchStore is read-only")

    async def delete(self, key):
        raise NotImplementedError(
            "_BrowserFetchStore is read-only")

    async def list(self):
        for k in await self._list_keys(self._base, ""):
            yield k

    async def list_prefix(self, prefix: str):
        for k in await self._list_keys(self._base, prefix):
            if k.startswith(prefix):
                yield k

    async def list_dir(self, prefix: str):
        for k in await self._list_keys(self._base, prefix):
            yield k


def _default_prototype():
    from zarr.core.buffer import default_buffer_prototype
    return default_buffer_prototype()


async def _pyodide_fetch_bytes(url: str, headers: dict) -> bytes | None:
    """Real, pyodide-only byte-range fetch, used by `_BrowserFetchStore` in production."""
    from pyodide.http import pyfetch
    response = await pyfetch(url, headers=headers)
    if response.status == 404:
        return None
    response.raise_for_status()
    return await response.bytes()


def _s3_list_url_and_prefix(base_url: str, prefix: str) -> tuple[str, str]:
    """Build the ListObjectsV2 request URL (with query string) for `base_url`/`prefix`."""
    from urllib.parse import urlencode

    parsed = _parse_storage_url(base_url)
    list_url = f"{parsed['protocol']}://{parsed['endpoint']}/{parsed['bucket']}"
    object_prefix = (parsed['object_path'].rstrip('/') +
                      '/' + prefix) if parsed['object_path'] else prefix
    query = urlencode(
        {"list-type": "2", "delimiter": "/", "prefix": object_prefix})
    return f"{list_url}?{query}", object_prefix


def _parse_s3_list_response(xml_text: str) -> list[str]:
    """
    Pure parsing of a ListObjectsV2 (`?list-type=2&delimiter=/`) XML response into
    the immediate child names one level below the requested prefix - the same
    thing the JS `ZarrFile`'s `#list_S3keys` extracts from `<CommonPrefixes>`, plus
    same-level object keys from `<Contents>`. Kept separate from the actual fetch
    so it can be tested against a real response without a pyodide runtime.
    """
    import xml.etree.ElementTree as ET

    root = ET.fromstring(xml_text)

    def _local(tag):
        return tag.split('}', 1)[-1]

    keys = []
    for el in root:
        if _local(el.tag) == "CommonPrefixes":
            for child in el:
                if _local(child.tag) == "Prefix":
                    p = child.text or ""
                    if p.endswith('/'):
                        p = p[:-1]
                    keys.append(p.rsplit('/', 1)[-1])
        elif _local(el.tag) == "Contents":
            for child in el:
                if _local(child.tag) == "Key":
                    k = child.text or ""
                    keys.append(k.rsplit('/', 1)[-1])
    return keys


async def _pyodide_s3_list_keys(base_url: str, prefix: str) -> list[str]:
    """
    Real, pyodide-only S3 "list objects, one level deep" for `_BrowserFetchStore`,
    mirroring the ListObjectsV2 (`?list-type=2&delimiter=/&prefix=...`) query the
    JS ZarrFile class already used for the same purpose in the browser.
    """
    from pyodide.http import pyfetch

    url, _ = _s3_list_url_and_prefix(base_url, prefix)
    response = await pyfetch(url)
    response.raise_for_status()
    text = await response.string()
    return _parse_s3_list_response(text)


class _zarrFile(FileAbstraction):
    """
    The single `FileAbstraction` implementation, backed by the real `zarr`
    package on every platform. Only the `Store` it opens differs:

    - a string `source` behaves exactly as before: `zarr.storage.LocalStore`,
      `zarr.storage.ZipStore`, or (on CPython) `zarr.storage.FsspecStore` for
      `http(s)://`/`s3://` URLs.
    - a non-string `source` (Pyodide only) is a browser `File` (`store_type=ZIP`,
      buffered once via `_read_whole_js_file` into the same real
      `zarr.storage.ZipStore`) or a JS Array of `File` from a directory picker
      (`store_type=FOLDER`, via `_BrowserFolderStore`); on Pyodide a string
      `source` still auto-detects to `S3` for `http(s)://` URLs exactly like
      CPython, just backed by `_BrowserFetchStore` instead of `FsspecStore`.

    Every other method below (attributes, groups, datasets, listing) is
    platform-agnostic: it only ever calls into the real, async `zarr` API via
    `self._root`, exactly as it did when this was the CPython-only branch.
    """

    def __init__(self, filename, mode: str = 'r',
                 store_type: StoreType = StoreType.AUTO, *, version: Version = None):
        """
        Initialize the Zarr file.

        Args:
            filename (str): Path to the Zarr file. On Pyodide, this may
                alternatively be a browser `File` (with `store_type=ZIP`) or a
                JS Array of `File` from a directory picker (with
                `store_type=FOLDER`); mode is always treated as read-only in
                that case.
            mode: {'r', 'r+', 'a', 'w', 'w-'} the mode for opening the file (default is 'r' for read-only).
                    'r' means read only (must exist); 'r+' means read/write (must exist);
                    'a' means read/write (create if doesn't exist); 'w' means create (overwrite if exists); 'w-' means create (fail if exists).
            store_type (str): Type of the store to use. Default is 'AUTO'. Must be explicit
                (`ZIP` or `FOLDER`) for a non-string `filename`.
            version (Version): Version of the file format to use. Default is None.
        """
        st = StoreType

        if isinstance(filename, str):
            if store_type == st.ZIP:
                if not filename.endswith('.zip'):
                    filename += '.zip'
            elif store_type == st.ZARR:
                if not filename.endswith('.zarr'):
                    filename += '.zarr'
            elif store_type == st.AUTO:
                if filename.startswith('http') or filename.startswith('s3'):
                    store_type = st.S3
                elif filename.endswith('.zip'):
                    store_type = st.ZIP
                elif filename.endswith('.zarr'):
                    store_type = st.ZARR
                else:
                    raise ValueError(
                        "When using 'auto' store_type, the filename must end with '.zip' or '.zarr' or start with 'http' or 's3'.")

            if mode not in ['r', 'r+', 'a', 'w', 'w-']:
                raise ValueError(
                    f"Invalid mode '{mode}'. Supported modes are 'r', 'r+', 'a', 'w', and 'w-'.")

            match store_type:
                case st.ZIP:
                    mode_zip = mode
                    if mode == 'w-':
                        mode_zip = 'x'
                    elif mode == 'r+':
                        mode_zip = 'a'
                    store = zarr.storage.ZipStore(filename, mode=mode_zip)
                case st.ZARR:
                    # TODO: Add support for the other modes
                    store = zarr.storage.LocalStore(
                        filename, read_only=(mode == 'r'))
                case st.S3:
                    if "pyodide" in sys.modules:
                        store = _BrowserFetchStore(
                            filename, read_only=(mode == 'r'))
                    else:
                        if importlib.util.find_spec('fsspec') is None:
                            raise ModuleNotFoundError(
                                "The fsspec module is required for using S3 storage")
                        import fsspec
                        parsed_url = _parse_storage_url(filename)

                        fs = fsspec.filesystem('s3', anon=True, asynchronous=True,
                                               client_kwargs={'endpoint_url': f"{parsed_url['protocol']}://{parsed_url['endpoint']}"})

                        store = zarr.storage.FsspecStore(fs, path=f"{parsed_url['bucket']}/{parsed_url['object_path']}",
                                                          read_only=(mode == 'r'))
                case _:
                    raise ValueError(
                        f"Unsupported store type '{store_type}'. Supported types are 'zip', 'zarr', and 'remote'.")
        else:
            if "pyodide" not in sys.modules:
                raise TypeError(
                    f"filename must be a string, got {type(filename)}")
            mode = 'r'
            match store_type:
                case st.ZIP:
                    filename_out = str(filename.name)
                    file_bytes = sync(_read_whole_js_file(filename))
                    store = zarr.storage.ZipStore(
                        io.BytesIO(file_bytes), mode='r')
                case st.FOLDER:
                    browser_store = _BrowserFolderStore(filename)
                    filename_out = browser_store.root_name
                    store = browser_store
                case _:
                    raise ValueError(
                        f"store_type must be 'ZIP' or 'FOLDER' for a non-string source, got {store_type}")
            filename = filename_out

        self._root = sync(zarr_async.open_group(store=store, mode=mode))
        self._store = store
        self.filename = filename
        self.version = version

        # -------------------- Attribute Management --------------------

    @staticmethod
    def _to_ZarrArray(obj: zarr.AsyncArray):
        """"
        Add attributes to Zarr.AsyncArray object to support numpy indexing and slicing,
        bridged through the platform's `sync()` rather than zarr's own (thread-based,
        Pyodide-incompatible) synchronous facade (`zarr.Array`).

        N.B. this is a temporary fix to make existing code compatible with zarr.AsyncArray
             Don't add any new functionality here and consider changing it in the future!
        """
        class _ZarrArray(zarr.AsyncArray):
            def __array__(self, dtype=None, copy=None):
                # TODO: implement dtype and copy
                # see https://numpy.org/doc/stable/user/basics.interoperability.html#dunder-array-interface
                return self[...]

            async def to_np_array(self, dtype=None, copy=None):
                # same as __array__ but using async code
                return np.array(await self.getitem(...))

            def __getitem__(self, index):
                return sync(self.getitem(index))
        # since @dataclass(frozen=True), we need to use object.__setattr__
        object.__setattr__(obj, '__class__', _ZarrArray)
        return obj

    async def create_attr(self, obj, name: str, data, **kwargs):
        for k in kwargs.keys():
            warnings.warn(
                f"'{k}' argument not supported by 'create_attr' in zarr")
        if isinstance(obj, str):
            obj = await self._root.getitem(obj)
        attrs = obj.attrs
        attrs[name] = data
        await obj.update_attributes(attrs)

    async def get_attr(self, obj, name: str):
        if isinstance(obj, str):
            obj = await self._root.getitem(obj)
        return obj.attrs[name]

    # -------------------- Group Management --------------------

    async def open_group(self, full_path: str, **kwargs):
        for k in kwargs.keys():
            warnings.warn(
                f"'{k}' argument not supported by 'open_group' in zarr")
        g = await self._root.getitem(full_path)
        return g

    async def create_group(self, full_path: str):
        g = await self._root.create_group(full_path)
        return g

    # -------------------- Dataset Management --------------------

    async def open_dataset(self, full_path: str):
        ds = await self._root.getitem(full_path)
        # "upgrade" the object to a _ZarrArray
        ds = _zarrFile._to_ZarrArray(ds)
        return ds

    async def create_dataset(self, parent_group, name: str, data, chunk_size=None, compression: 'FileAbstraction.Compression' = FileAbstraction.Compression()):
        if isinstance(parent_group, str):
            parent_group = await self.open_group(parent_group)
        compressor = None
        if chunk_size is None:
            chunk_size = 'auto'
        if compression is not None:
            compressor = compression.to_zarr_compressor()
        ds = await parent_group.create_array(
            name=name, data=data,
            chunks=chunk_size, compressors=compressor)
        # "upgrade" the object to a _ZarrArray
        ds = _zarrFile._to_ZarrArray(ds)
        return ds

    # -------------------- Listing --------------------

    async def list_objects(self, obj):
        if isinstance(obj, str):
            obj = await self._root.getitem(obj)
        return tuple([str(el) async for el in obj.keys()])

    async def object_exists(self, full_path) -> bool:
        return await self._root.contains(full_path)

    async def list_attributes(self, obj):
        if isinstance(obj, str):
            obj = await self._root.getitem(obj)
        return (str(attr) for attr in obj.attrs.keys())

    # -------------------- File Management --------------------

    def close(self):
        self._store.close()

    # -------------------- Properties --------------------

    async def is_read_only(self) -> bool:
        return self._store.read_only


_AbstractFile = _zarrFile
