"""
Tests for the browser-backed zarr `Store`/file-like adapters added for the
Pyodide branch of `_zarrFile` (`brimfile.file_abstraction`): `_JsFileIO`,
`_BrowserFolderStore`, and `_BrowserFetchStore`.

None of these classes import `js`/`pyodide` at module scope, by design: they
only ever call a small, documented slice of a JsProxy's shape (`.size`,
`.slice(start, stop).arrayBuffer()` returning something with `.to_bytes()`,
`.length` + indexing for a JS Array). That means they can be driven directly
against the *real* `zarr` package here, using small Python stand-ins that
duck-type that shape, without a real Pyodide/WASM runtime - which is exactly
what lets this file run as an ordinary (non-`pyodide`-marked) test.

What this does *not* cover: the pyodide-only glue in `_zarrFile.__init__`
that picks which of these classes to construct, and the two pyodide-only
byte/listing fetchers (`_pyodide_fetch_bytes`, `_pyodide_s3_list_keys`) that
call `pyodide.http.pyfetch` directly - those need a real pyodide runtime, and
are exercised instead by `tests/test_file_abstraction_pyodide.py` (see that
file's module docstring for how to run it, and why it needs network access
this environment did not have while writing these tests).
"""
import os

import numpy as np
import pytest

from brimfile.file_abstraction import (
    _AbstractFile,
    StoreType,
    sync,
    _read_whole_js_file,
    _BrowserFolderStore,
    _BrowserFetchStore,
    _parse_s3_list_response,
    _s3_list_url_and_prefix,
)
import io
import zarr
import zarr.api.asynchronous as zarr_async


# ---------------------------------------------------------------------------
# Minimal stand-ins for the JS objects these classes actually touch.
# ---------------------------------------------------------------------------

class _FakeJsArrayBuffer:
    """Stands in for the JsProxy of an ArrayBuffer: only `.to_bytes()` is used."""

    def __init__(self, data: bytes):
        self._data = data

    def to_bytes(self) -> bytes:
        return self._data


class _FakeJsBlob:
    """Stands in for the JsProxy of a Blob (what `File.slice()` returns)."""

    def __init__(self, data: bytes):
        self._data = data

    async def arrayBuffer(self):
        return _FakeJsArrayBuffer(self._data)


class _FakeJsFile:
    """Stands in for the JsProxy of a browser `File`."""

    def __init__(self, name: str, data: bytes, webkitRelativePath: str | None = None):
        self.name = name
        self.size = len(data)
        self._data = data
        if webkitRelativePath is not None:
            self.webkitRelativePath = webkitRelativePath

    def slice(self, start: int, stop: int) -> _FakeJsBlob:
        return _FakeJsBlob(self._data[start:stop])


class _FakeJsArray:
    """Stands in for a JS Array: `.length` + `__getitem__`, as `_BrowserFolderStore` expects
    (mirroring `Array.from(fileList)` being done on the JS side before reaching Python)."""

    def __init__(self, items):
        self._items = list(items)

    @property
    def length(self):
        return len(self._items)

    def __getitem__(self, i):
        return self._items[i]


# ---------------------------------------------------------------------------
# _JsFileIO + zarr.storage.ZipStore
# ---------------------------------------------------------------------------

class TestReadWholeJsFile:
    @pytest.fixture
    def zip_bytes(self, tmp_path):
        """Bytes of a real brim/zarr `.zip` file, written by the existing
        CPython `_AbstractFile` machinery (i.e. genuinely zarr-written, not
        hand-crafted), so opening it back up is an apples-to-apples
        comparison against `test_file_abstraction.py`'s own ZIP coverage."""
        filename = os.path.join(tmp_path, "sample")
        f = _AbstractFile(filename, mode="w", store_type=StoreType.ZIP)
        g = sync(f.create_group("g1"))
        sync(f.create_attr(g, "brim_version", "0.2"))
        data = np.arange(24).reshape(2, 3, 4).astype(float)
        sync(f.create_dataset(g, "arr", data))
        f.close()
        with open(filename + ".zip", "rb") as fh:
            return fh.read()

    def test_reads_the_whole_file(self, zip_bytes):
        got = sync(_read_whole_js_file(_FakeJsFile("sample.zip", zip_bytes)))
        assert got == zip_bytes

    def test_real_zarr_can_open_zipstore_over_the_buffered_bytes(self, zip_bytes):
        """The actual point of _read_whole_js_file: real zarr, via a real
        ZipStore, reading a browser File it never had a filesystem path for.

        (This buffers the whole file rather than streaming it lazily - see
        `_read_whole_js_file`'s docstring for why a lazy version doesn't work:
        `zipfile`'s own synchronous reads happen from deep inside a coroutine
        that a `sync()` call is already resolving, and re-entering `sync()`
        from there is a "call sync() from within a running loop" case both
        platforms' `sync()` correctly refuse.)"""
        js_file = _FakeJsFile("sample.zip", zip_bytes)
        file_bytes = sync(_read_whole_js_file(js_file))
        store = zarr.storage.ZipStore(io.BytesIO(file_bytes), mode="r")
        root = sync(zarr_async.open_group(store=store, mode="r"))
        g = sync(root.getitem("g1"))
        assert g.attrs["brim_version"] == "0.2"
        ds = sync(root.getitem("g1/arr"))
        assert ds.shape == (2, 3, 4)
        got = sync(ds.getitem((slice(None), slice(None), slice(None))))
        np.testing.assert_array_equal(got, np.arange(24).reshape(2, 3, 4))


# ---------------------------------------------------------------------------
# _BrowserFolderStore
# ---------------------------------------------------------------------------

class TestBrowserFolderStore:
    @pytest.fixture
    def folder_files(self, tmp_path):
        """A real zarr directory store (built the same way as ZARR-store
        tests elsewhere), turned into the JS Array of File-like stand-ins
        `_BrowserFolderStore` expects - as if picked via a directory input."""
        filename = os.path.join(tmp_path, "sample")
        f = _AbstractFile(filename, mode="w", store_type=StoreType.ZARR)
        g = sync(f.create_group("g1"))
        sync(f.create_attr(g, "brim_version", "0.2"))
        data = np.arange(24).reshape(2, 3, 4).astype(float)
        sync(f.create_dataset(g, "arr", data))
        f.close()

        root_dir = filename + ".zarr"
        root_name = os.path.basename(root_dir)
        items = []
        for dirpath, _dirnames, filenames in os.walk(root_dir):
            for name in filenames:
                full = os.path.join(dirpath, name)
                rel = os.path.relpath(full, os.path.dirname(root_dir))
                with open(full, "rb") as fh:
                    data_bytes = fh.read()
                items.append(_FakeJsFile(
                    name, data_bytes, webkitRelativePath=rel.replace(os.sep, "/")))
        return _FakeJsArray(items), root_name

    def test_root_name_is_derived_from_first_path_segment(self, folder_files):
        files, root_name = folder_files
        store = _BrowserFolderStore(files)
        assert store.root_name == root_name

    def test_real_zarr_can_open_folder_store(self, folder_files):
        files, _root_name = folder_files
        store = _BrowserFolderStore(files)
        root = sync(zarr_async.open_group(store=store, mode="r"))
        g = sync(root.getitem("g1"))
        assert g.attrs["brim_version"] == "0.2"
        ds = sync(root.getitem("g1/arr"))
        assert ds.shape == (2, 3, 4)
        got = sync(ds.getitem((slice(None), slice(None), slice(None))))
        np.testing.assert_array_equal(got, np.arange(24).reshape(2, 3, 4))

    def test_exists_and_missing_key(self, folder_files):
        files, _root_name = folder_files
        store = _BrowserFolderStore(files)
        assert sync(store.exists("g1/zarr.json")) is True
        assert sync(store.exists("does/not/exist")) is False
        assert sync(store.get("does/not/exist", zarr.core.buffer.default_buffer_prototype())) is None

    def test_read_only_store_rejects_writes(self, folder_files):
        files, _root_name = folder_files
        store = _BrowserFolderStore(files)
        with pytest.raises(NotImplementedError):
            sync(store.set("some/key", None))
        with pytest.raises(NotImplementedError):
            sync(store.delete("some/key"))


# ---------------------------------------------------------------------------
# _BrowserFetchStore - byte fetch/list logic, with an in-memory fake HTTP
# layer for pure store-behavior tests, and the real S3-list XML parser
# exercised against brimfile's existing local S3-emulating HTTP server.
# ---------------------------------------------------------------------------

class TestBrowserFetchStoreWithFakeTransport:
    @pytest.fixture
    def fake_remote(self, tmp_path):
        """An in-memory {key: bytes} store plus fetch_bytes/list_keys callables
        with the same signatures _BrowserFetchStore injects in production."""
        blobs = {
            "zarr.json": b'{"zarr_format": 3, "node_type": "group", "attributes": {}}',
            "g1/zarr.json": b'{"node_type": "group"}',
            "g1/arr/zarr.json": b'{"node_type": "array"}',
            "g1/arr/c/0": b"chunkdata",
        }

        async def fetch_bytes(url, headers):
            key = url.split("BASE/", 1)[1]
            data = blobs.get(key)
            if data is None:
                return None
            rng = headers.get("Range")
            if not rng:
                return data
            start_s, end_s = rng.removeprefix("bytes=").split("-")
            start = int(start_s)
            end = len(data) - 1 if end_s == "" else int(end_s)
            return data[start:end + 1]

        async def list_keys(base_url, prefix):
            seen = []
            plen = len(prefix)
            for k in blobs:
                if not k.startswith(prefix):
                    continue
                rest = k[plen:]
                top = rest.split("/", 1)[0]
                if top and top not in seen:
                    seen.append(top)
            return seen

        return blobs, fetch_bytes, list_keys

    def test_get_full_value(self, fake_remote):
        blobs, fetch_bytes, list_keys = fake_remote
        store = _BrowserFetchStore(
            "http://example/BASE", fetch_bytes=fetch_bytes, list_keys=list_keys)
        proto = zarr.core.buffer.default_buffer_prototype()
        buf = sync(store.get("g1/zarr.json", proto))
        assert buf.to_bytes() == blobs["g1/zarr.json"]

    def test_get_missing_returns_none(self, fake_remote):
        _blobs, fetch_bytes, list_keys = fake_remote
        store = _BrowserFetchStore(
            "http://example/BASE", fetch_bytes=fetch_bytes, list_keys=list_keys)
        proto = zarr.core.buffer.default_buffer_prototype()
        assert sync(store.get("does/not/exist", proto)) is None

    def test_get_byte_range_sets_range_header(self, fake_remote):
        blobs, fetch_bytes, list_keys = fake_remote
        store = _BrowserFetchStore(
            "http://example/BASE", fetch_bytes=fetch_bytes, list_keys=list_keys)
        proto = zarr.core.buffer.default_buffer_prototype()
        from zarr.abc.store import RangeByteRequest
        buf = sync(store.get("g1/arr/c/0", proto,
                              byte_range=RangeByteRequest(2, 5)))
        assert buf.to_bytes() == blobs["g1/arr/c/0"][2:5]

    def test_real_zarr_can_open_fetch_store(self, fake_remote):
        _blobs, fetch_bytes, list_keys = fake_remote
        store = _BrowserFetchStore(
            "http://example/BASE", fetch_bytes=fetch_bytes, list_keys=list_keys)
        root = sync(zarr_async.open_group(store=store, mode="r"))
        assert sync(root.contains("g1")) is True


class TestS3ListResponseParsing:
    """Exercises the real ListObjectsV2 XML parser used by the production,
    pyodide-only `_pyodide_s3_list_keys`, against brimfile's existing local
    S3-emulating HTTP server (`zarr_http_server`, from conftest.py) - so the
    one part of `_BrowserFetchStore` that can't be driven by a pyodide-free
    unit test (the actual `pyodide.http.pyfetch` call) is reduced to a single
    untested line, with the XML-handling logic around it fully covered here.
    """

    def test_parses_common_prefixes_for_a_real_zarr_directory(self, tmp_path, zarr_http_server):
        import urllib.request

        filename = os.path.join(tmp_path, "sample")
        f = _AbstractFile(filename, mode="w", store_type=StoreType.ZARR)
        g = sync(f.create_group("g1"))
        sync(f.create_group("g1/child_a"))
        sync(f.create_group("g1/child_b"))
        f.close()

        root_name = os.path.basename(filename + ".zarr")
        url, _ = _s3_list_url_and_prefix(
            f"{zarr_http_server}/{root_name}", "g1")
        with urllib.request.urlopen(url) as resp:
            xml_text = resp.read().decode("utf-8")

        keys = _parse_s3_list_response(xml_text)
        assert set(keys) == {"child_a", "child_b"}
