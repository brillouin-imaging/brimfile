"""
End-to-end test of the public `brim.File(...)` entry point through the
Pyodide-only "browser source" path (`store_type=StoreType.ZIP` or `FOLDER`
with a non-string `filename`), using the same genuinely-valid brim file the
rest of the test suite already relies on (`simple_brim_file`), rather than a
hand-built minimal one.

This monkeypatches `sys.modules["pyodide"]` to a harmless sentinel so
`_zarrFile.__init__`'s `"pyodide" not in sys.modules` runtime guard takes the
browser-source branch - it does *not* need a real `pyodide`/`js` module,
since `brimfile.file_abstraction` only imports those at module *import*
time (already done, on this platform, before this test ever runs), and
that's the only place they're referenced.
"""
import os
import io
import sys
import zipfile

import brimfile as brim
from brimfile.file_abstraction import StoreType


class _FakeJsArrayBuffer:
    def __init__(self, data: bytes):
        self._data = data

    def to_bytes(self) -> bytes:
        return self._data


class _FakeJsBlob:
    def __init__(self, data: bytes):
        self._data = data

    async def arrayBuffer(self):
        return _FakeJsArrayBuffer(self._data)


class _FakeJsFile:
    def __init__(self, name: str, data: bytes, webkitRelativePath: str | None = None):
        self.name = name
        self.size = len(data)
        self._data = data
        if webkitRelativePath is not None:
            self.webkitRelativePath = webkitRelativePath

    def slice(self, start: int, stop: int) -> _FakeJsBlob:
        return _FakeJsBlob(self._data[start:stop])


class _FakeJsArray:
    def __init__(self, items):
        self._items = list(items)

    @property
    def length(self):
        return len(self._items)

    def __getitem__(self, i):
        return self._items[i]


def _as_fake_zip_file(root_dir: str) -> _FakeJsFile:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for dirpath, _dirnames, filenames in os.walk(root_dir):
            for name in filenames:
                full = os.path.join(dirpath, name)
                arcname = os.path.relpath(full, root_dir)
                zf.write(full, arcname)
    return _FakeJsFile(os.path.basename(root_dir) + ".zip", buf.getvalue())


def _as_fake_file_array(root_dir: str) -> _FakeJsArray:
    items = []
    root_name = os.path.basename(root_dir)
    for dirpath, _dirnames, filenames in os.walk(root_dir):
        for name in filenames:
            full = os.path.join(dirpath, name)
            rel = os.path.join(root_name, os.path.relpath(full, root_dir))
            with open(full, "rb") as fh:
                data = fh.read()
            items.append(_FakeJsFile(
                name, data, webkitRelativePath=rel.replace(os.sep, "/")))
    return _FakeJsArray(items)


def test_file_opens_from_a_browser_zip_file(monkeypatch, simple_brim_file):
    monkeypatch.setitem(sys.modules, "pyodide", object())
    fake_file = _as_fake_zip_file(simple_brim_file)

    f = brim.File(fake_file, store_type=StoreType.ZIP)
    try:
        assert f.is_valid()
        d = f.get_data()
        assert d is not None
    finally:
        f.close()


def test_file_opens_from_a_browser_directory_picker(monkeypatch, simple_brim_file):
    monkeypatch.setitem(sys.modules, "pyodide", object())
    fake_files = _as_fake_file_array(simple_brim_file)

    f = brim.File(fake_files, store_type=StoreType.FOLDER)
    try:
        assert f.is_valid()
        d = f.get_data()
        assert d is not None
    finally:
        f.close()
