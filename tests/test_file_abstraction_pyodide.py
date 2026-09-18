"""
Tests for the Pyodide branch of `_zarrFile` (`brimfile.file_abstraction`,
the non-string `filename` path: a browser `File` (`StoreType.ZIP`), a
directory picker's file list (`StoreType.FOLDER`), or a URL (`StoreType.S3`,
same as CPython but backed by `_BrowserFetchStore`).

Unlike the old design, this branch no longer wraps a separate JS `ZarrFile`
class (`src/js/zarr_file.js`) - it uses the real `zarr` package directly, via
`_BrowserFolderStore`/`_BrowserFetchStore`/`_read_whole_js_file`
(`brimfile.file_abstraction`). Most of that new code is already covered
without a real pyodide runtime at all, using plain Python stand-ins for the
JsProxy shapes it touches - see `tests/test_browser_stores.py` and
`tests/test_file_browser_source.py`. What's left here, needing a genuine
pyodide/WASM runtime, is real `zarr`/`numcodecs` actually loading and running
inside it - driven via a small Node.js script (`tests/js/pyodide_driver.mjs`)
that loads pyodide (npm package, pinned to 0.29.x - see
tests/js/package.json), mounts the repo's `src/` directory into it, and
executes a list of operations against a real `_AbstractFile(js_source, ...)`.
All assertions happen here in Python.

IMPORTANT CAVEAT, discovered while writing this: `zarr` 3.x (and a matching
`numcodecs>=0.14`) is only bundled in Pyodide's *new* CalVer-style release
line, starting at Pyodide `314.0.1` (June 26, 2026) - confirmed directly
against https://pyodide.org/en/314.0.1/usage/packages-in-pyodide.html, which
lists `zarr 3.2.1` + `numcodecs 0.15.1`. (It was briefly missing from the very
first `314.0.0` release 17 days earlier due to a build issue, then fixed.)
The *older* `0.29.x` line pinned here (and by BrimView) never had zarr 3.x at
all: `0.27.0` bundled the old `zarr 2.18.3`, and by `0.29.1` zarr had been
dropped from the distribution entirely - so `micropip.install("zarr")` on
`0.29.x` would need to fetch a pure-Python wheel from PyPI, which then needs
`numcodecs>=0.14`, but `0.29.x` bundles the older, incompatible
`numcodecs==0.13.1` with no newer Emscripten-tagged wheel published to PyPI
to satisfy an upgrade. **In short: this refactor's Pyodide branch needs
Pyodide >= 314.0.1**, which for BrimView means upgrading the pinned Pyodide
version in `build_webapp.py` (currently `0.29.3`) - a separate, larger piece
of work (new Python 3.14, and Panel/Bokeh/JSPI compatibility with that
pyodide line would need checking) beyond this refactor itself.

Requires a one-time `npm install` under tests/js (see tests/README.md), a
pinned pyodide version of at least `314.0.1` there (this repo's own
`tests/js/package.json` still pins the older `~0.29.0` for now - bump it
alongside the actual runtime upgrade), and, unlike the rest of this suite,
genuine network access to Pyodide's package CDN (`cdn.jsdelivr.net`) - these
tests were not able to run end-to-end in the environment that wrote them for
that reason; see the module docstring above.
"""
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

import brimfile as brim
from brimfile.file_abstraction import sync as native_sync

REPO_ROOT = Path(__file__).resolve().parents[1]
JS_DIR = REPO_ROOT / "tests" / "js"
DRIVER = JS_DIR / "pyodide_driver.mjs"
RESULT_MARKER = "<<<r>>>\n"
PREFLIGHT_ERROR_MARKER = "<<<preflight_error>>>\n"


def _node_env_available() -> bool:
    return shutil.which("node") is not None and (JS_DIR / "node_modules").exists()


pytestmark = [
    pytest.mark.pyodide,
    pytest.mark.skipif(
        not _node_env_available(),
        reason="Node.js + `npm install` under tests/js are required for pyodide "
        "_zarrFile tests (see tests/README.md)",
    ),
]


def run_ops(source_kind: str, fixture_root_or_url: str, ops: list[dict], timeout: float = 300) -> list[dict]:
    """Run `ops` against the real pyodide `_zarrFile` and return the list of
    per-op results/errors. Raises with a clear message if the real `zarr`/
    `numcodecs` preflight inside the driver failed (see this module's
    docstring for the known version-gap caveat on Pyodide 0.29.x)."""
    proc = subprocess.run(
        ["node", str(DRIVER), source_kind, str(REPO_ROOT / "src"), fixture_root_or_url],
        input=json.dumps({"ops": ops}),
        capture_output=True,
        text=True,
        cwd=JS_DIR,
        timeout=timeout,
    )
    if proc.stdout.startswith(PREFLIGHT_ERROR_MARKER):
        pytest.fail(
            "Real zarr/numcodecs could not be installed inside pyodide - see "
            "this module's docstring for the known version-gap caveat:\n"
            + proc.stdout[len(PREFLIGHT_ERROR_MARKER):]
        )
    if proc.returncode != 0:
        raise RuntimeError(
            f"pyodide driver failed (exit {proc.returncode}):\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    idx = proc.stdout.rfind(RESULT_MARKER)
    if idx == -1:
        raise RuntimeError(
            f"pyodide driver produced no result marker:\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    return json.loads(proc.stdout[idx + len(RESULT_MARKER):])


@pytest.fixture(params=["zip", "folder"])
def browser_source(request, simple_brim_file):
    """`(source_kind, fixture_root)` for the two local-file browser sources;
    `simple_brim_file` is always a `.zarr` directory (see conftest.py), which
    is exactly what `pyodide_driver.mjs`'s "zip"/"folder" modes expect: it
    zips it up itself for "zip", and reads it directly for "folder"."""
    return request.param, simple_brim_file


@pytest.fixture
def brim_file_url(simple_brim_file, zarr_http_server):
    """URL at which the `simple_brim_file` fixture is reachable, for
    `_BrowserFetchStore`'s real (pyodide-only) `pyodide.http.pyfetch`-based
    fetch/list logic to read from."""
    return f"{zarr_http_server}/{os.path.basename(simple_brim_file)}"


class TestPyodideZarrFileParity:
    """Cross-checks the pyodide `_zarrFile` (run inside an actual
    pyodide/Node runtime, via real zarr) against the native `_zarrFile`
    reading the identical on-disk file, so both are verified against each
    other rather than only against themselves."""

    def test_get_attr_root_brim_version(self, simple_brim_file, browser_source):
        source_kind, fixture_root = browser_source
        results = run_ops(
            source_kind, fixture_root,
            [{"method": "get_attr", "args": ["/", "brim_version"]}],
        )
        assert results[0]["ok"] is True

        f = brim.File(simple_brim_file)
        expected = native_sync(f._file.get_attr("/", "brim_version"))
        f.close()

        assert results[0]["value"] == expected

    def test_object_exists(self, browser_source):
        source_kind, fixture_root = browser_source
        results = run_ops(
            source_kind, fixture_root,
            [
                {"method": "object_exists", "args": ["Brillouin_data"]},
                {"method": "object_exists", "args": ["does_not_exist"]},
            ],
        )
        assert [r["value"] for r in results] == [True, False]

    def test_list_objects(self, simple_brim_file, browser_source):
        source_kind, fixture_root = browser_source
        results = run_ops(
            source_kind, fixture_root,
            [{"method": "list_objects", "args": ["Brillouin_data"], "post": "sorted_list"}],
        )
        assert results[0]["ok"] is True

        f = brim.File(simple_brim_file)
        expected = sorted(native_sync(f._file.list_objects("Brillouin_data")))
        f.close()

        assert results[0]["value"] == expected

    def test_open_dataset_shape_and_data(self, simple_brim_file, browser_source):
        source_kind, fixture_root = browser_source
        results = run_ops(
            source_kind, fixture_root,
            [
                {"method": "open_dataset", "args": ["Brillouin_data/Data_0/PSD"], "post": "shape"},
                {"method": "open_dataset", "args": ["Brillouin_data/Data_0/PSD"], "post": "to_list"},
            ],
        )
        assert results[0]["ok"] is True
        assert results[1]["ok"] is True

        f = brim.File(simple_brim_file)
        ds = native_sync(f._file.open_dataset("Brillouin_data/Data_0/PSD"))
        expected_shape = list(ds.shape)
        expected_data = np.asarray(ds).tolist()
        f.close()

        assert results[0]["value"] == expected_shape
        assert results[1]["value"] == expected_data

    def test_get_attr_missing_raises_keyerror(self, browser_source):
        source_kind, fixture_root = browser_source
        results = run_ops(
            source_kind, fixture_root,
            [{"method": "get_attr", "args": [
                "Brillouin_data/Data_0", "nonexistent_attr_xyz"]}],
        )
        assert results[0]["ok"] is False
        assert results[0]["error"].startswith("KeyError")

    def test_is_read_only_always_true(self, browser_source):
        source_kind, fixture_root = browser_source
        results = run_ops(source_kind, fixture_root,
                           [{"method": "is_read_only", "args": []}])
        assert results[0]["value"] is True


class TestPyodideS3Source:
    """The URL/S3 path (`_BrowserFetchStore`), against brimfile's existing
    local S3-emulating HTTP server."""

    def test_get_attr_root_brim_version(self, simple_brim_file, brim_file_url):
        results = run_ops(
            "url", brim_file_url,
            [{"method": "get_attr", "args": ["/", "brim_version"]}],
        )
        assert results[0]["ok"] is True

        f = brim.File(simple_brim_file)
        expected = native_sync(f._file.get_attr("/", "brim_version"))
        f.close()

        assert results[0]["value"] == expected

    def test_list_objects(self, simple_brim_file, brim_file_url):
        results = run_ops(
            "url", brim_file_url,
            [{"method": "list_objects", "args": ["Brillouin_data"], "post": "sorted_list"}],
        )
        assert results[0]["ok"] is True

        f = brim.File(simple_brim_file)
        expected = sorted(native_sync(f._file.list_objects("Brillouin_data")))
        f.close()

        assert results[0]["value"] == expected
