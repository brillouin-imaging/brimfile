"""
Tests for the pyodide `_zarrFile` implementation of `FileAbstraction`
(`brimfile.file_abstraction`, the `if "pyodide" in sys.modules:` branch).

Unlike the native zarr backend, this branch can only be exercised inside a
real pyodide runtime wrapping the real `ZarrFile` class from
`src/js/zarr_file.js`. These tests drive that combination via a small Node.js
script (`tests/js/pyodide_driver.mjs`) that loads pyodide (npm package,
pinned to 0.29.x -- see tests/js/package.json), mounts the repo's `src/`
directory into it, and executes a list of operations against
`_AbstractFile(zarr_js, ...)`. All assertions happen here in Python.

Requires a one-time `npm install` under tests/js (see tests/README.md).
Tests are skipped automatically if Node.js or the installed dependencies are
not available.
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
ZARR_FILE_JS = REPO_ROOT / "src" / "js" / "zarr_file.js"
DRIVER = JS_DIR / "pyodide_driver.mjs"
RESULT_MARKER = "<<<RESULT>>>\n"


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


def run_ops(base_url: str, ops: list[dict], timeout: float = 180) -> list[dict]:
    """Run `ops` against the real pyodide `_zarrFile` (wrapping the real JS
    `ZarrFile`) and return the list of per-op results/errors."""
    proc = subprocess.run(
        ["node", str(DRIVER), str(ZARR_FILE_JS), str(REPO_ROOT / "src")],
        input=json.dumps({"base_url": base_url, "ops": ops}),
        capture_output=True,
        text=True,
        cwd=JS_DIR,
        timeout=timeout,
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


@pytest.fixture
def brim_file_url(simple_brim_file, zarr_http_server):
    """URL at which the `simple_brim_file` fixture is reachable, for the real
    JS `ZarrFile.init_from_url()` to read from."""
    return f"{zarr_http_server}/{os.path.basename(simple_brim_file)}"


class TestPyodideZarrFileParity:
    """Cross-checks the pyodide `_zarrFile` (wrapping the real zarr_file.js
    `ZarrFile`, run inside an actual pyodide/Node runtime) against the native
    zarr `_zarrFile` reading the identical on-disk file, so both backends are
    verified against each other rather than only against themselves."""

    def test_get_attr_root_brim_version(self, simple_brim_file, brim_file_url):
        results = run_ops(
            brim_file_url, [{"method": "get_attr", "args": ["/", "brim_version"]}]
        )
        assert results[0]["ok"] is True

        f = brim.File(simple_brim_file)
        expected = native_sync(f._file.get_attr("/", "brim_version"))
        f.close()

        assert results[0]["value"] == expected

    def test_object_exists(self, brim_file_url):
        results = run_ops(
            brim_file_url,
            [
                {"method": "object_exists", "args": ["Brillouin_data"]},
                {"method": "object_exists", "args": ["does_not_exist"]},
            ],
        )
        assert [r["value"] for r in results] == [True, False]

    def test_list_objects(self, simple_brim_file, brim_file_url):
        results = run_ops(
            brim_file_url,
            [{"method": "list_objects", "args": ["Brillouin_data"], "post": "sorted_list"}],
        )
        assert results[0]["ok"] is True

        f = brim.File(simple_brim_file)
        expected = sorted(native_sync(f._file.list_objects("Brillouin_data")))
        f.close()

        assert results[0]["value"] == expected

    def test_list_attributes(self, simple_brim_file, brim_file_url):
        results = run_ops(
            brim_file_url,
            [
                {
                    "method": "list_attributes",
                    "args": ["Brillouin_data/Data_0"],
                    "post": "sorted_list",
                }
            ],
        )
        assert results[0]["ok"] is True

        f = brim.File(simple_brim_file)
        expected = sorted(native_sync(f._file.list_attributes("Brillouin_data/Data_0")))
        f.close()

        assert results[0]["value"] == expected

    def test_open_dataset_shape_and_data(self, simple_brim_file, brim_file_url):
        results = run_ops(
            brim_file_url,
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

    def test_get_attr_missing_raises_keyerror(self, brim_file_url):
        results = run_ops(
            brim_file_url,
            [
                {
                    "method": "get_attr",
                    "args": ["Brillouin_data/Data_0", "nonexistent_attr_xyz"],
                }
            ],
        )
        assert results[0]["ok"] is False
        assert results[0]["error"].startswith("KeyError")

    def test_is_read_only_always_true(self, brim_file_url):
        # the JS ZarrFile backend is fetch-based and always read-only,
        # regardless of the underlying store's actual mode.
        results = run_ops(brim_file_url, [{"method": "is_read_only", "args": []}])
        assert results[0]["value"] is True
