# Brimfile Tests

This directory contains the comprehensive test suite for the brimfile package.

## Test Structure

The test suite is organized into multiple test files, each focusing on different components:

- **`conftest.py`**: Shared pytest fixtures and test utilities
- **`test_file.py`**: Tests for the File class (file creation, opening, data group management)
- **`test_data.py`**: Tests for the Data class (spectrum retrieval, metadata access, analysis results)
- **`test_metadata.py`**: Tests for the Metadata class (adding/retrieving metadata, types, conversions)
- **`test_analysis_results.py`**: Tests for the AnalysisResults class (image retrieval, quantities, peak types)
- **`test_integration.py`**: Integration tests for complete workflows and edge cases
- **`test_utils.py`**: Tests for utility functions
- **`test_file_abstraction.py`**: Tests for the native (zarr-backed) `_zarrFile` implementation of `FileAbstraction` (`brimfile.file_abstraction`), across `StoreType.ZARR` and `StoreType.ZIP`
- **`test_file_abstraction_pyodide.py`**: Tests for the pyodide `_zarrFile` implementation of `FileAbstraction`, run inside a real pyodide runtime wrapping the real `src/js/zarr_file.js` `ZarrFile` class (see "Pyodide/JS tests" below)
- **`general.py`**: Original demonstration script (kept for reference)

## Running the Tests

### Install Test Dependencies

```bash
pip install pytest
```

### Run All Tests

```bash
# Run all tests with verbose output
pytest tests/ -v

# Run specific test file
pytest tests/test_file.py -v

# Run specific test class
pytest tests/test_file.py::TestFileCreation -v

# Run specific test
pytest tests/test_file.py::TestFileCreation::test_create_file_auto_store -v
```

### Pyodide/JS tests

`test_file_abstraction_pyodide.py` exercises the pyodide branch of `_zarrFile`
against a real pyodide runtime (running in Node.js, no browser needed) wrapping
the real `ZarrFile` class from `src/js/zarr_file.js`. These tests are skipped
automatically unless Node.js is installed and the JS test dependencies have
been installed once:

```bash
cd tests/js
npm install
```

This installs, under `tests/js/node_modules` (not tracked, isolated from
`pyproject.toml`):
- `pyodide` (pinned to the `0.29.x` minor version)
- pinned local copies of `zarrita`, `@zarrita/storage`, and `fast-xml-parser`
  (the same packages `zarr_file.js` imports from a CDN at runtime; a custom
  Node ESM loader, `tests/js/loader.mjs`, redirects those CDN imports to these
  local, version-pinned copies so the tests don't depend on the CDN being
  reachable/unchanged)

Once installed, `pytest tests/ -v` picks the pyodide tests up automatically
(marked `@pytest.mark.pyodide`). To run only them:

```bash
pytest tests/test_file_abstraction_pyodide.py -v
```

To skip them explicitly (e.g. if Node isn't available):

```bash
pytest tests/ -v -m "not pyodide"
```

Internally, `tests/js/pyodide_driver.mjs` is a thin Node.js script that loads
pyodide, mounts the repository's `src/` directory into pyodide's virtual
filesystem, wraps a real `ZarrFile` instance (reading test fixtures over a
local HTTP server started by the `zarr_http_server` fixture in `conftest.py`)
with `_AbstractFile(...)`, and executes a list of operations sent as JSON over
stdin, reporting raw JSON results back over stdout. All actual assertions are
made in Python, in `test_file_abstraction_pyodide.py` -- the Node script is a
mechanical executor only.

### Test Configuration

The test configuration is defined in the `[tool.pytest.ini_options]` section of `pyproject.toml`:


## Test Fixtures

The `conftest.py` file provides shared fixtures:

- **`sample_data`**: Generated sample spectral data for testing
- **`simple_brim_file`**: Pre-created brim file with sample data
- **`empty_brim_file`**: Empty brim file for testing creation operations


## Writing New Tests

When adding new tests:

1. Follow the naming convention: `test_*.py` for files, `Test*` for classes, `test_*` for methods
2. Use descriptive test names that explain what is being tested
3. Organize tests into logical classes
4. Use fixtures from `conftest.py` when possible
5. Clean up any created files (fixtures handle this automatically)
6. Add docstrings to explain complex test scenarios

Example:

```python
class TestNewFeature:
    """Tests for the new feature."""
    
    def test_basic_functionality(self, simple_brim_file):
        """Test basic functionality of the feature."""
        f = brim.File(simple_brim_file)
        # Test code here
        f.close()
```

## Known Test Warnings

Some tests may produce warnings that are expected:

- "No units provided for X; None is assumed" - Expected when metadata items lack units
- "Cannot close the file" - May occur in some edge cases, handled gracefully

These warnings do not indicate test failures and are part of normal operation.