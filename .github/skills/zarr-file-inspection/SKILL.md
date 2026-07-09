---
name: zarr-file-inspection
description: Use whenever you need to open or inspect a .brim file's raw zarr content directly (bypassing the brimfile API) to independently verify group hierarchy, array shapes/dtypes, and attributes — and whenever you're unsure about the correct zarr API call, consulting https://zarr.readthedocs.io/en/stable/ rather than guessing.
license: MIT
---

# Direct zarr inspection of .brim files

A `.brim` file is a zarr hierarchy (brimfile depends on `zarr >= 3.1.1`; see `pyproject.toml` and
`src/brimfile/file_abstraction.py` for the exact store handling). Tests that only call brimfile's own accessors and
then assert against brimfile's own writers can pass even when both sides are subtly wrong relative to the spec. To
catch that class of bug, open the file directly with `zarr` and check the raw content — this skill covers how.

If you're ever unsure about the correct call for the installed zarr version (the API has real differences between
zarr v2 and v3, and even across v3 minor releases — synchronous vs. async access, `attrs` semantics, store types),
check https://zarr.readthedocs.io/en/stable/ instead of guessing from memory. Confirm the installed version first
(`python -c "import zarr; print(zarr.__version__)"`) since brimfile pins `zarr>=3.1.1` and the docs site defaults to
the latest release.

## Opening a `.brim` file directly

```python
import zarr

# .brim files are typically a directory store ending in .brim.zarr, opened read-only for inspection
root = zarr.open(filename, mode='r')

# top-level attributes (brim_version, Subtype, ...)
print(dict(root.attrs))

# navigate the hierarchy exactly as described in brim_file_specs.md
brillouin_data = root['Brillouin_data']
print(dict(brillouin_data.attrs))          # general Metadata lives here

data_0 = brillouin_data['Data_0']
print(dict(data_0.attrs))                  # Sparse, element_size, local Metadata overrides, ...
print(list(data_0.keys()))                 # child groups/arrays: PSD, Frequency, Scanning, Analysis_0, ...
```

## Checking array structure

```python
psd = data_0['PSD']
print(psd.shape, psd.dtype)                # confirm 4D (z,y,x,spectrum) for non-sparse vs. flattened for sparse

freq = data_0['Frequency']
print(freq.shape)                          # confirm broadcastability to psd.shape, not necessarily equal shape
```

## Checking groups vs. arrays

Use `zarr`'s own introspection rather than assuming — a name existing under a group doesn't tell you whether it's a
sub-group or an array:

```python
for key in data_0.keys():
    member = data_0[key]
    kind = "group" if isinstance(member, zarr.Group) else "array"
    print(key, kind)
```

(If this differs on the installed zarr version, check the docs for the current recommended way to distinguish
groups from arrays — older/newer APIs have offered `group_keys()`/`array_keys()` helpers that may or may not exist.)

## Checking the units convention

The spec attaches units either as a `units` attribute directly on an array, or — for scalar attributes — as a
sibling `{attribute_name}_units` string attribute at the same level. How a *scalar* metadata attribute itself is
represented can be version-specific (e.g. a flattened `Type.Field` key vs. a nested `Metadata` dict — check the
`brim-file-spec-conformance` skill for which representation applies to the file's declared version); the
units-sibling convention is what you're confirming here regardless of which representation is in play.

For a flattened-style representation:

```python
attrs = dict(data_0.attrs)
value = attrs.get('Experiment.Temperature')
units = attrs.get('Experiment.Temperature_units')
```

For a nested-`Metadata`-style representation, the same units convention applies one level deeper, e.g.
`attrs['Metadata']['Experiment'].get('Temperature_units')`. Don't assume one representation without checking which
the file's declared version actually uses.

## Checking the `Sparse` flag and related structure independently

```python
sparse = data_0.attrs.get('Sparse', False)   # spec: defaults to False if absent
if sparse:
    assert 'Cartesian_visualisation' in data_0['Scanning'] or 'Spatial_map' in data_0['Scanning']
else:
    # element_size + element_size_units are required for non-sparse data groups
    assert 'element_size' in data_0.attrs and 'element_size_units' in data_0.attrs
```

Don't hardcode assumptions about which zarr store class is in use (directory store, zip store, remote store via
`zarr[remote]`/`s3fs`) — brimfile supports multiple `StoreType`s (see `file_abstraction.py`). When a test needs to
open a store type other than a plain local directory, check the zarr docs' storage guide for the right constructor
rather than assuming `zarr.open(path)` is always sufficient.

## When to reach for this skill vs. just using brimfile

- Use brimfile's API (`File`, `Data`, `Metadata`, ...) to *set up* test fixtures — that's what it's for.
- Use direct zarr inspection to *verify* that what brimfile wrote (or what a hand-built spec-conformance fixture
  contains) matches the spec's literal structure — independent of whether brimfile's own read path also happens to
  agree. This is especially important for anything spec-critical that brimfile's read API might silently normalize
  or default away (e.g. a missing `Sparse` attribute defaulting to `False` in the Python API — check whether that
  default is actually absent on disk, not just absent from the returned Python value).
- If a discrepancy between "what zarr shows" and "what brimfile's API returns" appears, that's exactly the kind of
  spec-conformance bug worth a test for — pair this skill with the `brim-file-spec-conformance` skill to confirm
  which side (if either) is actually right per the spec.
