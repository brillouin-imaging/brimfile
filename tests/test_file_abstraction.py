"""
Tests for the native (zarr-backed) `_zarrFile` implementation of
`FileAbstraction`, defined in `brimfile.file_abstraction`.

These exercise the low-level abstraction layer directly (not through
`brimfile.File`), across both `StoreType.ZARR` and `StoreType.ZIP`.
"""
import os

import numpy as np
import pytest

from brimfile.file_abstraction import (
    _AbstractFile,
    FileAbstraction,
    StoreType,
    sync,
    _async_getitem,
)


@pytest.fixture(params=[StoreType.ZARR, StoreType.ZIP], ids=["zarr", "zip"])
def store_type(request):
    return request.param


@pytest.fixture
def writable_file(tmp_path, store_type):
    """A freshly created, writable `_AbstractFile` instance."""
    filename = os.path.join(tmp_path, "test_abstraction")
    f = _AbstractFile(filename, mode="w", store_type=store_type)
    yield f
    f.close()


class TestGroupManagement:
    def test_create_and_open_group(self, writable_file):
        f = writable_file
        g = sync(f.create_group("mygroup"))
        assert g is not None
        opened = sync(f.open_group("mygroup"))
        assert opened is not None

    def test_create_nested_group(self, writable_file):
        f = writable_file
        sync(f.create_group("parent"))
        g = sync(f.create_group("parent/child"))
        assert g is not None
        assert sync(f.object_exists("parent/child")) is True

    def test_open_nonexistent_group_raises(self, writable_file):
        f = writable_file
        with pytest.raises(Exception):
            sync(f.open_group("does_not_exist"))


class TestAttributeManagement:
    def test_create_and_get_attr_on_group_object(self, writable_file):
        f = writable_file
        g = sync(f.create_group("g1"))
        sync(f.create_attr(g, "brim_version", "0.2"))
        assert sync(f.get_attr(g, "brim_version")) == "0.2"

    def test_create_and_get_attr_by_path(self, writable_file):
        f = writable_file
        sync(f.create_group("g2"))
        sync(f.create_attr("g2", "answer", 42))
        assert sync(f.get_attr("g2", "answer")) == 42

    def test_get_attr_missing_raises_keyerror(self, writable_file):
        f = writable_file
        sync(f.create_group("g3"))
        with pytest.raises(KeyError):
            sync(f.get_attr("g3", "nonexistent"))

    def test_overwriting_attr_replaces_value(self, writable_file):
        f = writable_file
        sync(f.create_group("g4"))
        sync(f.create_attr("g4", "val", 1))
        sync(f.create_attr("g4", "val", 2))
        assert sync(f.get_attr("g4", "val")) == 2

    def test_adding_attr_does_not_clear_others(self, writable_file):
        f = writable_file
        sync(f.create_group("g5"))
        sync(f.create_attr("g5", "a", 1))
        sync(f.create_attr("g5", "b", 2))
        assert sync(f.get_attr("g5", "a")) == 1
        assert sync(f.get_attr("g5", "b")) == 2


class TestDatasetManagement:
    def test_create_and_open_dataset(self, writable_file):
        f = writable_file
        g = sync(f.create_group("data_grp"))
        data = np.arange(24).reshape(2, 3, 4).astype(float)
        sync(f.create_dataset(g, "arr", data))
        opened = sync(f.open_dataset("data_grp/arr"))
        assert opened.shape == data.shape
        np.testing.assert_array_equal(np.asarray(opened), data)

    def test_async_getitem_slice(self, writable_file):
        f = writable_file
        g = sync(f.create_group("data_grp2"))
        data = np.arange(24).reshape(2, 3, 4).astype(float)
        sync(f.create_dataset(g, "arr", data))
        ds = sync(f.open_dataset("data_grp2/arr"))
        sliced = sync(_async_getitem(ds, (0, slice(None), slice(None))))
        np.testing.assert_array_equal(sliced, data[0])

    def test_async_getitem_ellipsis(self, writable_file):
        f = writable_file
        g = sync(f.create_group("data_grp3"))
        data = np.arange(24).reshape(2, 3, 4).astype(float)
        sync(f.create_dataset(g, "arr", data))
        ds = sync(f.open_dataset("data_grp3/arr"))
        full = sync(_async_getitem(ds, Ellipsis))
        np.testing.assert_array_equal(np.asarray(full), data)

    def test_async_getitem_single_int_index(self, writable_file):
        f = writable_file
        g = sync(f.create_group("data_grp4"))
        data = np.arange(6).astype(float)
        sync(f.create_dataset(g, "arr", data))
        ds = sync(f.open_dataset("data_grp4/arr"))
        value = sync(_async_getitem(ds, 2))
        assert value == data[2]

    @pytest.mark.parametrize(
        "compression",
        [
            FileAbstraction.Compression(type=FileAbstraction.Compression.NONE),
            FileAbstraction.Compression(type=FileAbstraction.Compression.DEFAULT),
            FileAbstraction.Compression(type=FileAbstraction.Compression.ZLIB, level=1),
            FileAbstraction.Compression(type=FileAbstraction.Compression.BLOSC),
            FileAbstraction.Compression(type=FileAbstraction.Compression.GZIP, level=3),
            FileAbstraction.Compression(type=FileAbstraction.Compression.ZSTD, level=3),
        ],
        ids=["none", "default", "zlib", "blosc", "gzip", "zstd"],
    )
    def test_create_dataset_with_compression(self, writable_file, compression):
        f = writable_file
        g = sync(f.create_group("comp_grp"))
        data = np.arange(10).astype(float)
        ds = sync(f.create_dataset(g, "arr", data, compression=compression))
        np.testing.assert_array_equal(np.asarray(ds), data)

    def test_create_dataset_with_unsupported_compression_type_warns(self, writable_file):
        f = writable_file
        g = sync(f.create_group("comp_grp_bad"))
        data = np.arange(10).astype(float)
        compression = FileAbstraction.Compression(type="not-a-real-type")
        with pytest.warns(UserWarning, match="not supported by zarr"):
            ds = sync(f.create_dataset(g, "arr", data, compression=compression))
        np.testing.assert_array_equal(np.asarray(ds), data)

    def test_create_dataset_with_explicit_chunk_size(self, writable_file):
        f = writable_file
        g = sync(f.create_group("chunk_grp"))
        data = np.arange(20).astype(float)
        ds = sync(f.create_dataset(g, "arr", data, chunk_size=(5,)))
        np.testing.assert_array_equal(np.asarray(ds), data)


class TestListing:
    def test_list_objects(self, writable_file):
        f = writable_file
        sync(f.create_group("parent"))
        sync(f.create_group("parent/child1"))
        sync(f.create_group("parent/child2"))
        objs = sync(f.list_objects("parent"))
        assert set(objs) == {"child1", "child2"}

    def test_list_objects_empty_group(self, writable_file):
        f = writable_file
        sync(f.create_group("empty_grp"))
        assert tuple(sync(f.list_objects("empty_grp"))) == ()

    def test_object_exists(self, writable_file):
        f = writable_file
        sync(f.create_group("exists_grp"))
        assert sync(f.object_exists("exists_grp")) is True
        assert sync(f.object_exists("does_not_exist")) is False

    def test_list_attributes(self, writable_file):
        f = writable_file
        sync(f.create_group("attr_grp"))
        sync(f.create_attr("attr_grp", "a", 1))
        sync(f.create_attr("attr_grp", "b", 2))
        attrs = set(sync(f.list_attributes("attr_grp")))
        assert attrs == {"a", "b"}

    def test_list_attributes_none(self, writable_file):
        f = writable_file
        sync(f.create_group("no_attr_grp"))
        assert set(sync(f.list_attributes("no_attr_grp"))) == set()


class TestFileProperties:
    def test_is_read_only_false_when_writable(self, writable_file):
        assert sync(writable_file.is_read_only()) is False

    def test_invalid_mode_raises(self, tmp_path, store_type):
        filename = os.path.join(tmp_path, "bad_mode")
        with pytest.raises(ValueError):
            _AbstractFile(filename, mode="bogus", store_type=store_type)

    def test_auto_store_type_requires_recognizable_extension(self, tmp_path):
        filename = os.path.join(tmp_path, "no_extension_at_all")
        with pytest.raises(ValueError):
            _AbstractFile(filename, mode="w", store_type=StoreType.AUTO)
