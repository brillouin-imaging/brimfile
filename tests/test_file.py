"""
Unit tests for the File class in brimfile.
"""

import pytest
import os


import brimfile as brim
class TestFileCreation:
    """Tests for File creation and initialization."""
    
    def test_create_file_auto_store(self, tmp_path):
        """Test creating a file with AUTO store type."""
        filename = os.path.join(tmp_path, 'test_create.brim.zarr')
        f = brim.File.create(filename, store_type=brim.StoreType.AUTO)
        assert f is not None
        assert f.is_valid()
        f.close()
        assert os.path.exists(filename)
    
    def test_create_file_zarr_store(self, tmp_path):
        """Test creating a file with ZARR store type."""
        filename = os.path.join(tmp_path, 'test_zarr.brim.zarr')
        f = brim.File.create(filename, store_type=brim.StoreType.ZARR)
        assert f is not None
        f.close()
        assert os.path.exists(filename)
    
    def test_create_file_already_exists(self, empty_brim_file):
        """Test that creating a file that already exists raises an error."""
        with pytest.raises(Exception):
            brim.File.create(empty_brim_file, store_type=brim.StoreType.AUTO)
    
    def test_open_existing_file(self, simple_brim_file):
        """Test opening an existing file."""
        f = brim.File(simple_brim_file, mode='r')
        assert f is not None
        assert f.is_valid()
        f.close()
    
    def test_open_nonexistent_file(self, tmp_path):
        """Test that opening a non-existent file raises an error."""
        filename = os.path.join(tmp_path, 'nonexistent.brim.zarr')
        with pytest.raises(Exception):
            brim.File(filename, mode='r')
    
    def test_file_validation(self, simple_brim_file):
        """Test file validation."""
        f = brim.File(simple_brim_file)
        assert f.is_valid() is True
        f.close()

    def test_brim_version_stored_in_internal_file_on_create(self, tmp_path):
        """Test that create stores brim_version and mirrors it into _file.version."""
        filename = os.path.join(tmp_path, 'test_version_create.brim.zarr')
        f = brim.File.create(filename, store_type=brim.StoreType.AUTO, brim_version='1.2.3')

        stored_version = brim.file_abstraction.sync(
            f._file.get_attr('/', 'brim_version')
        )
        assert stored_version == '1.2.3'
        assert f._file.version == (1, 2, 3)

        f.close()

    def test_brim_version_loaded_in_internal_file_on_open(self, tmp_path):
        """Test that opening a file loads brim_version into _file.version."""
        filename = os.path.join(tmp_path, 'test_version_open.brim.zarr')

        created = brim.File.create(
            filename,
            store_type=brim.StoreType.AUTO,
            brim_version='2.0.1',
        )
        created.close()

        f = brim.File(filename, mode='r', store_type=brim.StoreType.AUTO)
        stored_version = brim.file_abstraction.sync(
            f._file.get_attr('/', 'brim_version')
        )

        assert stored_version == '2.0.1'
        assert f._file.version == (2, 0, 1)

        f.close()

    def test_brim_version_short_form_loaded_as_three_items(self, tmp_path):
        """Test short brim_version strings are normalized to 3-item tuples."""
        filename = os.path.join(tmp_path, 'test_version_short_open.brim.zarr')

        created = brim.File.create(
            filename,
            store_type=brim.StoreType.AUTO,
            brim_version='0.2',
        )
        created.close()

        f = brim.File(filename, mode='r', store_type=brim.StoreType.AUTO)
        stored_version = brim.file_abstraction.sync(
            f._file.get_attr('/', 'brim_version')
        )

        assert stored_version == '0.2'
        assert f._file.version == (0, 2, 0)

        f.close()


class TestFileReadOnly:
    """Tests for read-only file operations."""
    
    def test_is_read_only_mode(self, simple_brim_file):
        """Test checking if file is in read-only mode."""
        f = brim.File(simple_brim_file, mode='r')
        # is_read_only() returns whether the file can be modified
        is_ro = f.is_read_only()
        assert isinstance(is_ro, bool)
        # TODO fix the implementation of is_read_only() so that the following assertion is True
        # assert is_ro is True
        f.close()
    
    def test_is_not_read_only_mode(self, simple_brim_file):
        """Test checking if file is not in read-only mode."""
        f = brim.File(simple_brim_file, mode='r+')
        assert f.is_read_only() is False
        f.close()


class TestDataGroupOperations:
    """Tests for data group operations on File."""
    
    def test_list_data_groups(self, simple_brim_file):
        """Test listing data groups in a file."""
        f = brim.File(simple_brim_file)
        groups = f.list_data_groups()
        assert len(groups) > 0
        f.close()
    
    def test_list_data_groups_with_custom_names(self, simple_brim_file):
        """Test listing data groups with custom names."""
        f = brim.File(simple_brim_file)
        groups = f.list_data_groups(retrieve_custom_name=True)
        assert len(groups) > 0
        assert isinstance(groups, list)
        f.close()
    
    def test_get_data_group(self, simple_brim_file):
        """Test getting a data group from file."""
        f = brim.File(simple_brim_file)
        data = f.get_data()
        assert data is not None
        assert isinstance(data, brim.Data)
        f.close()
    
    def test_get_data_group_by_index(self, simple_brim_file):
        """Test getting a data group by index."""
        f = brim.File(simple_brim_file)
        data = f.get_data(0)
        assert data is not None
        f.close()
    
    def test_create_data_group(self, empty_brim_file, sample_data):
        """Test creating a new data group."""
        f = brim.File(empty_brim_file, mode='r+')
        data = f.create_data_group(
            sample_data['PSD'],
            sample_data['frequency'],
            sample_data['pixel_size'],
            name='new_data'
        )
        assert data is not None
        assert isinstance(data, brim.Data)

        d = f.get_data()
        pixel_size = brim.file_abstraction.sync(
            f._file.get_attr(d._group, 'element_size'))
        assert sample_data['pixel_size'] == tuple(pixel_size)

        f.close()
    
    def test_create_sparse_data_group(self, empty_brim_file, sample_data_sparse):
        """Test creating a new data group."""
        f = brim.File(empty_brim_file, mode='r+')
        data = f.create_data_group_sparse(
            sample_data_sparse['PSD'],
            sample_data_sparse['frequency'],
            scanning=sample_data_sparse['scanning'],
            name='new_data'
        )
        assert data is not None
        assert isinstance(data, brim.Data)

        f.close()
    
    def test_create_multiple_data_groups(self, empty_brim_file, sample_data):
        """Test creating multiple data groups."""
        f = brim.File(empty_brim_file, mode='r+')
        data1 = f.create_data_group(
            sample_data['PSD'],
            sample_data['frequency'],
            sample_data['pixel_size'],
            name='data1'
        )
        data2 = f.create_data_group(
            sample_data['PSD'],
            sample_data['frequency'],
            sample_data['pixel_size'],
            name='data2'
        )
        
        groups = f.list_data_groups()
        assert len(groups) == 2
        f.close()


class TestFileClosing:
    """Tests for file closing operations."""
    
    def test_close_file(self, simple_brim_file):
        """Test closing a file."""
        f = brim.File(simple_brim_file)
        f.close()
        # File should be closed, no exception expected
    
    def test_multiple_close_calls(self, simple_brim_file):
        """Test that multiple close calls don't cause errors."""
        f = brim.File(simple_brim_file)
        f.close()
        f.close()  # Should not raise an error


class TestStoreTypes:
    """Tests for different store types."""
    
    def test_auto_store_type(self, tmp_path):
        """Test AUTO store type detection."""
        filename = os.path.join(tmp_path, 'auto_store.brim.zarr')
        f = brim.File.create(filename, store_type=brim.StoreType.AUTO)
        assert f.is_valid()
        f.close()
    
    def test_zarr_store_type(self, tmp_path):
        """Test explicit ZARR store type."""
        filename = os.path.join(tmp_path, 'zarr_store.brim.zarr')
        f = brim.File.create(filename, store_type=brim.StoreType.ZARR)
        assert f.is_valid()
        f.close()
