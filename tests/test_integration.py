"""
Integration tests for brimfile - testing complete workflows.
"""

import pytest
import numpy as np
import os
from datetime import datetime


import brimfile as brim


class TestCompleteWorkflow:
    """Tests for complete read/write workflows."""
    
    def test_create_write_read_workflow(self, tmp_path, sample_data):
        """Test complete workflow: create file, write data, read it back."""
        filename = os.path.join(tmp_path, 'workflow_test.brim.zarr')
        
        # Create and write
        f = brim.File.create(filename, store_type=brim.StoreType.AUTO)
        
        data = f.create_data_group(
            sample_data['PSD'],
            sample_data['frequency'],
            sample_data['pixel_size'],
            name='test_workflow'
        )
        
        # Add metadata
        Attr = brim.Metadata.Item
        md = data.get_metadata()
        md.add(brim.Metadata.Type.Experiment, {
            'Datetime': datetime.now().isoformat(),
            'Temperature': Attr(22.0, 'C')
        })
        
        # Create analysis results
        ar = data.create_analysis_results_group(
            {
                'shift': sample_data['shift'],
                'shift_units': 'GHz',
                'width': sample_data['width'],
                'width_units': 'GHz'
            },
            {
                'shift': sample_data['shift'],
                'shift_units': 'GHz',
                'width': sample_data['width'],
                'width_units': 'GHz'
            }
        )
        
        f.close()
        
        # Read back and verify
        f = brim.File(filename, mode='r')
        data = f.get_data()
        
        # Verify spectrum
        coord = (0, 0, 0)
        PSD, frequency, _, _ = data.get_spectrum_in_image(coord)
        assert PSD is not None
        assert len(PSD) > 0
        
        # Verify metadata
        md = data.get_metadata()
        temp = md['Experiment.Temperature']
        assert temp.value == 22.0
        
        # Verify analysis results
        ar = data.get_analysis_results()
        Quantity = brim.Data.AnalysisResults.Quantity
        PeakType = brim.Data.AnalysisResults.PeakType
        img, _ = ar.get_image(Quantity.Shift, PeakType.average)
        assert img is not None
        
        f.close()
    
    def test_multiple_data_groups_workflow(self, tmp_path, sample_data):
        """Test workflow with multiple data groups."""
        filename = os.path.join(tmp_path, 'multi_data.brim.zarr')
        
        f = brim.File.create(filename, store_type=brim.StoreType.AUTO)
        
        # Create first data group
        data1 = f.create_data_group(
            sample_data['PSD'],
            sample_data['frequency'],
            sample_data['pixel_size'],
            name='data1'
        )
        
        # Create second data group
        data2 = f.create_data_group(
            sample_data['PSD'],
            sample_data['frequency'],
            sample_data['pixel_size'],
            name='data2'
        )
        
        f.close()
        
        # Read back
        f = brim.File(filename)
        groups = f.list_data_groups()
        assert len(groups) == 2
        
        # Access both groups
        d1 = f.get_data(0)
        d2 = f.get_data(1)
        assert d1 is not None
        assert d2 is not None
        
        f.close()
    
    def test_multiple_analysis_results_workflow(self, tmp_path, sample_data):
        """Test workflow with multiple analysis results in one data group."""
        filename = os.path.join(tmp_path, 'multi_ar.brim.zarr')
        
        f = brim.File.create(filename, store_type=brim.StoreType.AUTO)
        
        data = f.create_data_group(
            sample_data['PSD'],
            sample_data['frequency'],
            sample_data['pixel_size']
        )
        
        # Create first analysis results
        ar1 = data.create_analysis_results_group(
            {
                'shift': sample_data['shift'],
                'shift_units': 'GHz',
                'width': sample_data['width'],
                'width_units': 'GHz'
            },
            {
                'shift': sample_data['shift'],
                'shift_units': 'GHz',
                'width': sample_data['width'],
                'width_units': 'GHz'
            },
            name='analysis1'
        )
        
        # Create second analysis results
        ar2 = data.create_analysis_results_group(
            {
                'shift': sample_data['shift'],
                'shift_units': 'GHz',
                'width': sample_data['width'],
                'width_units': 'GHz'
            },
            {
                'shift': sample_data['shift'],
                'shift_units': 'GHz',
                'width': sample_data['width'],
                'width_units': 'GHz'
            },
            name='analysis2'
        )
        
        f.close()
        
        # Read back
        f = brim.File(filename)
        data = f.get_data()
        ar_list = data.list_AnalysisResults()
        assert len(ar_list) == 2
        
        f.close()


class TestDataConsistency:
    """Tests for data consistency across operations."""
    
    def test_spectrum_consistency(self, simple_brim_file):
        """Test that spectrum data is consistent across multiple reads."""
        f = brim.File(simple_brim_file)
        data = f.get_data()
        
        coord = (1, 2, 3)
        
        # Read spectrum twice
        PSD1, freq1, _, _ = data.get_spectrum_in_image(coord)
        PSD2, freq2, _, _ = data.get_spectrum_in_image(coord)
        
        # Should be identical
        np.testing.assert_array_equal(PSD1, PSD2)
        np.testing.assert_array_equal(freq1, freq2)
        
        f.close()
    
    def test_image_consistency(self, simple_brim_file):
        """Test that image data is consistent across multiple reads."""
        f = brim.File(simple_brim_file)
        data = f.get_data()
        ar = data.get_analysis_results()
        
        Quantity = brim.Data.AnalysisResults.Quantity
        PeakType = brim.Data.AnalysisResults.PeakType
        
        # Read image twice
        img1, px_size1 = ar.get_image(Quantity.Shift, PeakType.average)
        img2, px_size2 = ar.get_image(Quantity.Shift, PeakType.average)
        
        # Should be identical
        np.testing.assert_array_equal(img1, img2)
        f.close()
    
    def test_metadata_consistency(self, simple_brim_file):
        """Test that metadata is consistent across multiple reads."""
        f = brim.File(simple_brim_file)
        data = f.get_data()
        md = data.get_metadata()
        
        # Read metadata twice
        temp1 = md['Experiment.Temperature']
        temp2 = md['Experiment.Temperature']
        
        assert temp1.value == temp2.value
        assert temp1.units == temp2.units
        
        f.close()


class TestReadOnlyBehavior:
    """Tests for read-only file behavior."""
    
    def test_cannot_modify_in_read_mode(self, simple_brim_file, sample_data):
        """Test that modifications fail in read-only mode."""
        f = brim.File(simple_brim_file, mode='r')
        
        # Attempting to create data group should fail
        with pytest.raises(Exception):
            f.create_data_group(
                sample_data['PSD'],
                sample_data['frequency'],
                sample_data['pixel_size']
            )
        
        f.close()
    
    def test_can_modify_in_write_mode(self, simple_brim_file):
        """Test that modifications work in write mode."""
        f = brim.File(simple_brim_file, mode='r+')
        data = f.get_data()
        md = data.get_metadata()
        
        # Should be able to add metadata
        Attr = brim.Metadata.Item
        with pytest.warns(UserWarning, match="Unknown field 'NewValue'"):
            md.add(brim.Metadata.Type.Experiment, {'NewValue': Attr(100, 'units')}, local=True)
        
        f.close()


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    def test_empty_file_operations(self, empty_brim_file):
        """Test operations on an empty file."""
        f = brim.File(empty_brim_file)
        
        # List data groups should return empty
        groups = f.list_data_groups()
        assert len(groups) == 0
        
        f.close()
    
    def test_single_point_spectrum(self, tmp_path):
        """Test handling of single-point spectra."""
        filename = os.path.join(tmp_path, 'single_point.brim.zarr')
        
        # Create minimal data
        PSD = np.random.rand(1, 1, 1, 50)
        frequency = np.linspace(5, 10, 50)
        pixel_size = (1.0, 1.0, 1.0)
        
        f = brim.File.create(filename, store_type=brim.StoreType.AUTO)
        data = f.create_data_group(PSD, frequency, pixel_size)
        f.close()
        
        # Read back
        f = brim.File(filename)
        data = f.get_data()
        PSD_read, freq_read, _, _ = data.get_spectrum_in_image((0, 0, 0))
        
        assert len(PSD_read) == 50
        f.close()
    
    def test_large_frequency_array(self, tmp_path):
        """Test handling of large frequency arrays."""
        filename = os.path.join(tmp_path, 'large_freq.brim.zarr')
        
        # Create data with large frequency array
        PSD = np.random.rand(2, 2, 2, 1000)
        frequency = np.linspace(1, 20, 1000)
        pixel_size = (1.0, 1.0, 1.0)
        
        f = brim.File.create(filename, store_type=brim.StoreType.AUTO)
        data = f.create_data_group(PSD, frequency, pixel_size)
        f.close()
        
        # Read back
        f = brim.File(filename)
        data = f.get_data()
        PSD_read, freq_read, _, _ = data.get_spectrum_in_image((0, 0, 0))
        
        assert len(PSD_read) == 1000
        assert len(freq_read) == 1000
        f.close()


class TestFileLifecycle:
    """Tests for file lifecycle management."""
    
    def test_create_close_reopen(self, tmp_path, sample_data):
        """Test creating, closing, and reopening a file."""
        filename = os.path.join(tmp_path, 'lifecycle.brim.zarr')
        
        # Create
        f = brim.File.create(filename, store_type=brim.StoreType.AUTO)
        data = f.create_data_group(
            sample_data['PSD'],
            sample_data['frequency'],
            sample_data['pixel_size']
        )
        f.close()
        
        # Reopen in read mode
        f = brim.File(filename, mode='r')
        data = f.get_data()
        assert data is not None
        f.close()
        
        # Reopen in write mode
        f = brim.File(filename, mode='r+')
        data = f.get_data()
        md = data.get_metadata()
        Attr = brim.Metadata.Item
        with pytest.warns(UserWarning, match="Unknown field 'CustomLifecycleField'"):
            md.add(brim.Metadata.Type.Experiment, {'CustomLifecycleField': Attr(1, 'unit')}, local=True)
        f.close()
    
    def test_multiple_sequential_operations(self, tmp_path, sample_data):
        """Test multiple sequential file operations."""
        filename = os.path.join(tmp_path, 'sequential.brim.zarr')
        
        # Create and add first data group
        f = brim.File.create(filename, store_type=brim.StoreType.AUTO)
        data1 = f.create_data_group(
            sample_data['PSD'],
            sample_data['frequency'],
            sample_data['pixel_size'],
            name='data1'
        )
        f.close()
        
        # Reopen and add second data group
        f = brim.File(filename, mode='r+')
        data2 = f.create_data_group(
            sample_data['PSD'],
            sample_data['frequency'],
            sample_data['pixel_size'],
            name='data2'
        )
        f.close()
        
        # Verify both exist
        f = brim.File(filename)
        groups = f.list_data_groups()
        assert len(groups) == 2
        f.close()
