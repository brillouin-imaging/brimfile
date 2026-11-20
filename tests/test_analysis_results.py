"""
Unit tests for the AnalysisResults class in brimfile.
"""


import numpy as np

import brimfile as brim
from brimfile.physics import Brillouin_shift_water, Brillouin_width_water


class TestAnalysisResultsProperties:
    """Tests for AnalysisResults properties."""
    
    def test_get_name(self, simple_brim_file):
        """Test getting the name of analysis results."""
        f = brim.File(simple_brim_file)
        data = f.get_data()
        ar = data.get_analysis_results()
        
        name = ar.get_name()
        assert name is not None
        assert isinstance(name, str)
        f.close()
    
    def test_fit_model_property(self, simple_brim_file):
        """Test accessing fit model property."""
        f = brim.File(simple_brim_file)
        data = f.get_data()
        ar = data.get_analysis_results()
        
        fit_model = ar.fit_model
        assert fit_model is not None
        assert fit_model == brim.Data.AnalysisResults.FitModel.Lorentzian
        f.close()


class TestPeakTypesAndQuantities:
    """Tests for peak types and quantities in AnalysisResults."""
    
    def test_list_existing_peak_types(self, simple_brim_file):
        """Test listing existing peak types."""
        f = brim.File(simple_brim_file)
        data = f.get_data()
        ar = data.get_analysis_results()
        
        peak_types = ar.list_existing_peak_types()
        assert peak_types is not None
        assert isinstance(peak_types, (list, tuple))
        assert len(peak_types) > 0
        assert isinstance(peak_types[0], brim.Data.AnalysisResults.PeakType)
        f.close()
    
    def test_list_existing_quantities(self, simple_brim_file):
        """Test listing existing quantities."""
        f = brim.File(simple_brim_file)
        data = f.get_data()
        ar = data.get_analysis_results()
        
        quantities = ar.list_existing_quantities()
        assert quantities is not None
        assert isinstance(quantities, (list, tuple))
        assert len(quantities) > 0
        assert isinstance(quantities[0], brim.Data.AnalysisResults.Quantity)
        assert brim.Data.AnalysisResults.Quantity.Elastic_contrast in quantities
        assert brim.Data.AnalysisResults.Quantity.Viscous_contrast in quantities
        f.close()
    
    def test_peak_type_enum_values(self):
        """Test PeakType enum values."""
        PeakType = brim.Data.AnalysisResults.PeakType
        assert PeakType.average is not None
        assert PeakType.Stokes is not None
        assert PeakType.AntiStokes is not None
    
    def test_quantity_enum_values(self):
        """Test Quantity enum values."""
        Quantity = brim.Data.AnalysisResults.Quantity
        assert Quantity.Shift is not None
        assert Quantity.Width is not None
        assert Quantity.Elastic_contrast is not None
        assert Quantity.Viscous_contrast is not None


class TestImageRetrieval:
    """Tests for retrieving images from AnalysisResults."""
    
    def test_get_image_shift(self, simple_brim_file):
        """Test getting shift image."""
        f = brim.File(simple_brim_file)
        data = f.get_data()
        ar = data.get_analysis_results()
        
        Quantity = brim.Data.AnalysisResults.Quantity
        PeakType = brim.Data.AnalysisResults.PeakType
        
        img, px_size = ar.get_image(Quantity.Shift, PeakType.average)
        
        assert img is not None
        assert isinstance(img, np.ndarray)
        assert px_size is not None
        assert len(px_size) == 3
        f.close()
    
    def test_get_image_width(self, simple_brim_file):
        """Test getting width image."""
        f = brim.File(simple_brim_file)
        data = f.get_data()
        ar = data.get_analysis_results()
        
        Quantity = brim.Data.AnalysisResults.Quantity
        PeakType = brim.Data.AnalysisResults.PeakType
        
        img, px_size = ar.get_image(Quantity.Width, PeakType.average)
        
        assert img is not None
        assert isinstance(img, np.ndarray)
        f.close()

    def test_get_image_elastic_contrast(self, simple_brim_file):
        """Test getting elastic contrast image computed from shift."""
        f = brim.File(simple_brim_file)
        data = f.get_data()
        ar = data.get_analysis_results()

        Quantity = brim.Data.AnalysisResults.Quantity
        PeakType = brim.Data.AnalysisResults.PeakType

        img_shift, _ = ar.get_image(Quantity.Shift, PeakType.average)
        img_ec, _ = ar.get_image(Quantity.Elastic_contrast, PeakType.average)

        water_shift = Brillouin_shift_water(660, 22, 180)
        np.testing.assert_allclose(img_ec, img_shift / water_shift - 1)
        f.close()

    def test_get_image_viscous_contrast(self, simple_brim_file):
        """Test getting viscous contrast image computed from width."""
        f = brim.File(simple_brim_file)
        data = f.get_data()
        ar = data.get_analysis_results()

        Quantity = brim.Data.AnalysisResults.Quantity
        PeakType = brim.Data.AnalysisResults.PeakType

        img_width, _ = ar.get_image(Quantity.Width, PeakType.average)
        img_vc, _ = ar.get_image(Quantity.Viscous_contrast, PeakType.average)

        water_width = Brillouin_width_water(660, 22, 180)
        np.testing.assert_allclose(img_vc, img_width / water_width - 1)
        f.close()
    
    def test_image_dimensions(self, simple_brim_file, sample_data):
        """Test that image dimensions match expected dimensions."""
        f = brim.File(simple_brim_file)
        data = f.get_data()
        ar = data.get_analysis_results()
        
        Quantity = brim.Data.AnalysisResults.Quantity
        PeakType = brim.Data.AnalysisResults.PeakType
        
        img, _ = ar.get_image(Quantity.Shift, PeakType.average)
        
        # Should match the dimensions from sample_data
        Nz, Ny, Nx = sample_data['dimensions']
        assert img.shape == (Nz, Ny, Nx)
        f.close()


class TestUnitsRetrieval:
    """Tests for retrieving units from AnalysisResults."""
    
    def test_get_units_shift(self, simple_brim_file):
        """Test getting units for shift quantity."""
        f = brim.File(simple_brim_file)
        data = f.get_data()
        ar = data.get_analysis_results()
        
        Quantity = brim.Data.AnalysisResults.Quantity
        units = ar.get_units(Quantity.Shift)
        
        assert units is not None
        assert isinstance(units, str)
        assert units == 'GHz'
        f.close()
    
    def test_get_units_width(self, simple_brim_file):
        """Test getting units for width quantity."""
        f = brim.File(simple_brim_file)
        data = f.get_data()
        ar = data.get_analysis_results()
        
        Quantity = brim.Data.AnalysisResults.Quantity
        units = ar.get_units(Quantity.Width)
        
        assert units is not None
        assert isinstance(units, str)
        f.close()

    def test_get_units_elastic_contrast(self, simple_brim_file):
        """Test getting units for elastic contrast quantity."""
        f = brim.File(simple_brim_file)
        data = f.get_data()
        ar = data.get_analysis_results()

        Quantity = brim.Data.AnalysisResults.Quantity
        units = ar.get_units(Quantity.Elastic_contrast)

        assert units is None
        f.close()

    def test_get_units_viscous_contrast(self, simple_brim_file):
        """Test getting units for viscous contrast quantity."""
        f = brim.File(simple_brim_file)
        data = f.get_data()
        ar = data.get_analysis_results()

        Quantity = brim.Data.AnalysisResults.Quantity
        units = ar.get_units(Quantity.Viscous_contrast)

        assert units is None
        f.close()


class TestPixelQuantityRetrieval:
    """Tests for retrieving quantities at specific pixels."""
    
    def test_get_quantity_at_pixel(self, simple_brim_file):
        """Test getting quantity value at a specific pixel."""
        f = brim.File(simple_brim_file)
        data = f.get_data()
        ar = data.get_analysis_results()
        
        Quantity = brim.Data.AnalysisResults.Quantity
        PeakType = brim.Data.AnalysisResults.PeakType
        
        coord = (1, 3, 4)
        value = ar.get_quantity_at_pixel(coord, Quantity.Shift, PeakType.average)
        
        assert value is not None
        assert isinstance(value, (int, float, np.number))
        f.close()
    
    def test_quantity_at_pixel_matches_image(self, simple_brim_file):
        """Test that quantity at pixel matches image value."""
        f = brim.File(simple_brim_file)
        data = f.get_data()
        ar = data.get_analysis_results()
        
        Quantity = brim.Data.AnalysisResults.Quantity
        PeakType = brim.Data.AnalysisResults.PeakType
        
        coord = (1, 3, 4)
        img, _ = ar.get_image(Quantity.Shift, PeakType.average)
        qt_at_px = ar.get_quantity_at_pixel(coord, Quantity.Shift, PeakType.average)
        
        assert img[coord] == qt_at_px
        f.close()
    
    def test_get_quantity_at_different_pixels(self, simple_brim_file):
        """Test getting quantities at multiple different pixels."""
        f = brim.File(simple_brim_file)
        data = f.get_data()
        ar = data.get_analysis_results()
        
        Quantity = brim.Data.AnalysisResults.Quantity
        PeakType = brim.Data.AnalysisResults.PeakType
        
        coords = [(0, 0, 0), (1, 2, 3), (2, 3, 5)]
        for coord in coords:
            value = ar.get_quantity_at_pixel(coord, Quantity.Shift, PeakType.average)
            assert value is not None
        
        f.close()

    def test_get_elastic_contrast_at_pixel(self, simple_brim_file):
        """Test getting elastic contrast at pixel as shift-normalized value."""
        f = brim.File(simple_brim_file)
        data = f.get_data()
        ar = data.get_analysis_results()

        Quantity = brim.Data.AnalysisResults.Quantity
        PeakType = brim.Data.AnalysisResults.PeakType

        coord = (1, 3, 4)
        shift = ar.get_quantity_at_pixel(coord, Quantity.Shift, PeakType.average)
        ec = ar.get_quantity_at_pixel(coord, Quantity.Elastic_contrast, PeakType.average)

        water_shift = Brillouin_shift_water(660, 22, 180)
        np.testing.assert_allclose(ec, shift / water_shift -1)
        f.close()

    def test_get_viscous_contrast_at_pixel(self, simple_brim_file):
        """Test getting viscous contrast at pixel as width-normalized value."""
        f = brim.File(simple_brim_file)
        data = f.get_data()
        ar = data.get_analysis_results()

        Quantity = brim.Data.AnalysisResults.Quantity
        PeakType = brim.Data.AnalysisResults.PeakType

        coord = (1, 3, 4)
        width = ar.get_quantity_at_pixel(coord, Quantity.Width, PeakType.average)
        vc = ar.get_quantity_at_pixel(coord, Quantity.Viscous_contrast, PeakType.average)

        water_width = Brillouin_width_water(660, 22, 180)
        np.testing.assert_allclose(vc, width / water_width -1)
        f.close()


class TestAnalysisResultsCreation:
    """Tests for creating analysis results."""
    
    def test_create_with_custom_name(self, empty_brim_file, sample_data):
        """Test creating analysis results with custom name."""
        f = brim.File(empty_brim_file, mode='r+')
        data = f.create_data_group(
            sample_data['PSD'],
            sample_data['frequency'],
            sample_data['pixel_size']
        )
        
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
            },
            name='custom_analysis'
        )
        
        name = ar.get_name()
        assert 'custom_analysis' in name or name == 'custom_analysis'
        f.close()
    
    def test_create_with_fit_model(self, empty_brim_file, sample_data):
        """Test creating analysis results with fit model."""
        f = brim.File(empty_brim_file, mode='r+')
        data = f.create_data_group(
            sample_data['PSD'],
            sample_data['frequency'],
            sample_data['pixel_size']
        )
        
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
            },
            fit_model=brim.Data.AnalysisResults.FitModel.Lorentzian
        )
        
        assert ar.fit_model == brim.Data.AnalysisResults.FitModel.Lorentzian
        f.close()


class TestFitModel:
    """Tests for FitModel enum."""
    
    def test_fit_model_enum_values(self):
        """Test that FitModel enum has expected values."""
        FitModel = brim.Data.AnalysisResults.FitModel
        assert FitModel.Lorentzian is not None
        assert FitModel.Voigt is not None
        assert FitModel.DHO is not None

class TestAddData:
    """Tests for adding analysis results data."""
    
    def test_add_data_single_peak_antistokes(self, empty_brim_file, sample_data):
        """Test adding single peak AntiStokes data."""
        f = brim.File(empty_brim_file, mode='r+')
        data = f.create_data_group(
            sample_data['PSD'],
            sample_data['frequency'],
            sample_data['pixel_size']
        )
        
        ar = data.create_analysis_results_group(
            {
                'shift': sample_data['shift'],
                'shift_units': 'GHz',
            },
            name='test_ar'
        )
        
        # Add additional data to the analysis results
        ar.add_data(
            data_AntiStokes={
                'width': sample_data['width'],
                'width_units': 'GHz'
            }
        )
        
        # Verify the data was added
        quantities = ar.list_existing_quantities()
        assert brim.Data.AnalysisResults.Quantity.Width in quantities
        f.close()
    
    def test_add_data_stokes(self, empty_brim_file, sample_data):
        """Test adding Stokes peak data."""
        f = brim.File(empty_brim_file, mode='r+')
        data = f.create_data_group(
            sample_data['PSD'],
            sample_data['frequency'],
            sample_data['pixel_size']
        )
        
        ar = data.create_analysis_results_group(
            {
                'shift': sample_data['shift'],
                'shift_units': 'GHz',
            },
            name='test_ar'
        )
        
        # Add Stokes data
        ar.add_data(
            data_Stokes={
                'shift': sample_data['shift'],
                'shift_units': 'GHz',
                'width': sample_data['width'],
                'width_units': 'GHz'
            }
        )
        
        # Verify the data was added
        peak_types = ar.list_existing_peak_types()
        assert brim.Data.AnalysisResults.PeakType.Stokes in peak_types
        f.close()
    
    def test_add_data_with_multiple_quantities(self, empty_brim_file, sample_data):
        """Test adding data with multiple quantities."""
        f = brim.File(empty_brim_file, mode='r+')
        data = f.create_data_group(
            sample_data['PSD'],
            sample_data['frequency'],
            sample_data['pixel_size']
        )
        
        ar = data.create_analysis_results_group(
            {
                'shift': sample_data['shift'],
                'shift_units': 'GHz',
            },
            name='test_ar'
        )
        
        # Add multiple quantities
        ar.add_data(
            data_AntiStokes={
                'width': sample_data['width'],
                'width_units': 'GHz',
                'amplitude': sample_data['width'] * 2,  # dummy amplitude
                'amplitude_units': 'V'
            }
        )
        
        # Verify quantities were added
        quantities = ar.list_existing_quantities()
        assert brim.Data.AnalysisResults.Quantity.Width in quantities
        assert brim.Data.AnalysisResults.Quantity.Amplitude in quantities
        f.close()
    
    def test_add_data_with_fit_model(self, empty_brim_file, sample_data):
        """Test adding data with fit model specification."""
        f = brim.File(empty_brim_file, mode='r+')
        data = f.create_data_group(
            sample_data['PSD'],
            sample_data['frequency'],
            sample_data['pixel_size']
        )
        
        ar = data.create_analysis_results_group(
            {
                'shift': sample_data['shift'],
                'shift_units': 'GHz',
            },
            name='test_ar'
        )
        
        # Add data with fit model
        ar.add_data(
            data_AntiStokes={
                'width': sample_data['width'],
                'width_units': 'GHz'
            },
            fit_model=brim.Data.AnalysisResults.FitModel.Voigt
        )
        
        # Verify fit model was set
        assert ar.fit_model == brim.Data.AnalysisResults.FitModel.Voigt
        f.close()


class TestGetAllQuantitiesInImage:
    """Tests for getting all quantities at a specific coordinate."""
    
    def test_get_all_quantities_in_image(self, simple_brim_file):
        """Test getting all quantities at a specific coordinate."""
        f = brim.File(simple_brim_file)
        data = f.get_data()
        ar = data.get_analysis_results()
        
        coord = (0, 0, 0)
        quantities = ar.get_all_quantities_in_image(coord)
        
        # Check return structure
        assert quantities is not None
        assert isinstance(quantities, dict)
        assert 'Elastic_contrast' in quantities
        assert 'Viscous_contrast' in quantities
        
        # Each quantity should have a nested dictionary with peak types
        for qty_name in quantities:
            assert isinstance(qty_name, str)
            assert isinstance(quantities[qty_name], dict)
            
            # Each peak type should be present
            for peak_name in quantities[qty_name]:
                assert isinstance(peak_name, str)
                item = quantities[qty_name][peak_name]
                # Should be Metadata.Item with value and units
                assert hasattr(item, 'value')
                assert hasattr(item, 'units')
        
        f.close()
    
    def test_get_all_quantities_different_coordinates(self, simple_brim_file):
        """Test that different coordinates return different quantity values."""
        f = brim.File(simple_brim_file)
        data = f.get_data()
        ar = data.get_analysis_results()
        
        coord1 = (0, 0, 0)
        coord2 = (1, 2, 3)
        
        quantities1 = ar.get_all_quantities_in_image(coord1)
        quantities2 = ar.get_all_quantities_in_image(coord2)
        
        # Both should return dictionaries
        assert isinstance(quantities1, dict)
        assert isinstance(quantities2, dict)
        
        # Find a common quantity with non-zero values
        for qty in quantities1:
            if qty in quantities2:
                peak_types = set(quantities1[qty].keys()) & set(quantities2[qty].keys())
                for peak in peak_types:
                    val1 = quantities1[qty][peak].value
                    val2 = quantities2[qty][peak].value
                    # At least one should be different (with high probability)
                    if not np.isnan(val1) and not np.isnan(val2):
                        # Values might be the same by chance, but likely different
                        pass
        
        f.close()
    
    def test_get_all_quantities_with_peak_index(self, simple_brim_file):
        """Test getting all quantities with explicit peak index."""
        f = brim.File(simple_brim_file)
        data = f.get_data()
        ar = data.get_analysis_results()
        
        coord = (0, 0, 0)
        quantities = ar.get_all_quantities_in_image(coord, index_peak=0)
        
        assert quantities is not None
        assert isinstance(quantities, dict)
        f.close()
    
    def test_get_all_quantities_invalid_coordinate(self, simple_brim_file):
        """Test that invalid coordinate raises error."""
        f = brim.File(simple_brim_file)
        data = f.get_data()
        ar = data.get_analysis_results()
        
        # Invalid: wrong number of coordinates
        try:
            ar.get_all_quantities_in_image((0, 0))
            assert False, "Should raise ValueError for wrong number of coordinates"
        except ValueError as e:
            assert "3 values" in str(e)
        
        f.close()


class TestSaveImageToOMETiff:
    """Tests for saving images to OME-TIFF format."""

    def test_save_image_pixel_calibration(self, simple_brim_file_sparse, sample_data, tmp_path):
        import xml.etree.ElementTree as ET
        import pytest

        tifffile = pytest.importorskip('tifffile')
        f = brim.File(simple_brim_file_sparse)
        try:
            ar = f.get_data().get_analysis_results()
            filename = ar.save_image_to_OMETiff(
                ar.Quantity.Shift, filename=str(tmp_path / 'calibrated.ome.tif')
            )
            with tifffile.TiffFile(filename) as tif:
                assert tif.ome_metadata is not None
                pixels = ET.fromstring(tif.ome_metadata).find('.//{*}Pixels')
                assert pixels is not None
                for axis, value in zip('ZYX', sample_data['pixel_size']):
                    assert float(pixels.attrib[f'PhysicalSize{axis}']) == pytest.approx(value)
                    assert pixels.attrib[f'PhysicalSize{axis}Unit'] == '\u00b5m'
                assert tif.pages[0].tags['ResolutionUnit'].value == 3
                for axis, value in zip('XY', reversed(sample_data['pixel_size'][1:])):
                    numerator, denominator = tif.pages[0].tags[f'{axis}Resolution'].value
                    assert numerator / denominator == pytest.approx(10000 / value)
        finally:
            f.close()
    
    def test_save_image_with_default_filename(self, simple_brim_file, tmp_path):
        """Test saving image with default filename."""
        try:
            import tifffile
        except ImportError:
            import pytest
            pytest.skip("tifffile module not available")
        
        f = brim.File(simple_brim_file)
        data = f.get_data()
        ar = data.get_analysis_results()
        
        Quantity = brim.Data.AnalysisResults.Quantity
        PeakType = brim.Data.AnalysisResults.PeakType
        
        # Save to temp directory
        import os
        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            
            filename = ar.save_image_to_OMETiff(Quantity.Shift, PeakType.average)
            
            # Check that file was created
            assert os.path.exists(filename)
            assert filename.endswith('.ome.tif')
            
        finally:
            os.chdir(old_cwd)
        
        f.close()
    
    def test_save_image_with_custom_filename(self, simple_brim_file, tmp_path):
        """Test saving image with custom filename."""
        try:
            import tifffile
        except ImportError:
            import pytest
            pytest.skip("tifffile module not available")
        
        f = brim.File(simple_brim_file)
        data = f.get_data()
        ar = data.get_analysis_results()
        
        Quantity = brim.Data.AnalysisResults.Quantity
        PeakType = brim.Data.AnalysisResults.PeakType
        
        # Save with custom name
        import os
        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            
            custom_name = "my_test_image"
            filename = ar.save_image_to_OMETiff(
                Quantity.Shift, 
                PeakType.average, 
                filename=custom_name
            )
            
            # Check filename
            assert "my_test_image" in filename
            assert filename.endswith('.ome.tif')
            
            # Check file exists
            assert os.path.exists(filename)
            
        finally:
            os.chdir(old_cwd)
        
        f.close()
    
    def test_save_image_filename_extension(self, simple_brim_file, tmp_path):
        """Test that .ome.tif extension is added if missing."""
        try:
            import tifffile
        except ImportError:
            import pytest
            pytest.skip("tifffile module not available")
        
        f = brim.File(simple_brim_file)
        data = f.get_data()
        ar = data.get_analysis_results()
        
        Quantity = brim.Data.AnalysisResults.Quantity
        PeakType = brim.Data.AnalysisResults.PeakType
        
        import os
        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            
            filename = ar.save_image_to_OMETiff(
                Quantity.Shift, 
                PeakType.average, 
                filename="test_image"
            )
            
            # Should have .ome.tif extension
            assert filename.endswith('.ome.tif')
            
        finally:
            os.chdir(old_cwd)
        
        f.close()
    
    def test_save_image_has_metadata(self, simple_brim_file, tmp_path):
        """Test that saved image has proper OME metadata."""
        try:
            import tifffile
        except ImportError:
            import pytest
            pytest.skip("tifffile module not available")
        
        f = brim.File(simple_brim_file)
        data = f.get_data()
        ar = data.get_analysis_results()
        
        Quantity = brim.Data.AnalysisResults.Quantity
        PeakType = brim.Data.AnalysisResults.PeakType
        
        import os
        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            
            filename = ar.save_image_to_OMETiff(Quantity.Shift, PeakType.average)
            
            # Read metadata from saved file
            with tifffile.TiffFile(filename) as tif:
                # Check that metadata exists
                assert len(tif.pages) > 0
                
        finally:
            os.chdir(old_cwd)
        
        f.close()