"""
Unit tests for the Data class in brimfile.
"""


import numpy as np
import pytest

import brimfile as brim


class TestDataProperties:
    """Tests for Data class properties and basic methods."""
    
    def test_get_name(self, simple_brim_file):
        """Test getting the name of a data group."""
        f = brim.File(simple_brim_file)
        data = f.get_data()
        name = data.get_name()
        assert name is not None
        assert isinstance(name, str)
        f.close()
    
    def test_get_name_sparse(self, simple_brim_file_sparse):
        """Test getting the name of a sparse data group."""
        f = brim.File(simple_brim_file_sparse)
        data = f.get_data()
        name = data.get_name()
        assert name is not None
        assert isinstance(name, str)
        f.close()
    
    def test_get_index(self, simple_brim_file):
        """Test getting the index of a data group."""
        f = brim.File(simple_brim_file)
        data = f.get_data()
        index = data.get_index()
        assert isinstance(index, int)
        assert index >= 0
        f.close()
    
    def test_get_index_sparse(self, simple_brim_file_sparse):
        """Test getting the index of a sparse data group."""
        f = brim.File(simple_brim_file_sparse)
        data = f.get_data()
        index = data.get_index()
        assert isinstance(index, int)
        assert index >= 0
        f.close()
    
    def test_get_num_parameters(self, simple_brim_file):
        """Test getting the number of parameters."""
        f = brim.File(simple_brim_file)
        data = f.get_data()
        n_pars = data.get_num_parameters()
        assert isinstance(n_pars, tuple)
        f.close()
    
    def test_get_num_parameters_sparse(self, simple_brim_file_sparse):
        """Test getting the number of parameters for sparse data."""
        f = brim.File(simple_brim_file_sparse)
        data = f.get_data()
        n_pars = data.get_num_parameters()
        assert isinstance(n_pars, tuple)
        f.close()
    
    def test_sparse_flag(self, simple_brim_file, simple_brim_file_sparse):
        """Test that the sparse flag is properly set."""
        # Non-sparse data
        f = brim.File(simple_brim_file)
        data = f.get_data()
        assert data._sparse == False
        f.close()
        
        # Sparse data
        f = brim.File(simple_brim_file_sparse)
        data = f.get_data()
        assert data._sparse == True
        f.close()


class TestSpectrumRetrieval:
    """Tests for spectrum retrieval from Data."""
    
    def test_get_spectrum_in_image(self, simple_brim_file):
        """Test getting spectrum at a specific pixel."""
        f = brim.File(simple_brim_file)
        data = f.get_data()
        
        coord = (0, 0, 0)
        PSD, frequency, PSD_units, frequency_units = data.get_spectrum_in_image(coord)
        
        assert PSD is not None
        assert frequency is not None
        assert isinstance(PSD, np.ndarray)
        assert isinstance(frequency, np.ndarray)
        assert len(PSD) == len(frequency)
        f.close()
    
    def test_get_spectrum_in_image_sparse(self, simple_brim_file_sparse):
        """Test getting spectrum at a specific pixel for sparse data."""
        f = brim.File(simple_brim_file_sparse)
        data = f.get_data()
        
        coord = (0, 0, 0)
        PSD, frequency, PSD_units, frequency_units = data.get_spectrum_in_image(coord)
        
        assert PSD is not None
        assert frequency is not None
        assert isinstance(PSD, np.ndarray)
        assert isinstance(frequency, np.ndarray)
        assert len(PSD) == len(frequency)
        f.close()
    
    def test_get_spectrum_different_coordinates(self, simple_brim_file):
        """Test getting spectra at different coordinates."""
        f = brim.File(simple_brim_file)
        data = f.get_data()
        
        coords = [(0, 0, 0), (1, 2, 3), (2, 4, 6)]
        for coord in coords:
            PSD, frequency, _, _ = data.get_spectrum_in_image(coord)
            assert PSD is not None
            assert len(PSD) > 0
        
        f.close()
    
    def test_get_spectrum_different_coordinates_sparse(self, simple_brim_file_sparse):
        """Test getting spectra at different coordinates for sparse data."""
        f = brim.File(simple_brim_file_sparse)
        data = f.get_data()
        
        # For sparse data, we need to check valid coordinates
        coords = [(0, 0, 0), (0, 1, 0), (0, 0, 1)]
        for coord in coords:
            PSD, frequency, _, _ = data.get_spectrum_in_image(coord)
            assert PSD is not None
            assert len(PSD) > 0
        
        f.close()
    
    def test_spectrum_units(self, simple_brim_file):
        """Test that spectrum units are returned."""
        f = brim.File(simple_brim_file)
        data = f.get_data()
        
        coord = (0, 0, 0)
        _, _, PSD_units, frequency_units = data.get_spectrum_in_image(coord)
        
        # Units should be strings or None
        assert PSD_units is None or isinstance(PSD_units, str)
        assert frequency_units is None or isinstance(frequency_units, str)
        f.close()
    
    def test_spectrum_units_sparse(self, simple_brim_file_sparse):
        """Test that spectrum units are returned for sparse data."""
        f = brim.File(simple_brim_file_sparse)
        data = f.get_data()
        
        coord = (0, 0, 0)
        _, _, PSD_units, frequency_units = data.get_spectrum_in_image(coord)
        
        # Units should be strings or None
        assert PSD_units is None or isinstance(PSD_units, str)
        assert frequency_units is None or isinstance(frequency_units, str)
        f.close()


class TestPSDRetrieval:
    """Tests for PSD spatial map retrieval."""
    
    def test_get_PSD_as_spatial_map(self, simple_brim_file):
        """Test getting PSD as spatial map for non-sparse data."""
        f = brim.File(simple_brim_file)
        data = f.get_data()
        
        PSD, frequency, PSD_units, frequency_units = data.get_PSD_as_spatial_map()
        
        assert PSD is not None
        assert frequency is not None
        assert isinstance(PSD, np.ndarray)
        assert isinstance(frequency, np.ndarray)
        
        # PSD should be at least 4D: (z, y, x, spectrum)
        assert PSD.ndim >= 4
        
        # frequency should be broadcastable to PSD shape
        assert frequency.ndim == 1 or frequency.shape == PSD.shape
        f.close()
    
    def test_get_PSD_as_spatial_map_sparse(self, simple_brim_file_sparse):
        """Test getting PSD as spatial map for sparse data."""
        f = brim.File(simple_brim_file_sparse)
        data = f.get_data()
        
        PSD, frequency, PSD_units, frequency_units = data.get_PSD_as_spatial_map()
        
        assert PSD is not None
        assert frequency is not None
        assert isinstance(PSD, np.ndarray)
        assert isinstance(frequency, np.ndarray)
        
        # For sparse data, PSD should be reshaped to spatial dimensions
        # PSD should be at least 4D: (z, y, x, spectrum)
        assert PSD.ndim >= 4
        
        # frequency should be broadcastable to PSD shape
        assert frequency.ndim == 1 or frequency.shape == PSD.shape
        f.close()
    
    def test_PSD_spatial_map_shapes_match(self, simple_brim_file, simple_brim_file_sparse):
        """Test that sparse and non-sparse data produce compatible spatial maps."""
        # Non-sparse data
        f_ns = brim.File(simple_brim_file)
        data_ns = f_ns.get_data()
        PSD_ns, freq_ns, _, _ = data_ns.get_PSD_as_spatial_map()
        
        # Sparse data
        f_s = brim.File(simple_brim_file_sparse)
        data_s = f_s.get_data()
        PSD_s, freq_s, _, _ = data_s.get_PSD_as_spatial_map()
        
        # Both should have the same spatial dimensions (first 3 dimensions)
        assert PSD_ns.shape[:3] == PSD_s.shape[:3]
        
        # Both should have the same frequency dimension (last dimension)
        assert PSD_ns.shape[-1] == PSD_s.shape[-1]
        
        f_ns.close()
        f_s.close()


class TestMetadataAccess:
    """Tests for metadata access from Data."""
    
    def test_get_metadata(self, simple_brim_file):
        """Test getting metadata from data group."""
        f = brim.File(simple_brim_file)
        data = f.get_data()
        metadata = data.get_metadata()
        
        assert metadata is not None
        assert isinstance(metadata, brim.Metadata)
        f.close()


class TestAnalysisResults:
    """Tests for analysis results in Data."""
    
    def test_list_analysis_results(self, simple_brim_file):
        """Test listing analysis results."""
        f = brim.File(simple_brim_file)
        data = f.get_data()
        ar_list = data.list_AnalysisResults()
        
        assert ar_list is not None
        assert len(ar_list) > 0
        f.close()
    
    def test_list_analysis_results_sparse(self, simple_brim_file_sparse):
        """Test listing analysis results for sparse data."""
        f = brim.File(simple_brim_file_sparse)
        data = f.get_data()
        ar_list = data.list_AnalysisResults()
        
        assert ar_list is not None
        assert len(ar_list) > 0
        f.close()
    
    def test_list_analysis_results_with_names(self, simple_brim_file):
        """Test listing analysis results with custom names."""
        f = brim.File(simple_brim_file)
        data = f.get_data()
        ar_results = data.list_AnalysisResults(retrieve_custom_name=True)
        
        assert ar_results is not None
        assert isinstance(ar_results, list)
        f.close()
    
    def test_list_analysis_results_with_names_sparse(self, simple_brim_file_sparse):
        """Test listing analysis results with custom names for sparse data."""
        f = brim.File(simple_brim_file_sparse)
        data = f.get_data()
        ar_results = data.list_AnalysisResults(retrieve_custom_name=True)
        
        assert ar_results is not None
        assert isinstance(ar_results, list)
        f.close()
    
    def test_get_analysis_results(self, simple_brim_file):
        """Test getting analysis results."""
        f = brim.File(simple_brim_file)
        data = f.get_data()
        ar = data.get_analysis_results()
        
        assert ar is not None
        assert isinstance(ar, brim.Data.AnalysisResults)
        f.close()
    
    def test_get_analysis_results_sparse(self, simple_brim_file_sparse):
        """Test getting analysis results for sparse data."""
        f = brim.File(simple_brim_file_sparse)
        data = f.get_data()
        ar = data.get_analysis_results()
        
        assert ar is not None
        assert isinstance(ar, brim.Data.AnalysisResults)
        f.close()
    
    def test_get_analysis_results_by_index(self, simple_brim_file):
        """Test getting analysis results by index."""
        f = brim.File(simple_brim_file)
        data = f.get_data()
        ar = data.get_analysis_results(0)
        
        assert ar is not None
        f.close()
    
    def test_get_analysis_results_by_index_sparse(self, simple_brim_file_sparse):
        """Test getting analysis results by index for sparse data."""
        f = brim.File(simple_brim_file_sparse)
        data = f.get_data()
        ar = data.get_analysis_results(0)
        
        assert ar is not None
        f.close()
    
    def test_create_analysis_results(self, empty_brim_file, sample_data):
        """Test creating new analysis results."""
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
            name='test_ar'
        )
        
        assert ar is not None
        f.close()
    
    def test_create_analysis_results_sparse(self, empty_brim_file, sample_data_sparse):
        """Test creating new analysis results for sparse data."""
        f = brim.File(empty_brim_file, mode='r+')
        data = f.create_data_group_sparse(
            sample_data_sparse['PSD'],
            sample_data_sparse['frequency'],
            scanning=sample_data_sparse['scanning']
        )
        
        ar = data.create_analysis_results_group(
            {
                'shift': sample_data_sparse['shift'],
                'shift_units': 'GHz',
                'width': sample_data_sparse['width'],
                'width_units': 'GHz'
            },
            {
                'shift': sample_data_sparse['shift'],
                'shift_units': 'GHz',
                'width': sample_data_sparse['width'],
                'width_units': 'GHz'
            },
            name='test_ar_sparse'
        )
        
        assert ar is not None
        f.close()
    
    def test_get_spectrum_and_all_quantities_in_image(self, simple_brim_file):
        """Test getting spectrum and all quantities at a specific coordinate for non-sparse data."""
        f = brim.File(simple_brim_file)
        data = f.get_data()
        ar = data.get_analysis_results()
        
        coord = (0, 0, 0)
        spectrum, quantities = data.get_spectrum_and_all_quantities_in_image(ar, coord)
        
        # Check spectrum tuple structure
        assert spectrum is not None
        assert isinstance(spectrum, tuple)
        assert len(spectrum) == 4
        PSD, frequency, PSD_units, frequency_units = spectrum
        
        # Validate spectrum components
        assert PSD is not None
        assert frequency is not None
        assert isinstance(PSD, np.ndarray)
        assert isinstance(frequency, np.ndarray)
        assert len(PSD) == len(frequency)
        
        # Check units
        assert PSD_units is None or isinstance(PSD_units, str)
        assert frequency_units is None or isinstance(frequency_units, str)
        
        # Check quantities dictionary
        assert quantities is not None
        assert isinstance(quantities, dict)
        assert 'Elastic_contrast' in quantities
        assert 'Viscous_contrast' in quantities
        
        # Quantities should contain the expected peak types as keys
        for peak_name in quantities:
            assert isinstance(peak_name, str)
            # Each peak should have a dictionary of quantities
            assert isinstance(quantities[peak_name], dict)
            for quantity_name in quantities[peak_name]:
                assert isinstance(quantity_name, str)
        
        f.close()
    
    def test_get_spectrum_and_all_quantities_in_image_sparse(self, simple_brim_file_sparse):
        """Test getting spectrum and all quantities at a specific coordinate for sparse data."""
        f = brim.File(simple_brim_file_sparse)
        data = f.get_data()
        ar = data.get_analysis_results()
        
        coord = (0, 0, 0)
        spectrum, quantities = data.get_spectrum_and_all_quantities_in_image(ar, coord)
        
        # Check spectrum tuple structure
        assert spectrum is not None
        assert isinstance(spectrum, tuple)
        assert len(spectrum) == 4
        PSD, frequency, PSD_units, frequency_units = spectrum
        
        # Validate spectrum components
        assert PSD is not None
        assert frequency is not None
        assert isinstance(PSD, np.ndarray)
        assert isinstance(frequency, np.ndarray)
        assert len(PSD) == len(frequency)
        
        # Check units
        assert PSD_units is None or isinstance(PSD_units, str)
        assert frequency_units is None or isinstance(frequency_units, str)
        
        # Check quantities dictionary
        assert quantities is not None
        assert isinstance(quantities, dict)
        assert 'Elastic_contrast' in quantities
        assert 'Viscous_contrast' in quantities
        
        # Quantities should contain the expected peak types as keys
        for peak_name in quantities:
            assert isinstance(peak_name, str)
            # Each peak should have a dictionary of quantities
            assert isinstance(quantities[peak_name], dict)
            for quantity_name in quantities[peak_name]:
                assert isinstance(quantity_name, str)
        
        f.close()
    
    def test_spectrum_and_quantities_with_different_coordinates(self, simple_brim_file):
        """Test that different coordinates return different spectra and quantities for non-sparse data."""
        f = brim.File(simple_brim_file)
        data = f.get_data()
        ar = data.get_analysis_results()
        
        coord1 = (0, 0, 0)
        coord2 = (1, 2, 3)
        
        spectrum1, quantities1 = data.get_spectrum_and_all_quantities_in_image(ar, coord1)
        spectrum2, quantities2 = data.get_spectrum_and_all_quantities_in_image(ar, coord2)
        
        # Spectra should be different at different coordinates
        PSD1, freq1, _, _ = spectrum1
        PSD2, freq2, _, _ = spectrum2
        
        # Frequencies should be the same
        np.testing.assert_array_equal(freq1, freq2)
        
        # PSDs should be different (with high probability for generated test data)
        assert not np.allclose(PSD1, PSD2)
        
        f.close()
    
    def test_spectrum_and_quantities_with_different_coordinates_sparse(self, simple_brim_file_sparse):
        """Test that different coordinates return different spectra and quantities for sparse data."""
        f = brim.File(simple_brim_file_sparse)
        data = f.get_data()
        ar = data.get_analysis_results()
        
        coord1 = (0, 0, 0)
        coord2 = (0, 1, 0)
        
        spectrum1, quantities1 = data.get_spectrum_and_all_quantities_in_image(ar, coord1)
        spectrum2, quantities2 = data.get_spectrum_and_all_quantities_in_image(ar, coord2)
        
        # Spectra should be different at different coordinates
        PSD1, freq1, _, _ = spectrum1
        PSD2, freq2, _, _ = spectrum2
        
        # Frequencies should be the same
        np.testing.assert_array_equal(freq1, freq2)
        
        # PSDs should be different (with high probability for generated test data)
        assert not np.allclose(PSD1, PSD2)
        
        f.close()
    
    def test_spectrum_and_quantities_default_peak_index(self, simple_brim_file):
        """Test getting spectrum and quantities with default peak index (0)."""
        f = brim.File(simple_brim_file)
        data = f.get_data()
        ar = data.get_analysis_results()
        
        coord = (0, 0, 0)
        spectrum, quantities = data.get_spectrum_and_all_quantities_in_image(ar, coord)
        
        # Should work without specifying index_peak (defaults to 0)
        assert spectrum is not None
        assert quantities is not None
        
        f.close()
    
    def test_spectrum_and_quantities_explicit_peak_index(self, simple_brim_file):
        """Test getting spectrum and quantities with explicit peak index."""
        f = brim.File(simple_brim_file)
        data = f.get_data()
        ar = data.get_analysis_results()
        
        coord = (0, 0, 0)
        # Explicitly pass index_peak=0
        spectrum, quantities = data.get_spectrum_and_all_quantities_in_image(ar, coord, index_peak=0)
        
        assert spectrum is not None
        assert quantities is not None
        
        f.close()


class TestDataCreation:
    """Tests for creating data groups with different parameters."""
    
    def test_create_with_custom_name(self, empty_brim_file, sample_data):
        """Test creating data group with custom name."""
        f = brim.File(empty_brim_file, mode='r+')
        data = f.create_data_group(
            sample_data['PSD'],
            sample_data['frequency'],
            sample_data['pixel_size'],
            name='custom_name'
        )
        
        name = data.get_name()
        assert 'custom_name' in name or name == 'custom_name'
        f.close()
    
    def test_create_sparse_with_custom_name(self, empty_brim_file, sample_data_sparse):
        """Test creating sparse data group with custom name."""
        f = brim.File(empty_brim_file, mode='r+')
        data = f.create_data_group_sparse(
            sample_data_sparse['PSD'],
            sample_data_sparse['frequency'],
            scanning=sample_data_sparse['scanning'],
            name='custom_sparse_name'
        )
        
        name = data.get_name()
        assert 'custom_sparse_name' in name or name == 'custom_sparse_name'
        f.close()
    
    def test_create_with_different_dimensions(self, empty_brim_file):
        """Test creating data with different dimensions."""
        f = brim.File(empty_brim_file, mode='r+')
        
        # Smaller dataset
        PSD = np.random.rand(2, 3, 4, 50)
        frequency = np.linspace(5, 10, 50)
        pixel_size = (1.0, 1.0, 1.0)
        
        data = f.create_data_group(PSD, frequency, pixel_size)
        assert data is not None
        f.close()
    
    def test_create_sparse_with_different_dimensions(self, empty_brim_file):
        """Test creating sparse data with different dimensions."""
        f = brim.File(empty_brim_file, mode='r+')
        
        # Smaller dataset - flattened for sparse
        n_spectra = 24  # 2 * 3 * 4
        n_freq = 50
        PSD = np.random.rand(n_spectra, n_freq)
        frequency = np.linspace(5, 10, n_freq)
        
        # Create spatial mapping
        indices = np.arange(n_spectra)
        cartesian_vis = np.reshape(indices, (2, 3, 4))
        scanning = {
            'Cartesian_visualisation': cartesian_vis,
            'Cartesian_visualisation_pixel': (1.0, 1.0, 1.0),
            'Cartesian_visualisation_pixel_unit': 'um'
        }
        
        data = f.create_data_group_sparse(PSD, frequency, scanning=scanning)
        assert data is not None
        f.close()


class TestSparseDataConsistency:
    """Tests to ensure sparse and non-sparse data behave consistently."""
    
    def test_spectrum_retrieval_consistency(self, simple_brim_file, simple_brim_file_sparse, sample_data, sample_data_sparse):
        """Test that the same coordinate yields consistent results for sparse and non-sparse."""
        # Get spectrum from non-sparse data
        f_ns = brim.File(simple_brim_file)
        data_ns = f_ns.get_data()
        PSD_ns, freq_ns, _, _ = data_ns.get_spectrum_in_image((0, 0, 0))
        
        # Get spectrum from sparse data
        f_s = brim.File(simple_brim_file_sparse)
        data_s = f_s.get_data()
        PSD_s, freq_s, _, _ = data_s.get_spectrum_in_image((0, 0, 0))
        
        # Should have same length
        assert len(PSD_ns) == len(PSD_s)
        assert len(freq_ns) == len(freq_s)
        
        # Frequency should be the same
        np.testing.assert_array_almost_equal(freq_ns, freq_s)
        
        # PSD should be very similar (allowing for rounding errors)
        np.testing.assert_array_almost_equal(PSD_ns, PSD_s, decimal=5)
        
        f_ns.close()
        f_s.close()
    
    def test_spatial_map_dimensions(self, simple_brim_file_sparse, sample_data_sparse):
        """Test that spatial map has correct dimensions for sparse data."""
        f = brim.File(simple_brim_file_sparse)
        data = f.get_data()
        
        # Get the spatial map dimensions from the Cartesian visualisation
        expected_shape = sample_data_sparse['scanning']['Cartesian_visualisation'].shape
        
        # Get PSD as spatial map
        PSD, _, _, _ = data.get_PSD_as_spatial_map()
        
        # First 3 dimensions should match the expected spatial dimensions
        assert PSD.shape[:3] == expected_shape
        
        f.close()
    
    def test_pixel_size_consistency(self, simple_brim_file, simple_brim_file_sparse):
        """Test that pixel size is properly retrieved for both sparse and non-sparse."""
        # Non-sparse data
        f_ns = brim.File(simple_brim_file)
        data_ns = f_ns.get_data()
        px_size_ns = data_ns._spatial_map_px_size
        assert px_size_ns is not None
        assert len(px_size_ns) == 3
        
        # Sparse data
        f_s = brim.File(simple_brim_file_sparse)
        data_s = f_s.get_data()
        px_size_s = data_s._spatial_map_px_size
        assert px_size_s is not None
        assert len(px_size_s) == 3
        
        # Both should have the same pixel sizes
        for i in range(3):
            assert px_size_ns[i].value == px_size_s[i].value
        
        f_ns.close()
        f_s.close()
    
    def test_get_parameters_no_parameters(self, simple_brim_file):
        """Test getting parameters when none exist."""
        f = brim.File(simple_brim_file)
        data = f.get_data()
        
        pars, pars_names = data.get_parameters()
        
        # Parameters may or may not exist in test data
        assert pars is None or isinstance(pars, np.ndarray)
        assert pars_names is None or isinstance(pars_names, (list, np.ndarray))
        f.close()
    
    def test_get_parameters_sparse(self, simple_brim_file_sparse):
        """Test getting parameters for sparse data when none exist."""
        f = brim.File(simple_brim_file_sparse)
        data = f.get_data()
        
        pars, pars_names = data.get_parameters()
        
        # Parameters may or may not exist in test data
        assert pars is None or isinstance(pars, np.ndarray)
        assert pars_names is None or isinstance(pars_names, (list, np.ndarray))
        f.close()



class TestInterpolateSpectra:
    """Tests for Data.interpolate_spectra (pure numpy, no file needed)."""

    interp = staticmethod(brim.Data.interpolate_spectra)

    @staticmethod
    def _linear_case(shape=(3, 4, 10)):
        # PSD = 2 * frequency, so linear interpolation is exact inside the range
        freq = np.linspace(0, 9, shape[-1])
        PSD = np.broadcast_to(2 * freq, shape).copy()
        return PSD, freq

    # ---- valid inputs ----
    def test_1d_psd_no_new_frequency_returns_inputs(self):
        PSD, freq = self._linear_case((10,))
        out_psd, out_freq = self.interp(PSD, freq)
        np.testing.assert_array_equal(out_psd, PSD)
        np.testing.assert_array_equal(out_freq, freq)

    def test_1d_frequency_no_new_frequency_returns_inputs_unchanged(self):
        PSD, freq = self._linear_case((3, 10))
        out_psd, out_freq = self.interp(PSD, freq)
        assert out_psd is PSD
        assert out_freq is freq

    def test_new_frequency_values_1d_shared(self):
        PSD, freq = self._linear_case((2, 3, 10))
        new_f = np.linspace(1, 8, 10)
        out_psd, out_freq = self.interp(PSD, freq, new_frequency=new_f)
        np.testing.assert_allclose(out_psd, np.broadcast_to(2 * new_f, PSD.shape))
        assert out_freq.shape == new_f.shape

    def test_per_spectrum_frequency_without_new_frequency(self):
        # the generated frequency is the mean of the per-spectrum frequencies
        base = np.linspace(0, 9, 10)
        freq = np.stack([base, base + 1, base + 2])  # (3, 10)
        PSD = 2 * freq
        out_psd, out_freq = self.interp(PSD, freq)
        np.testing.assert_allclose(out_freq, base + 1)
        assert out_psd.shape == PSD.shape
        for i in range(3):
            expected = np.interp(out_freq, freq[i], PSD[i])
            np.testing.assert_allclose(out_psd[i], expected)

    def test_generated_frequency_is_mean_over_all_leading_dims(self):
        base = np.linspace(0, 9, 10)
        offsets = np.arange(6, dtype=float).reshape(2, 3, 1)
        freq = base + offsets  # (2, 3, 10)
        out_psd, out_freq = self.interp(2 * freq, freq)
        assert out_freq.shape == (10,)
        np.testing.assert_allclose(out_freq, base + offsets.mean())
        assert out_psd.shape == freq.shape

    def test_generated_frequency_uses_sorted_frequency(self):
        base = np.linspace(0, 9, 10)
        freq = np.stack([base[::-1], base + 2])
        _, out_freq = self.interp(2 * freq, freq)
        np.testing.assert_allclose(out_freq, base + 1)

    def test_generated_frequency_from_broadcast_frequency(self):
        # a (1, n) frequency broadcast to a larger PSD averages to itself
        base = np.linspace(0, 9, 10)
        PSD = np.ones((4, 10))
        out_psd, out_freq = self.interp(PSD, base[None, :])
        np.testing.assert_allclose(out_freq, base)
        np.testing.assert_allclose(out_psd, PSD)

    def test_per_spectrum_new_frequency(self):
        base = np.linspace(0, 9, 10)
        freq = np.stack([base, base + 1])  # (2, 10)
        PSD = 2 * freq
        new_f = np.stack([np.linspace(2, 7, 10), np.linspace(3, 8, 10)])
        out_psd, out_freq = self.interp(PSD, freq, new_frequency=new_f)
        assert out_freq.shape == new_f.shape
        np.testing.assert_allclose(out_psd, 2 * new_f)

    def test_new_frequency_fewer_dims_broadcast(self):
        PSD, freq = self._linear_case((2, 3, 10))
        new_f = np.linspace(1, 8, 10)[None, :]  # (1, 10)
        out_psd, out_freq = self.interp(PSD, freq, new_frequency=new_f)
        assert out_freq.shape == (1, 10)
        np.testing.assert_allclose(out_psd, np.broadcast_to(2 * new_f, PSD.shape))

    def test_out_of_range_uses_endpoint_values(self):
        PSD, freq = self._linear_case((10,))
        new_f = np.array([-5.0, 100.0] + [4.0] * 8)
        out_psd, _ = self.interp(PSD, freq, new_frequency=new_f)
        assert out_psd[0] == PSD[0]
        assert out_psd[1] == PSD[-1]

    # ---- frequency validation ----
    def test_frequency_with_duplicates_is_accepted(self):
        PSD = np.ones((2, 5))
        freq = np.array([0, 1, 1, 2, 3.0])
        out_psd, _ = self.interp(PSD, freq, new_frequency=np.arange(5.0))
        np.testing.assert_allclose(out_psd, 1.0)

    def test_decreasing_frequency_is_sorted(self):
        freq = np.arange(10.0)
        PSD = np.broadcast_to(2 * freq, (3, 10))
        new_f = np.linspace(1, 8, 10)
        out_psd, _ = self.interp(PSD[:, ::-1], freq[::-1], new_frequency=new_f)
        np.testing.assert_allclose(out_psd, np.broadcast_to(2 * new_f, (3, 10)))

    def test_unsorted_frequency_is_sorted_with_psd(self):
        rng = np.random.default_rng(0)
        freq = np.linspace(0, 9, 10)
        perm = rng.permutation(10)
        new_f = np.linspace(1, 8, 10)
        out_psd, _ = self.interp(2 * freq[perm][None, :].repeat(2, 0), freq[perm], new_frequency=new_f)
        np.testing.assert_allclose(out_psd, np.broadcast_to(2 * new_f, (2, 10)))

    def test_per_spectrum_unsorted_frequency_without_new_frequency(self):
        base = np.linspace(0, 9, 10)
        freq = np.stack([base[::-1], base])  # first row decreasing
        PSD = 2 * freq
        out_psd, out_freq = self.interp(PSD, freq)
        np.testing.assert_allclose(out_freq, base)
        np.testing.assert_allclose(out_psd, np.broadcast_to(2 * base, (2, 10)))

    def test_sorting_does_not_modify_inputs(self):
        freq = np.arange(10.0)[::-1].copy()
        PSD = np.broadcast_to(2 * freq, (2, 10)).copy()
        freq_c, PSD_c = freq.copy(), PSD.copy()
        self.interp(PSD, freq, new_frequency=np.linspace(1, 8, 10))
        np.testing.assert_array_equal(freq, freq_c)
        np.testing.assert_array_equal(PSD, PSD_c)

    def test_frequency_non_finite(self):
        PSD = np.ones((2, 5))
        freq = np.array([0, 1, 2, 3, np.inf])
        with pytest.raises(ValueError):
            self.interp(PSD, freq, new_frequency=np.arange(5.0))
        freq = np.array([0, 1, 2, np.nan, 4.0])
        with pytest.raises(ValueError):
            self.interp(PSD, freq, new_frequency=np.arange(5.0))

    # ---- shape mismatches: frequency vs PSD ----
    @pytest.mark.parametrize("psd_shape, freq_shape", [
        ((3, 4), (5,)),         # last axis differs
        ((3, 4), (2, 4)),       # leading axis differs
        ((3, 4), (3, 5)),       # same ndim, last axis differs
        ((4,), (3, 4)),         # frequency has more dims than PSD
        ((2, 3, 4), (2, 4, 4)), # middle axis differs
        ((2, 3, 4), (3,)),      # 1D mismatch in 3D PSD
    ])
    def test_frequency_not_broadcastable_to_psd(self, psd_shape, freq_shape):
        PSD = np.ones(psd_shape)
        freq = np.broadcast_to(np.arange(freq_shape[-1], dtype=float), freq_shape)
        with pytest.raises(ValueError, match="broadcastable"):
            self.interp(PSD, freq, new_frequency=np.linspace(0, 1, psd_shape[-1]))

    @pytest.mark.parametrize("psd_shape, freq_shape", [
        ((3, 4), (4,)),
        ((3, 4), (3, 4)),
        ((2, 3, 4), (3, 4)),
        ((2, 3, 4), (2, 3, 4)),
    ])
    def test_frequency_broadcastable_shapes_accepted(self, psd_shape, freq_shape):
        PSD = np.ones(psd_shape)
        freq = np.broadcast_to(np.arange(freq_shape[-1], dtype=float), freq_shape)
        out_psd, _ = self.interp(PSD, freq, new_frequency=np.linspace(0, 3, 4))
        assert out_psd.shape == psd_shape

    # ---- shape mismatches: new_frequency vs PSD ----
    @pytest.mark.parametrize("psd_shape, new_shape", [
        ((3, 4), (5,)),          # last axis differs
        ((3, 4), (2, 4)),        # leading axis differs
        ((3, 4), (3, 5)),        # same ndim, last axis differs
        ((4,), (3, 4)),          # more dims than PSD
        ((3, 4), (2, 3, 4)),     # more dims than PSD
        ((2, 3, 4), (2, 4)),     # middle axis differs
        ((2, 3, 4), (5,)),
        ((1, 4), (3, 4)),        # broadcasts to a larger shape than PSD
    ])
    def test_new_frequency_incompatible_shape(self, psd_shape, new_shape):
        PSD = np.ones(psd_shape)
        freq = np.arange(psd_shape[-1], dtype=float)
        new_f = np.zeros(new_shape)
        with pytest.raises(ValueError):
            self.interp(PSD, freq, new_frequency=new_f)

    def test_new_frequency_non_finite(self):
        PSD = np.ones((2, 4))
        freq = np.arange(4.0)
        for bad in (np.nan, np.inf):
            new_f = np.array([0, 1, bad, 3.0])
            with pytest.raises(ValueError, match="finite"):
                self.interp(PSD, freq, new_frequency=new_f)
