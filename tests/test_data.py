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
        spectra, qts = data.get_spectrum_and_all_quantities_in_image(ar, [coord])
        spectrum, quantities = spectra[0], qts[0]
        
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
        spectra, qts = data.get_spectrum_and_all_quantities_in_image(ar, [coord])
        spectrum, quantities = spectra[0], qts[0]
        
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
        
        spectra, qts = data.get_spectrum_and_all_quantities_in_image(ar, [coord1, coord2])
        spectrum1, spectrum2 = spectra
        quantities1, quantities2 = qts
        
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
        
        spectra, qts = data.get_spectrum_and_all_quantities_in_image(ar, [coord1, coord2])
        spectrum1, spectrum2 = spectra
        quantities1, quantities2 = qts
        
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
        spectra, qts = data.get_spectrum_and_all_quantities_in_image(ar, [coord])
        spectrum, quantities = spectra[0], qts[0]
        
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
        spectra, qts = data.get_spectrum_and_all_quantities_in_image(ar, [coord], index_peak=0)
        spectrum, quantities = spectra[0], qts[0]
        
        assert spectrum is not None
        assert quantities is not None
        
        f.close()

    @pytest.mark.parametrize("fixture_name", ["simple_brim_file", "simple_brim_file_sparse"])
    def test_single_point_returns_single_result(self, fixture_name, request):
        """A single coordinate returns (spectrum, quantities) rather than lists."""
        f = brim.File(request.getfixturevalue(fixture_name))
        data = f.get_data()
        ar = data.get_analysis_results()
        spectrum, quantities = data.get_spectrum_and_all_quantities_in_image(ar, (0, 0, 0))
        assert isinstance(spectrum, tuple) and len(spectrum) == 4
        assert isinstance(quantities, dict)
        f.close()

    @pytest.mark.parametrize("fixture_name", ["simple_brim_file", "simple_brim_file_sparse"])
    def test_spectra_and_quantities_many_points_match_single_points(self, fixture_name, request):
        """Batched retrieval must give the same results as retrieving points one by one (in any order)."""
        f = brim.File(request.getfixturevalue(fixture_name))
        data = f.get_data()
        ar = data.get_analysis_results()
        
        coords = [(1, 2, 3), (0, 0, 0), (1, 2, 3), (0, 1, 0)]
        spectra, qts = data.get_spectrum_and_all_quantities_in_image(ar, coords)
        assert len(spectra) == len(qts) == len(coords)
        for c, spectrum, quantities in zip(coords, spectra, qts):
            s1, q1 = data.get_spectrum_and_all_quantities_in_image(ar, [c])
            for a, b in zip(spectrum[:2], s1[0][:2]):
                np.testing.assert_array_equal(a, b)
            assert spectrum[2:] == s1[0][2:]
            assert quantities.keys() == q1[0].keys()
            for qt_name, peaks in quantities.items():
                for pt_name, item in peaks.items():
                    np.testing.assert_allclose(item.value, q1[0][qt_name][pt_name].value)
                    assert item.units == q1[0][qt_name][pt_name].units
        
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

