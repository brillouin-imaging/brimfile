"""
Unit tests for calibration group creation and retrieval.
"""

import numpy as np
import pytest

import brimfile as brim


class TestCalibration:
    """Tests for Data.create_calibration_group and Data.get_calibration."""

    def test_create_and_get_calibration_non_sparse(self, empty_brim_file, sample_data):
        """Test non-sparse calibration retrieval with 3D index mapping."""
        f = brim.File(empty_brim_file, mode='r+')
        data = f.create_data_group(
            sample_data['PSD'],
            sample_data['frequency'],
            sample_data['pixel_size'],
        )

        index = np.zeros(sample_data['dimensions'], dtype=np.int32)
        index[1, 2, 3] = 1
        spectra = np.stack(
            [sample_data['PSD'][0, 0, 0, :], sample_data['PSD'][1, 2, 3, :]],
            axis=0,
        )

        data.create_calibration_group(
            index=index,
            calibration_data=[
                {
                    'spectra': spectra,
                    'shift': 7.0,
                    'shift_units': 'GHz',
                }
            ],
        )
        cal = data.get_calibration()

        s0, shift0 = cal.get_spectrum_at_coor((0, 0, 0), m=0)
        s1, shift1 = cal.get_spectrum_at_coor((1, 2, 3), m=0)

        np.testing.assert_allclose(s0, spectra[0])
        np.testing.assert_allclose(s1, spectra[1])
        assert shift0.value == 7.0
        assert shift0.units == 'GHz'
        assert shift1.value == 7.0
        assert shift1.units == 'GHz'
        f.close()

    def test_create_and_get_calibration_sparse(self, empty_brim_file, sample_data_sparse):
        """Test sparse calibration retrieval with 1D index mapping."""
        f = brim.File(empty_brim_file, mode='r+')
        data = f.create_data_group_sparse(
            sample_data_sparse['PSD'],
            sample_data_sparse['frequency'],
            scanning=sample_data_sparse['scanning'],
        )

        n_points = sample_data_sparse['PSD'].shape[0]
        index = np.arange(n_points, dtype=np.int32) % 2
        spectra = np.stack(
            [sample_data_sparse['PSD'][0, :], sample_data_sparse['PSD'][1, :]],
            axis=0,
        )

        data.create_calibration_group(
            index=index,
            calibration_data=[
                {
                    'spectra': spectra,
                    'shift': 8.0,
                    'shift_units': 'GHz',
                }
            ],
        )
        cal = data.get_calibration()

        s0, _ = cal.get_spectrum_at_coor((0, 0, 0), m=0)
        s1, _ = cal.get_spectrum_at_coor((0, 0, 1), m=0)

        np.testing.assert_allclose(s0, spectra[0])
        np.testing.assert_allclose(s1, spectra[1])
        f.close()

    def test_create_calibration_defaults_shift_units(self, empty_brim_file, sample_data):
        """Test default Shift units are set to GHz when omitted."""
        f = brim.File(empty_brim_file, mode='r+')
        data = f.create_data_group(
            sample_data['PSD'],
            sample_data['frequency'],
            sample_data['pixel_size'],
        )

        with pytest.warns(UserWarning, match="defaulting to GHz"):
            data.create_calibration_group(
                calibration_data=[
                    {
                        'spectra': sample_data['PSD'][0, 0, 0, :][None, :],
                        'shift': 7.2,
                    }
                ],
            )

        cal = data.get_calibration()
        _, shift = cal.get_spectrum_at_coor((0, 0, 0), m=0)
        assert shift.units == 'GHz'
        f.close()

    def test_get_calibration_without_group_raises(self, simple_brim_file):
        """Test get_calibration fails if no calibration group exists."""
        f = brim.File(simple_brim_file)
        data = f.get_data()
        with pytest.raises(ValueError, match='No calibration group found'):
            data.get_calibration()
        f.close()

    def test_get_spectrum_validation_errors(self, empty_brim_file, sample_data):
        """Test coordinate and material validation in calibration retrieval."""
        f = brim.File(empty_brim_file, mode='r+')
        data = f.create_data_group(
            sample_data['PSD'],
            sample_data['frequency'],
            sample_data['pixel_size'],
        )
        data.create_calibration_group(
            calibration_data=[
                {
                    'spectra': sample_data['PSD'][0, 0, 0, :][None, :],
                    'shift': 7.3,
                    'shift_units': 'GHz',
                }
            ],
        )

        cal = data.get_calibration()
        with pytest.raises(ValueError, match='coor must contain 3 values'):
            cal.get_spectrum_at_coor((0, 0), m=0)
        with pytest.raises(IndexError, match='Calibration material 1 not found'):
            cal.get_spectrum_at_coor((0, 0, 0), m=1)
        f.close()

    def test_same_as_calibration_link(self, empty_brim_file, sample_data):
        """Test Same_as links to calibration in another data group."""
        f = brim.File(empty_brim_file, mode='r+')
        data0 = f.create_data_group(
            sample_data['PSD'],
            sample_data['frequency'],
            sample_data['pixel_size'],
        )
        data1 = f.create_data_group(
            sample_data['PSD'],
            sample_data['frequency'],
            sample_data['pixel_size'],
        )

        ref_spectrum = sample_data['PSD'][0, 0, 0, :][None, :]
        data0.create_calibration_group(
            calibration_data=[
                {
                    'spectra': ref_spectrum,
                    'shift': 7.1,
                    'shift_units': 'GHz',
                }
            ],
        )
        data1.create_calibration_group(same_as=0)

        cal = data1.get_calibration()
        spectrum, shift = cal.get_spectrum_at_coor((0, 0, 0), m=0)
        np.testing.assert_allclose(spectrum, ref_spectrum[0])
        assert shift.value == 7.1
        assert shift.units == 'GHz'
        f.close()

    def test_same_as_invalid_reference_raises(self, empty_brim_file, sample_data):
        """Test Same_as fails when referenced data group does not exist."""
        f = brim.File(empty_brim_file, mode='r+')
        data = f.create_data_group(
            sample_data['PSD'],
            sample_data['frequency'],
            sample_data['pixel_size'],
        )

        data.create_calibration_group(same_as=99)
        with pytest.raises(ValueError, match='references non-existing calibration index 99'):
            data.get_calibration()
        f.close()

    def test_multiple_spectra_require_index(self, empty_brim_file, sample_data):
        """Test multiple spectra in one calibration dataset require Index."""
        f = brim.File(empty_brim_file, mode='r+')
        data = f.create_data_group(
            sample_data['PSD'],
            sample_data['frequency'],
            sample_data['pixel_size'],
        )

        spectra = np.stack(
            [sample_data['PSD'][0, 0, 0, :], sample_data['PSD'][0, 0, 1, :]],
            axis=0,
        )
        with pytest.raises(ValueError, match="must contain only one spectrum"):
            data.create_calibration_group(
                calibration_data=[
                    {
                        'spectra': spectra,
                        'shift': 7.4,
                        'shift_units': 'GHz',
                    }
                ],
            )
        f.close()