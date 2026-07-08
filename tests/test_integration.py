"""
Integration tests for brimfile - testing complete workflows.
"""

import pytest
import numpy as np
import os
from datetime import datetime
import json


import brimfile as brim
from brimfile.metadata.schema import METADATA_SCHEMA
from brimfile.validation.main import ValidationLevel, ValidationType, validate_json
from brimfile.validation.json_descriptor import generate_json_descriptor
from brimfile.validation.versions import get_supported_versions


SUPPORTED_VERSIONS = get_supported_versions()


def _required_metadata_payload(md_type, *, seed: int):
    payload = {}
    cursor = seed
    for field in METADATA_SCHEMA[md_type]:
        if not field.required:
            continue
        if field.enum_type is not None:
            value = list(field.enum_type)[0].value
            units = None
        elif field.python_type is float:
            value = float(cursor)
            units = 'arb' if field.units_required else None
        elif field.python_type is str:
            value = f'value_{cursor}'
            units = 'arb' if field.units_required else None
        else:
            value = [float(cursor), float(cursor) + 1.0]
            units = 'arb' if field.units_required else None

        payload[field.name] = brim.Metadata.Item(value, units)
        cursor += 1
    return payload


def _add_minimally_valid_required_metadata(metadata_obj):
    # spec: root Brillouin_data metadata should include all metadata types.
    for i, md_type in enumerate(brim.Metadata.Type):
        metadata_obj.add(md_type, _required_metadata_payload(md_type, seed=10 * (i + 1)), local=False)


def _descriptor_errors_matching(errors, *, err_type=None, level=None, path_contains=None):
    out = []
    for err in errors:
        if err_type is not None and err.type != err_type:
            continue
        if level is not None and err.level != level:
            continue
        if path_contains is not None and (err.path is None or path_contains not in err.path):
            continue
        out.append(err)
    return out


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


class TestJsonDescriptorConformance:
    """Spec-conformance checks based on generated JSON descriptors from real files."""

    @pytest.mark.parametrize('brim_version', SUPPORTED_VERSIONS)
    @pytest.mark.parametrize('sparse', [False, True])
    @pytest.mark.parametrize('metadata_scope', ['global_only', 'local_only', 'both'])
    def test_written_file_descriptor_validates_for_version_sparse_and_metadata_scope(
        self,
        tmp_path,
        sample_data,
        sample_data_sparse,
        brim_version,
        sparse,
        metadata_scope,
    ):
        """spec: descriptor generated from written file should validate without errors/criticals."""
        filename = os.path.join(
            tmp_path,
            f'descriptor_v{brim_version}_{"sparse" if sparse else "dense"}_{metadata_scope}.brim.zarr',
        )

        f = brim.File.create(filename, store_type=brim.StoreType.AUTO, brim_version=brim_version)
        if sparse:
            data = f.create_data_group_sparse(
                sample_data_sparse['PSD'],
                sample_data_sparse['frequency'],
                scanning=sample_data_sparse['scanning'],
                name='d0',
            )
        else:
            data = f.create_data_group(
                sample_data['PSD'],
                sample_data['frequency'],
                sample_data['pixel_size'],
                name='d0',
            )

        md = data.get_metadata()
        _add_minimally_valid_required_metadata(md)

        # spec: local metadata should override global values at Data_{n} level.
        if metadata_scope in ('local_only', 'both'):
            md.add(
                brim.Metadata.Type.Experiment,
                {'Temperature': brim.Metadata.Item(37.5, 'C')},
                local=True,
            )
        if metadata_scope == 'local_only':
            md.add(
                brim.Metadata.Type.Experiment,
                {'Temperature': brim.Metadata.Item(22.0, 'C')},
                local=False,
            )

        analysis_input = sample_data_sparse if sparse else sample_data
        data.create_analysis_results_group(
            {
                'shift': analysis_input['shift'],
                'shift_units': 'GHz',
                'width': analysis_input['width'],
                'width_units': 'GHz',
            },
            {
                'shift': analysis_input['shift'],
                'shift_units': 'GHz',
                'width': analysis_input['width'],
                'width_units': 'GHz',
            },
            fit_model=brim.Data.AnalysisResults.FitModel.Lorentzian,
        )

        f.close()
        reader = brim.File(filename, mode='r')
        descriptor_json = generate_json_descriptor(reader._file)
        reader.close()

        validation_errors = validate_json(descriptor_json)
        blocking = [
            err for err in validation_errors
            if err.level in (ValidationLevel.ERROR, ValidationLevel.CRITICAL)
        ]
        assert blocking == []

    @pytest.mark.parametrize('brim_version', SUPPORTED_VERSIONS)
    def test_generated_descriptor_rejects_missing_root_brim_version(self, tmp_path, sample_data, brim_version):
        """spec: root brim_version is required and must be present in descriptor."""
        filename = os.path.join(tmp_path, f'descriptor_missing_version_{brim_version}.brim.zarr')

        f = brim.File.create(filename, store_type=brim.StoreType.AUTO, brim_version=brim_version)
        data = f.create_data_group(
            sample_data['PSD'],
            sample_data['frequency'],
            sample_data['pixel_size'],
            name='d0',
        )
        md = data.get_metadata()
        _add_minimally_valid_required_metadata(md)
        data.create_analysis_results_group(
            {
                'shift': sample_data['shift'],
                'shift_units': 'GHz',
                'width': sample_data['width'],
                'width_units': 'GHz',
            },
            {
                'shift': sample_data['shift'],
                'shift_units': 'GHz',
                'width': sample_data['width'],
                'width_units': 'GHz',
            },
            fit_model=brim.Data.AnalysisResults.FitModel.Lorentzian,
        )
        f.close()
        reader = brim.File(filename, mode='r')
        descriptor_json = generate_json_descriptor(reader._file)
        reader.close()

        descriptor = json.loads(descriptor_json)
        del descriptor['attributes']['brim_version']

        validation_errors = validate_json(json.dumps(descriptor))
        missing_version_errors = _descriptor_errors_matching(
            validation_errors,
            err_type=ValidationType.MISSING_ATTRIBUTE,
            level=ValidationLevel.ERROR,
            path_contains='brim_version',
        )
        assert len(missing_version_errors) == 1
