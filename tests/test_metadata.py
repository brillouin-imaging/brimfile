"""
Unit tests for the Metadata class in brimfile.
"""

import pytest
from datetime import datetime
import zarr

import brimfile as brim
from brimfile.metadata.types import MetadataItemValidity
from brimfile.metadata.schema import METADATA_SCHEMA
from brimfile.validation.versions import get_supported_versions


SUPPORTED_VERSIONS = get_supported_versions()


def _sample_value_for_field(field, *, seed: int):
    """Create a valid value/units pair for a metadata schema field."""
    if field.enum_type is not None:
        enum_values = list(field.enum_type)
        value = enum_values[seed % len(enum_values)].value
        units = None
    elif field.python_type is float:
        value = float(seed) + 0.5
        units = 'arb' if field.units_required else None
    elif field.python_type is str:
        value = f'value_{seed}'
        units = 'arb' if field.units_required else None
    else:
        # The current schema only uses list[float] beyond primitive str/float.
        value = [float(seed), float(seed) + 1.0]
        units = 'arb' if field.units_required else None
    return value, units


_REPRESENTATIVE_FIELD_BY_TYPE = {
    brim.Metadata.Type.Experiment: 'Temperature',
    brim.Metadata.Type.Optics: 'Wavelength',
    brim.Metadata.Type.Brillouin: 'Scattering_angle',
    brim.Metadata.Type.Acquisition: 'Acquisition_time',
    brim.Metadata.Type.Spectrometer: 'Resolution',
}


def _field_for_type(md_type, field_name):
    for field in METADATA_SCHEMA[md_type]:
        if field.name == field_name:
            return field
    raise KeyError(f"Field '{field_name}' not found for metadata type '{md_type.value}'.")


SCHEMA_FIELDS = [
    (md_type, _field_for_type(md_type, field_name))
    for md_type, field_name in _REPRESENTATIVE_FIELD_BY_TYPE.items()
]

UNITS_REQUIRED_FIELDS = [
    (md_type, field)
    for md_type, fields in METADATA_SCHEMA.items()
    for field in fields
    if field.units_required
]

ENUM_FIELDS = [
    (md_type, field)
    for md_type, fields in METADATA_SCHEMA.items()
    for field in fields
    if field.enum_type is not None
]


class TestMetadataItem:
    """Tests for Metadata.Item class."""
    
    def test_create_item_with_units(self):
        """Test creating a metadata item with units."""
        item = brim.Metadata.Item(22.0, 'C')
        assert item.value == 22.0
        assert item.units == 'C'
    
    def test_create_item_without_units(self):
        """Test creating a metadata item without units."""
        item = brim.Metadata.Item(100)
        assert item.value == 100
        assert item.units is None
    
    def test_item_string_representation(self):
        """Test string representation of metadata item."""
        item = brim.Metadata.Item(660, 'nm')
        str_rep = str(item)
        assert isinstance(str_rep, str)
        assert 'MetadataItem' in str_rep


class TestMetadataAddition:
    """Tests for adding metadata."""
    
    def test_add_experiment_metadata(self, simple_brim_file):
        """Test adding experiment metadata."""
        f = brim.File(simple_brim_file, mode='r+')
        data = f.get_data()
        md = data.get_metadata()
        
        Attr = brim.Metadata.Item
        temp = Attr(25.0, 'C')
        md.add(brim.Metadata.Type.Experiment, {'Temperature': temp}, local=True)
        
        # Verify it was added
        retrieved_temp = md['Experiment.Temperature']
        assert retrieved_temp.value == 25.0
        assert retrieved_temp.units == 'C'
        f.close()
    
    def test_add_optics_metadata(self, simple_brim_file):
        """Test adding optics metadata."""
        f = brim.File(simple_brim_file, mode='r+')
        data = f.get_data()
        md = data.get_metadata()
        
        Attr = brim.Metadata.Item
        wavelength = Attr(532, 'nm')
        md.add(brim.Metadata.Type.Optics, {'Wavelength': wavelength}, local=True)
        
        retrieved = md['Optics.Wavelength']
        assert retrieved.value == 532
        f.close()
    
    def test_add_global_metadata(self, simple_brim_file):
        """Test adding global (non-local) metadata."""
        f = brim.File(simple_brim_file, mode='r+')
        data = f.get_data()
        md = data.get_metadata()
        
        with pytest.warns(UserWarning, match="Unknown field 'ExperimenterName'"):
            md.add(
                brim.Metadata.Type.Experiment,
                {'ExperimenterName': 'Test User'},
                local=False,
            )

        retrieved = md['Experiment.ExperimenterName']
        assert retrieved.value == 'Test User'
        f.close()
    
    def test_add_datetime_metadata(self, simple_brim_file):
        """Test adding datetime metadata."""
        f = brim.File(simple_brim_file, mode='r+')
        data = f.get_data()
        md = data.get_metadata()
        
        datetime_now = datetime.now().isoformat()
        md.add(
            brim.Metadata.Type.Experiment,
            {'Datetime': datetime_now},
            local=True
        )
        
        retrieved = md['Experiment.Datetime']
        assert retrieved.value == datetime_now
        f.close()


class TestMetadataRetrieval:
    """Tests for retrieving metadata."""
    
    def test_retrieve_single_metadata(self, simple_brim_file):
        """Test retrieving a single metadata item."""
        f = brim.File(simple_brim_file)
        data = f.get_data()
        md = data.get_metadata()
        
        # Should have temperature from fixture
        temp = md['Experiment.Temperature']
        assert temp is not None
        assert temp.value == 22.0
        assert temp.units == 'C'
        f.close()
    
    def test_retrieve_wavelength(self, simple_brim_file):
        """Test retrieving wavelength metadata."""
        f = brim.File(simple_brim_file)
        data = f.get_data()
        md = data.get_metadata()
        
        wavelength = md['Optics.Wavelength']
        assert wavelength is not None
        assert wavelength.value == 660
        assert wavelength.units == 'nm'
        f.close()
    
    def test_retrieve_nonexistent_metadata(self, simple_brim_file):
        """Test that retrieving non-existent metadata raises KeyError."""
        f = brim.File(simple_brim_file)
        data = f.get_data()
        md = data.get_metadata()
        
        with pytest.raises(KeyError):
            _ = md['Experiment.NonExistent']
        f.close()


class TestMetadataDictConversion:
    """Tests for converting metadata to dictionaries."""
    
    def test_all_to_dict(self, simple_brim_file):
        """Test converting all metadata to dictionary."""
        f = brim.File(simple_brim_file)
        data = f.get_data()
        md = data.get_metadata()
        
        all_md = md.all_to_dict()
        assert all_md is not None
        assert isinstance(all_md, dict)
        f.close()
    
    def test_to_dict_by_type(self, simple_brim_file):
        """Test converting metadata of specific type to dictionary."""
        f = brim.File(simple_brim_file)
        data = f.get_data()
        md = data.get_metadata()
        
        exp_md = md.to_dict(brim.Metadata.Type.Experiment)
        assert exp_md is not None
        assert isinstance(exp_md, dict)
        f.close()
    
    def test_to_dict_optics(self, simple_brim_file):
        """Test converting optics metadata to dictionary."""
        f = brim.File(simple_brim_file)
        data = f.get_data()
        md = data.get_metadata()
        
        optics_md = md.to_dict(brim.Metadata.Type.Optics)
        assert optics_md is not None
        assert 'Wavelength' in optics_md
        f.close()

    def test_to_dict_validate_include_missing_adds_required_fields(self, simple_brim_file):
        """Test validated dict includes required-but-missing schema fields."""
        f = brim.File(simple_brim_file)
        data = f.get_data()
        md = data.get_metadata()

        optics_md = md.to_dict(brim.Metadata.Type.Optics, validate=True, include_missing=True)

        assert optics_md['Wavelength'].value == 660.0
        assert optics_md['Wavelength'].validity == MetadataItemValidity.VALID
        assert optics_md['Power'].value is None
        assert optics_md['Power'].validity == MetadataItemValidity.MISSING_FIELD
        assert optics_md['Lens_NA'].value is None
        assert optics_md['Lens_NA'].validity == MetadataItemValidity.MISSING_FIELD
        f.close()

    def test_to_dict_include_missing_ignored_without_validation(self, simple_brim_file):
        """Test include_missing has no effect unless validate=True."""
        f = brim.File(simple_brim_file)
        data = f.get_data()
        md = data.get_metadata()

        optics_md = md.to_dict(brim.Metadata.Type.Optics, include_missing=True)

        assert 'Wavelength' in optics_md
        assert 'Power' not in optics_md
        assert 'Lens_NA' not in optics_md
        f.close()


class TestMetadataTypes:
    """Tests for different metadata types."""
    
    def test_experiment_type(self):
        """Test Experiment metadata type."""
        assert brim.Metadata.Type.Experiment.value == 'Experiment'
    
    def test_optics_type(self):
        """Test Optics metadata type."""
        assert brim.Metadata.Type.Optics.value == 'Optics'
    
    def test_brillouin_type(self):
        """Test Brillouin metadata type."""
        assert brim.Metadata.Type.Brillouin.value == 'Brillouin'
    
    def test_acquisition_type(self):
        """Test Acquisition metadata type."""
        assert brim.Metadata.Type.Acquisition.value == 'Acquisition'
    
    def test_spectrometer_type(self):
        """Test Spectrometer metadata type."""
        assert brim.Metadata.Type.Spectrometer.value == 'Spectrometer'


class TestMetadataUpdate:
    """Tests for updating existing metadata."""
    
    def test_update_metadata_value(self, simple_brim_file):
        """Test updating an existing metadata value."""
        f = brim.File(simple_brim_file, mode='r+')
        data = f.get_data()
        md = data.get_metadata()
        
        # Add initial value
        Attr = brim.Metadata.Item
        md.add(brim.Metadata.Type.Experiment, {'Temperature': Attr(20.0, 'C')}, local=True)
        
        # Update with new value
        md.add(brim.Metadata.Type.Experiment, {'Temperature': Attr(25.0, 'C')}, local=True)
        
        # Check updated value
        temp = md['Experiment.Temperature']
        assert temp.value == 25.0
        f.close()

    @pytest.mark.parametrize('brim_version', SUPPORTED_VERSIONS)
    @pytest.mark.parametrize('sparse', [False, True])
    def test_add_local_metadata_does_not_overwrite_existing_fields(
        self,
        tmp_path,
        sample_data,
        sample_data_sparse,
        brim_version,
        sparse,
    ):
        """Spec: metadata entries at the same scope/type should accumulate, not replace prior fields."""
        filename = tmp_path / f'metadata_local_merge_v{brim_version}_{"sparse" if sparse else "dense"}.brim.zarr'

        f = brim.File.create(str(filename), store_type=brim.StoreType.AUTO, brim_version=brim_version)
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
        md.add(
            brim.Metadata.Type.Experiment,
            {'Temperature': brim.Metadata.Item(21.0, 'C')},
            local=True,
        )
        md.add(
            brim.Metadata.Type.Experiment,
            {'Datetime': '2026-01-01T10:00:00'},
            local=True,
        )

        exp_md = md.to_dict(brim.Metadata.Type.Experiment)
        assert exp_md['Temperature'].value == 21.0
        assert exp_md['Temperature'].units == 'C'
        assert exp_md['Datetime'].value == '2026-01-01T10:00:00'
        f.close()

        # Independent on-disk check: adding a second metadata key must not drop existing keys.
        root = zarr.open(str(filename), mode='r')
        data_0 = root['Brillouin_data']['Data_0']

        if brim_version.startswith('0.1'):
            assert data_0.attrs['Experiment.Temperature'] == 21.0
            assert data_0.attrs['Experiment.Temperature_units'] == 'C'
            assert data_0.attrs['Experiment.Datetime'] == '2026-01-01T10:00:00'
        else:
            exp_dict = data_0.attrs['Metadata']['Experiment']
            assert exp_dict['Temperature'] == 21.0
            assert exp_dict['Temperature_units'] == 'C'
            assert exp_dict['Datetime'] == '2026-01-01T10:00:00'

    @pytest.mark.parametrize('brim_version', SUPPORTED_VERSIONS)
    def test_add_global_metadata_does_not_overwrite_existing_fields(
        self,
        tmp_path,
        sample_data,
        brim_version,
    ):
        """Spec: /Brillouin_data Metadata should preserve existing fields when adding new ones."""
        filename = tmp_path / f'metadata_global_merge_v{brim_version}.brim.zarr'

        f = brim.File.create(str(filename), store_type=brim.StoreType.AUTO, brim_version=brim_version)
        data = f.create_data_group(
            sample_data['PSD'],
            sample_data['frequency'],
            sample_data['pixel_size'],
            name='d0',
        )

        md = data.get_metadata()
        md.add(
            brim.Metadata.Type.Experiment,
            {'Temperature': brim.Metadata.Item(24.0, 'C')},
            local=False,
        )
        md.add(
            brim.Metadata.Type.Experiment,
            {'Datetime': '2026-01-02T11:30:00'},
            local=False,
        )

        exp_md = md.to_dict(brim.Metadata.Type.Experiment)
        assert exp_md['Temperature'].value == 24.0
        assert exp_md['Temperature'].units == 'C'
        assert exp_md['Datetime'].value == '2026-01-02T11:30:00'
        f.close()

        # Independent on-disk check: global metadata key additions are merged.
        root = zarr.open(str(filename), mode='r')
        exp_dict = root['Brillouin_data'].attrs['Metadata']['Experiment']
        assert exp_dict['Temperature'] == 24.0
        assert exp_dict['Temperature_units'] == 'C'
        assert exp_dict['Datetime'] == '2026-01-02T11:30:00'


class TestLocalVsGlobalMetadata:
    """Tests for local vs global metadata handling."""
    
    def test_local_metadata_in_data_group(self, simple_brim_file):
        """Test that local metadata is specific to data group."""
        f = brim.File(simple_brim_file, mode='r+')
        data = f.get_data()
        md = data.get_metadata()
        
        Attr = brim.Metadata.Item
        with pytest.warns(UserWarning, match="Unknown field 'LocalValue'"):
            md.add(
                brim.Metadata.Type.Experiment,
                {'LocalValue': Attr(100, 'units')},
                local=True
            )
        
        # Should be retrievable from this data group
        local_val = md['Experiment.LocalValue']
        assert local_val.value == 100
        f.close()
    
    def test_global_metadata_accessible(self, simple_brim_file):
        """Test that global metadata is accessible from data group."""
        f = brim.File(simple_brim_file, mode='r+')
        data = f.get_data()
        md = data.get_metadata()
        
        # Global metadata should be accessible
        wavelength = md['Optics.Wavelength']
        assert wavelength is not None
        f.close()

    def test_local_metadata_overrides_global_value_in_dict(self, simple_brim_file):
        """Test local metadata values take precedence over global metadata values."""
        f = brim.File(simple_brim_file, mode='r+')
        data = f.get_data()
        md = data.get_metadata()

        md.add(
            brim.Metadata.Type.Experiment,
            {'Temperature': brim.Metadata.Item(37.0, 'C')},
            local=True,
        )

        exp_md = md.to_dict(brim.Metadata.Type.Experiment)
        assert exp_md['Temperature'].value == 37.0
        assert exp_md['Temperature'].units == 'C'
        f.close()


class TestMetadataValidationIntegration:
    """Integration tests ensuring Metadata.add uses validation logic."""

    def test_add_rejects_unknown_field_with_close_match(self, simple_brim_file):
        """Typos close to schema names should raise instead of being silently accepted."""
        f = brim.File(simple_brim_file, mode='r+')
        data = f.get_data()
        md = data.get_metadata()

        with pytest.raises(ValueError, match='Did you mean'):
            md.add(
                brim.Metadata.Type.Experiment,
                {'Temprature': brim.Metadata.Item(25.0, 'C')},
                local=True,
            )
        f.close()

    def test_add_allows_normalized_field_name(self, simple_brim_file):
        """Known fields are accepted even with non-canonical casing/separators."""
        f = brim.File(simple_brim_file, mode='r+')
        data = f.get_data()
        md = data.get_metadata()

        with pytest.warns(
            UserWarning,
            match="Field name 'temperature' normalized to 'Temperature' for metadata type 'Experiment'",
        ):
            md.add(
                brim.Metadata.Type.Experiment,
                {'temperature': brim.Metadata.Item(23, 'C')},
                local=True,
            )

        temp = md['Experiment.Temperature']
        assert temp.value == 23.0
        assert temp.units == 'C'
        f.close()


class TestMetadataInheritanceMatrix:
    """Coverage for global/local/both metadata visibility and precedence.

    spec clauses (pinned):
    - https://github.com/brillouin-imaging/Brillouin-standard-file/blob/2bb6187fe3ff40194f011d43b51b1bd3887244ed/docs/brim_file_specs.md#L106
    - https://github.com/brillouin-imaging/Brillouin-standard-file/blob/2bb6187fe3ff40194f011d43b51b1bd3887244ed/docs/brim_file_metadata.md#L10
    """

    @pytest.mark.parametrize('file_fixture_name', ['simple_brim_file', 'simple_brim_file_sparse'])
    @pytest.mark.parametrize('md_type, field', SCHEMA_FIELDS)
    @pytest.mark.parametrize('scope', ['global_only', 'local_only', 'both'])
    def test_metadata_scope_precedence_in_to_dict(self, request, file_fixture_name, md_type, field, scope):
        """Spec: Data-level metadata overrides Brillouin_data metadata of the same field."""
        filename = request.getfixturevalue(file_fixture_name)
        f = brim.File(filename, mode='r+')
        data = f.get_data()
        md = data.get_metadata()

        global_value, global_units = _sample_value_for_field(field, seed=11)
        local_value, local_units = _sample_value_for_field(field, seed=77)

        if scope in ('global_only', 'both'):
            md.add(md_type, {field.name: brim.Metadata.Item(global_value, global_units)}, local=False)
        if scope in ('local_only', 'both'):
            md.add(md_type, {field.name: brim.Metadata.Item(local_value, local_units)}, local=True)

        out = md.to_dict(md_type)
        assert field.name in out

        if scope == 'global_only':
            assert out[field.name].value == global_value
            assert out[field.name].units == global_units
        else:
            # local_only + both should both resolve to local
            assert out[field.name].value == local_value
            assert out[field.name].units == local_units

        f.close()

    @pytest.mark.parametrize('file_fixture_name', ['simple_brim_file', 'simple_brim_file_sparse'])
    @pytest.mark.parametrize('md_type, field', SCHEMA_FIELDS)
    @pytest.mark.parametrize('scope', ['global_only', 'local_only', 'both'])
    def test_metadata_scope_precedence_in_getitem(self, request, file_fixture_name, md_type, field, scope):
        """Spec: single-item metadata reads follow the same precedence as dict reads."""
        filename = request.getfixturevalue(file_fixture_name)
        f = brim.File(filename, mode='r+')
        data = f.get_data()
        md = data.get_metadata()

        global_value, global_units = _sample_value_for_field(field, seed=3)
        local_value, local_units = _sample_value_for_field(field, seed=9)

        if scope in ('global_only', 'both'):
            md.add(md_type, {field.name: brim.Metadata.Item(global_value, global_units)}, local=False)
        if scope in ('local_only', 'both'):
            md.add(md_type, {field.name: brim.Metadata.Item(local_value, local_units)}, local=True)

        item = md[f'{md_type.value}.{field.name}']

        if scope == 'global_only':
            assert item.value == global_value
            assert item.units == global_units
        else:
            assert item.value == local_value
            assert item.units == local_units

        f.close()


class TestMetadataSchemaValidationAcrossLayouts:
    """Validation checks for units-required and enum fields across storage layouts."""

    @pytest.mark.parametrize('file_fixture_name', ['simple_brim_file', 'simple_brim_file_sparse'])
    @pytest.mark.parametrize('md_type, field', UNITS_REQUIRED_FIELDS)
    @pytest.mark.parametrize('local', [False, True])
    def test_units_required_fields_accept_valid_units(self, request, file_fixture_name, md_type, field, local):
        """Units-required fields should accept valid values in both global and local scopes."""
        filename = request.getfixturevalue(file_fixture_name)
        f = brim.File(filename, mode='r+')
        data = f.get_data()
        md = data.get_metadata()

        value, units = _sample_value_for_field(field, seed=31)
        md.add(
            md_type,
            {field.name: brim.Metadata.Item(value, units)},
            local=local,
        )

        out = md[f'{md_type.value}.{field.name}']
        assert out.value == value
        assert out.units == units
        f.close()

    @pytest.mark.parametrize('file_fixture_name', ['simple_brim_file', 'simple_brim_file_sparse'])
    @pytest.mark.parametrize('md_type, field', UNITS_REQUIRED_FIELDS)
    @pytest.mark.parametrize('local', [False, True])
    def test_units_required_fields_reject_missing_units(self, request, file_fixture_name, md_type, field, local):
        """Units-required fields should reject values without units in all scopes/layouts."""
        filename = request.getfixturevalue(file_fixture_name)
        f = brim.File(filename, mode='r+')
        data = f.get_data()
        md = data.get_metadata()

        value, _ = _sample_value_for_field(field, seed=44)
        with pytest.raises(ValueError, match='requires units'):
            md.add(
                md_type,
                {field.name: brim.Metadata.Item(value)},
                local=local,
            )

        f.close()

    @pytest.mark.parametrize('file_fixture_name', ['simple_brim_file', 'simple_brim_file_sparse'])
    @pytest.mark.parametrize('md_type, field', ENUM_FIELDS)
    @pytest.mark.parametrize('local', [False, True])
    def test_enum_fields_accept_valid_values(self, request, file_fixture_name, md_type, field, local):
        """Enum fields should accept valid enum members in both local and global metadata."""
        filename = request.getfixturevalue(file_fixture_name)
        f = brim.File(filename, mode='r+')
        data = f.get_data()
        md = data.get_metadata()

        valid_value = list(field.enum_type)[0].value
        md.add(
            md_type,
            {field.name: brim.Metadata.Item(valid_value)},
            local=local,
        )

        out = md[f'{md_type.value}.{field.name}']
        assert out.value == valid_value
        assert out.units is None
        f.close()

    @pytest.mark.parametrize('file_fixture_name', ['simple_brim_file', 'simple_brim_file_sparse'])
    @pytest.mark.parametrize('md_type, field', ENUM_FIELDS)
    @pytest.mark.parametrize('local', [False, True])
    def test_enum_fields_reject_invalid_values(self, request, file_fixture_name, md_type, field, local):
        """Enum fields should reject invalid strings in both local and global metadata."""
        filename = request.getfixturevalue(file_fixture_name)
        f = brim.File(filename, mode='r+')
        data = f.get_data()
        md = data.get_metadata()

        with pytest.raises(ValueError):
            md.add(
                md_type,
                {field.name: brim.Metadata.Item('__invalid_enum_value__')},
                local=local,
            )

        f.close()
