"""Unit tests for validation rules in brimfile.validation.main.

The assertions in this file are derived from the BRIM specification at a pinned
commit for stable traceability:
- https://github.com/brillouin-imaging/Brillouin-standard-file/blob/2bb6187fe3ff40194f011d43b51b1bd3887244ed/docs/brim_file_specs.md
- https://github.com/brillouin-imaging/Brillouin-standard-file/blob/2bb6187fe3ff40194f011d43b51b1bd3887244ed/docs/brim_file_metadata.md

Clause links used in this module:
- root brim_version requirement: .../brim_file_specs.md#L94
- Data/Frequency broadcasting: .../brim_file_specs.md#L117
- Sparse scanning constraints: .../brim_file_specs.md#L119
- Data-level metadata override and _arrays: .../brim_file_specs.md#L106
- metadata scopes/override semantics: .../brim_file_metadata.md#L10
"""

import json
import sys

import pytest

from brimfile.validation.main import (
    ValidationLevel,
    ValidationType,
    validate_analysis_group,
    validate_data_group,
    validate_json,
    validate_root_attrs,
)
from brimfile.metadata.schema import Type as MetadataType
from brimfile.validation.versions import get_supported_versions, get_version_rules


SUPPORTED_VERSIONS = get_supported_versions()


def _array(shape, *, dtype="float64", attributes=None):
    return {
        "node_type": "array",
        "shape": tuple(shape),
        "dtype": dtype,
        "attributes": attributes or {},
    }


def _group(*, attributes=None, **children):
    node = {"node_type": "group", "attributes": attributes or {}}
    node.update(children)
    return node


def _analysis_group(*, result_shape, fit_model="Lorentzian"):
    return _group(
        attributes={"Fit_model": fit_model},
        Shift_AS_0=_array(result_shape, attributes={"Units": "GHz"}),
    )


def _non_sparse_data_group(*, psd_shape=(2, 3, 4, 151), frequency_shape=(151,)):
    return _group(
        attributes={
            "Sparse": False,
            "element_size": [1.0, 1.0, 1.0],
            "element_size_units": "um",
        },
        PSD=_array(psd_shape),
        Frequency=_array(frequency_shape, attributes={"Units": "GHz"}),
        Analysis_0=_analysis_group(result_shape=psd_shape[:-1]),
    )


def _sparse_data_group(*, psd_shape=(50, 151), frequency_shape=(151,), scanning=None):
    if scanning is None:
        scanning = _group(
            Cartesian_visualisation=_array((5, 5, 2), dtype="int32")
        )

    return _group(
        attributes={"Sparse": True},
        PSD=_array(psd_shape),
        Frequency=_array(frequency_shape, attributes={"Units": "GHz"}),
        Scanning=scanning,
        Analysis_0=_analysis_group(result_shape=psd_shape[:-1]),
    )


def _errors_matching(errors, *, err_type=None, level=None, path_contains=None):
    """Return the subset of errors matching the given filters."""
    matches = []
    for err in errors:
        if err_type is not None and err.type != err_type:
            continue
        if level is not None and err.level != level:
            continue
        if path_contains is not None and (err.path is None or path_contains not in err.path):
            continue
        matches.append(err)
    return matches


def test_validate_data_group_accepts_minimal_non_sparse_layout():
    node = _non_sparse_data_group()

    errors = validate_data_group(node, path="Brillouin_data/Data_0")

    assert errors == []


def test_validate_data_group_accepts_frequency_broadcasting():
    # spec clause: pinned docs/brim_file_specs.md#L117
    node = _non_sparse_data_group(psd_shape=(2, 3, 4, 151), frequency_shape=(1, 1, 1, 151))

    errors = validate_data_group(node, path="Brillouin_data/Data_0")

    freq_shape_errors = _errors_matching(
        errors,
        err_type=ValidationType.INVALID_SHAPE,
        path_contains="Frequency",
    )
    assert freq_shape_errors == []


def test_validate_data_group_rejects_non_broadcastable_frequency_shape():
    node = _non_sparse_data_group(psd_shape=(2, 3, 4, 151), frequency_shape=(2, 3, 5))

    errors = validate_data_group(node, path="Brillouin_data/Data_0")

    freq_shape_errors = _errors_matching(
        errors,
        err_type=ValidationType.INVALID_SHAPE,
        level=ValidationLevel.CRITICAL,
        path_contains="Frequency",
    )
    assert len(freq_shape_errors) == 1


def test_validate_data_group_requires_scanning_group_for_sparse_data():
    # spec clause: pinned docs/brim_file_specs.md#L119-L120
    node = _group(
        attributes={"Sparse": True},
        PSD=_array((50, 151)),
        Frequency=_array((151,), attributes={"Units": "GHz"}),
        Analysis_0=_analysis_group(result_shape=(50,)),
    )

    errors = validate_data_group(node, path="Brillouin_data/Data_0")

    scanning_errors = _errors_matching(
        errors,
        err_type=ValidationType.MISSING_ARRAY,
        level=ValidationLevel.CRITICAL,
        path_contains="Data_0",
    )
    assert scanning_errors
    assert any("Scanning" in err.message for err in scanning_errors)


def test_validate_data_group_requires_spatial_map_or_cartesian_visualisation_when_sparse():
    node = _sparse_data_group(scanning=_group())

    errors = validate_data_group(node, path="Brillouin_data/Data_0")

    scanning_errors = _errors_matching(
        errors,
        err_type=ValidationType.MISSING_ARRAY,
        level=ValidationLevel.CRITICAL,
        path_contains="Scanning",
    )
    assert len(scanning_errors) == 1


def test_validate_data_group_requires_3d_cartesian_visualisation():
    node = _sparse_data_group(
        scanning=_group(Cartesian_visualisation=_array((10, 5), dtype="int32"))
    )

    errors = validate_data_group(node, path="Brillouin_data/Data_0")

    cart_shape_errors = _errors_matching(
        errors,
        err_type=ValidationType.INVALID_SHAPE,
        level=ValidationLevel.CRITICAL,
        path_contains="Cartesian_visualisation",
    )
    assert len(cart_shape_errors) == 1


def test_validate_analysis_group_requires_fit_model_attribute():
    node = _group(
        attributes={},
        Shift_AS_0=_array((2, 3, 4), attributes={"Units": "GHz"}),
    )

    errors = validate_analysis_group(node, path="Brillouin_data/Data_0/Analysis_0", PSD_shape=(2, 3, 4, 151))

    fit_model_errors = _errors_matching(
        errors,
        err_type=ValidationType.MISSING_ATTRIBUTE,
        level=ValidationLevel.ERROR,
        path_contains="Fit_model",
    )
    assert len(fit_model_errors) == 1


@pytest.mark.xfail(
    strict=True,
    reason="Spec allows analysis arrays to match only spatial PSD dimensions; validator currently enforces PSD[:-1].",
)
def test_spec_allows_analysis_spatial_shape_when_psd_has_extra_parameter_axes():
    # n_PSD = 5: Z, Y, X, parameter_0, frequency
    # Put the quantity key first to avoid early-return key-order artifacts
    # in the current validator implementation.
    analysis_node = {
        "Shift_AS_0": _array((2, 3, 4), attributes={"Units": "GHz"}),
        "node_type": "group",
        "attributes": {"Fit_model": "Lorentzian"},
    }

    errors = validate_analysis_group(
        analysis_node,
        path="Brillouin_data/Data_0/Analysis_0",
        PSD_shape=(2, 3, 4, 7, 151),
    )

    shape_errors = _errors_matching(
        errors,
        err_type=ValidationType.INVALID_SHAPE,
        path_contains="Shift_AS_0",
    )

    # TODO: decide if the specs should allow analysis arrays to match only the spatial dimensions of PSD (i.e. without additional parameter axes).
    assert shape_errors == []


def test_spec_allows_analysis_shape_matching_psd_without_frequency_axis():
    analysis_node = {
        "Shift_AS_0": _array((2, 3, 4, 7), attributes={"Units": "GHz"}),
        "node_type": "group",
        "attributes": {"Fit_model": "Lorentzian"},
    }

    errors = validate_analysis_group(
        analysis_node,
        path="Brillouin_data/Data_0/Analysis_0",
        PSD_shape=(2, 3, 4, 7, 151),
    )

    shape_errors = _errors_matching(
        errors,
        err_type=ValidationType.INVALID_SHAPE,
        path_contains="Shift_AS_0",
    )
    assert shape_errors == []


def test_spec_parameters_shape_for_non_sparse_psd_with_extra_axes():
    # n_PSD = 5 -> Parameters should have n_PSD-3 = 2 dims,
    # with last dim size n_PSD-4 = 1.
    node = _group(
        attributes={
            "Sparse": False,
            "element_size": [1.0, 1.0, 1.0],
            "element_size_units": "um",
        },
        PSD=_array((2, 3, 4, 7, 151)),
        Frequency=_array((151,), attributes={"Units": "GHz"}),
        Parameters=_array((7, 1), attributes={"Name": ["Angle_deg"]}),
        Analysis_0=_analysis_group(result_shape=(2, 3, 4, 7)),
    )

    errors = validate_data_group(node, path="Brillouin_data/Data_0")

    parameter_shape_errors = _errors_matching(
        errors,
        err_type=ValidationType.INVALID_SHAPE,
        path_contains="Parameters",
    )
    assert parameter_shape_errors == []


def test_element_size_units_is_required():
    node = _group(
        attributes={
            "Sparse": False,
            "element_size": [1.0, 1.0, 1.0],
        },
        PSD=_array((2, 3, 4, 151)),
        Frequency=_array((151,), attributes={"Units": "GHz"}),
        Analysis_0=_analysis_group(result_shape=(2, 3, 4)),
    )

    errors = validate_data_group(node, path="Brillouin_data/Data_0")

    units_errors = _errors_matching(
        errors,
        err_type=ValidationType.MISSING_ATTRIBUTE,
        path_contains="element_size_units",
    )
    assert len(units_errors) == 1


def test_spec_fit_model_must_match_allowed_enum_values():
    errors = validate_analysis_group(
        _analysis_group(result_shape=(2, 3, 4), fit_model="NotARealModel"),
        path="Brillouin_data/Data_0/Analysis_0",
        PSD_shape=(2, 3, 4, 151),
    )

    enum_errors = _errors_matching(
        errors,
        err_type=ValidationType.INVALID_VALUE,
        path_contains="Fit_model",
    )
    assert len(enum_errors) == 1


def test_analysis_quantity_validation_is_not_key_order_sensitive():
    node = {
        "node_type": "group",
        "attributes": {"Fit_model": "Lorentzian"},
        "Unrelated": _group(),
        "Shift_AS_0": _group(),
    }

    errors = validate_analysis_group(
        node,
        path="Brillouin_data/Data_0/Analysis_0",
        PSD_shape=(2, 3, 4, 151),
    )

    type_errors = _errors_matching(
        errors,
        err_type=ValidationType.INVALID_TYPE,
        path_contains="Shift_AS_0",
    )
    assert len(type_errors) == 1


def test_validate_root_attrs_accepts_singlepoint_vipa_with_required_feature():
    attrs = {
        "brim_version": "0.1",
        "Subtype": "SinglePoint_VIPA_v0.1",
        "Subtype_features": ["2DArray_per_spectrum"],
    }

    errors = validate_root_attrs(attrs)

    subtype_errors = _errors_matching(
        errors,
        err_type=ValidationType.INVALID_VALUE,
        path_contains="Subtype",
    )
    assert subtype_errors == []


def test_validate_root_attrs_rejects_unknown_subtype_value():
    attrs = {
        "brim_version": "0.1",
        "Subtype": "UnknownSubtype_v0.1",
        "Subtype_features": ["2DArray_per_spectrum"],
    }

    errors = validate_root_attrs(attrs)

    subtype_errors = _errors_matching(
        errors,
        err_type=ValidationType.INVALID_VALUE,
        level=ValidationLevel.ERROR,
        path_contains="Subtype",
    )
    assert len(subtype_errors) == 1


def test_validate_root_attrs_requires_required_singlepoint_feature_in_subtype_features():
    attrs = {
        "brim_version": "0.1",
        "Subtype": "SinglePoint_VIPA_v0.1",
        "Subtype_features": ["Spectral_line"],
    }

    errors = validate_root_attrs(attrs)

    feature_errors = _errors_matching(
        errors,
        err_type=ValidationType.INVALID_VALUE,
        level=ValidationLevel.ERROR,
        path_contains="Subtype_features",
    )
    assert len(feature_errors) == 1


def test_validate_root_attrs_warns_on_unknown_feature_for_singlepoint_subtype():
    attrs = {
        "brim_version": "0.1",
        "Subtype": "SinglePoint_VIPA_v0.1",
        "Subtype_features": ["2DArray_per_spectrum", "UnknownFeature"],
    }

    errors = validate_root_attrs(attrs)

    feature_warnings = _errors_matching(
        errors,
        err_type=ValidationType.INVALID_VALUE,
        level=ValidationLevel.WARNING,
        path_contains="Subtype_features",
    )
    assert len(feature_warnings) == 1


@pytest.mark.parametrize('brim_version', SUPPORTED_VERSIONS)
def test_validate_root_attrs_accepts_all_supported_versions(brim_version):
    # spec clause: pinned docs/brim_file_specs.md#L94
    attrs = {
        "brim_version": brim_version,
    }

    errors = validate_root_attrs(attrs)

    version_errors = _errors_matching(
        errors,
        err_type=ValidationType.INVALID_VALUE,
        level=ValidationLevel.ERROR,
        path_contains="brim_version",
    )
    assert version_errors == []


def test_validate_root_attrs_rejects_missing_brim_version():
    attrs = {}

    errors = validate_root_attrs(attrs)

    missing_version_errors = _errors_matching(
        errors,
        err_type=ValidationType.MISSING_ATTRIBUTE,
        level=ValidationLevel.ERROR,
        path_contains="brim_version",
    )
    assert len(missing_version_errors) == 1


@pytest.mark.parametrize('invalid_version', ['9.9', 'garbage', '1'])
def test_validate_root_attrs_lists_all_registered_versions_when_unsupported(invalid_version):
    attrs = {
        "brim_version": invalid_version,
    }

    errors = validate_root_attrs(attrs)

    version_errors = _errors_matching(
        errors,
        err_type=ValidationType.INVALID_VALUE,
        level=ValidationLevel.ERROR,
        path_contains="brim_version",
    )
    assert len(version_errors) == 1
    for supported_version in SUPPORTED_VERSIONS:
        assert supported_version in version_errors[0].message


def test_validate_data_group_v0_2_rejects_flattened_metadata_overrides():
    node = _non_sparse_data_group()
    node["attributes"]["Experiment.Temperature"] = 25.0
    node["attributes"]["Experiment.Temperature_units"] = "C"

    errors = validate_data_group(
        node,
        path="Brillouin_data/Data_0",
        version_rules=get_version_rules("0.2"),
    )

    flattened_errors = _errors_matching(
        errors,
        err_type=ValidationType.INVALID_VALUE,
        level=ValidationLevel.ERROR,
        path_contains="Experiment.Temperature",
    )
    assert len(flattened_errors) == 2


def test_validate_data_group_v0_2_accepts_nested_metadata_override_and_arrays_group():
    # spec clauses: pinned docs/brim_file_specs.md#L106-L108 and docs/brim_file_metadata.md#L10-L18
    node = _non_sparse_data_group()
    node["attributes"]["Metadata"] = {
        "Experiment": {
            "_arrays": ["Temperature"],
        }
    }
    node["Metadata"] = _group(
        Experiment=_group(
            Temperature=_array((2, 3, 4), dtype="float64"),
        )
    )

    errors = validate_data_group(
        node,
        path="Brillouin_data/Data_0",
        version_rules=get_version_rules("0.2"),
    )

    metadata_errors = _errors_matching(
        errors,
        path_contains="Data_0/Metadata",
    )
    assert metadata_errors == []


def test_validate_data_group_v0_2_requires_arrays_declared_in_metadata():
    node = _non_sparse_data_group()
    node["attributes"]["Metadata"] = {
        "Experiment": {
            "_arrays": ["Temperature"],
        }
    }
    node["Metadata"] = _group(
        Experiment=_group()
    )

    errors = validate_data_group(
        node,
        path="Brillouin_data/Data_0",
        version_rules=get_version_rules("0.2"),
    )

    missing_array_errors = _errors_matching(
        errors,
        err_type=ValidationType.MISSING_ARRAY,
        level=ValidationLevel.ERROR,
        path_contains="Data_0/Metadata/Experiment/Temperature",
    )
    assert len(missing_array_errors) == 1


def test_validate_json_v0_1_does_not_import_v0_2_rules_module():
    sys.modules.pop("brimfile.validation.versions.v0_2", None)

    descriptor = {
        "node_type": "group",
        "attributes": {
            "brim_version": "0.1",
        },
        "Brillouin_data": {
            "node_type": "group",
            "attributes": {
                "Metadata": {},
            },
            "Data_0": {
                "node_type": "group",
                "attributes": {
                    "Sparse": True,
                },
                "PSD": _array((3, 10)),
                "Frequency": _array((10,), attributes={"Units": "GHz"}),
                "Scanning": _group(Cartesian_visualisation=_array((1, 1, 3), dtype="int32")),
                "Analysis_0": _analysis_group(result_shape=(3,)),
            },
        },
    }

    validate_json(json.dumps(descriptor))

    assert "brimfile.validation.versions.v0_2" not in sys.modules


def test_singlepoint_vipa_requires_2darray_storage_in_data_or_calibration_raw_data():
    node = _non_sparse_data_group()

    errors = validate_data_group(
        node,
        path="Brillouin_data/Data_0",
        subtype="SinglePoint_VIPA_v0.1",
        subtype_features={"2DArray_per_spectrum"},
    )

    raw_feature_errors = _errors_matching(
        errors,
        err_type=ValidationType.MISSING_ARRAY,
        level=ValidationLevel.ERROR,
        path_contains="Data_0",
    )
    assert any("2DArray_per_spectrum" in err.message for err in raw_feature_errors)


def test_singlepoint_vipa_accepts_raw_data_with_matching_spatial_shape():
    node = _non_sparse_data_group()
    node["Raw_data"] = _array((2, 3, 4, 12, 24))

    errors = validate_data_group(
        node,
        path="Brillouin_data/Data_0",
        subtype="SinglePoint_VIPA_v0.1",
        subtype_features={"2DArray_per_spectrum"},
    )

    raw_shape_errors = _errors_matching(
        errors,
        err_type=ValidationType.INVALID_SHAPE,
        path_contains="Raw_data",
    )
    assert raw_shape_errors == []


def test_singlepoint_vipa_rejects_raw_data_with_wrong_spatial_prefix():
    node = _non_sparse_data_group()
    node["Raw_data"] = _group(**{"2DArray_per_spectrum": _array((2, 3, 99, 12, 24))})

    errors = validate_data_group(
        node,
        path="Brillouin_data/Data_0",
        subtype="SinglePoint_VIPA_v0.1",
        subtype_features={"2DArray_per_spectrum"},
    )

    raw_shape_errors = _errors_matching(
        errors,
        err_type=ValidationType.INVALID_SHAPE,
        level=ValidationLevel.ERROR,
        path_contains="Raw_data",
    )
    assert len(raw_shape_errors) >= 1


def test_singlepoint_vipa_validates_spectral_line_shape_in_analysis_group():
    node = _non_sparse_data_group()
    node["Raw_data"] = _array((2, 3, 4, 12, 24))
    node["Analysis_0"]["Spectral_line"] = _array((2, 3, 4, 3))

    errors = validate_data_group(
        node,
        path="Brillouin_data/Data_0",
        subtype="SinglePoint_VIPA_v0.1",
        subtype_features={"2DArray_per_spectrum", "Spectral_line"},
    )

    spectral_line_errors = _errors_matching(
        errors,
        err_type=ValidationType.INVALID_SHAPE,
        level=ValidationLevel.ERROR,
        path_contains="Analysis_0/Spectral_line",
    )
    assert len(spectral_line_errors) == 1


def test_singlepoint_vipa_accepts_calibration_raw_data_mirror_structure():
    node = _non_sparse_data_group()
    node["Calibration"] = _group(
        **{
            "0": _array((5, 151)),
            "Index": _array((2, 3, 4), dtype="int32"),
        },
        Raw_data=_group(
            **{
                "0": _group(
                **{
                    "2DArray_per_spectrum": _array((5, 12, 24)),
                    "Spectral_line": _array((5, 4)),
                }
                )
            }
        ),
    )

    errors = validate_data_group(
        node,
        path="Brillouin_data/Data_0",
        subtype="SinglePoint_VIPA_v0.1",
        subtype_features={"2DArray_per_spectrum", "Spectral_line"},
    )

    cal_raw_errors = _errors_matching(
        errors,
        err_type=ValidationType.MISSING_GROUP,
        path_contains="Calibration/Raw_data",
    )
    assert cal_raw_errors == []


def test_singlepoint_vipa_accepts_calibration_raw_data_numeric_material_groups():
    node = _non_sparse_data_group()
    node["Calibration"] = _group(
        **{
            "0": _array((5, 151), attributes={"Shift": 7.0, "Units": "GHz"}),
            "1": _array((3, 151), attributes={"Shift": 8.5, "Units": "GHz"}),
            "Index": _array((2, 3, 4), dtype="int32"),
        },
        Raw_data=_group(
            **{
                "0": _group(
                    **{
                        "2DArray_per_spectrum": _array((5, 12, 24)),
                        "Spectral_line": _array((5, 4)),
                    }
                ),
                "1": _group(
                    **{
                        "2DArray_per_spectrum": _array((3, 2, 12, 24)),
                        "Spectral_line": _array((3, 2, 4)),
                    }
                ),
            }
        ),
    )

    errors = validate_data_group(
        node,
        path="Brillouin_data/Data_0",
        subtype="SinglePoint_VIPA_v0.1",
        subtype_features={"2DArray_per_spectrum", "Spectral_line"},
    )

    calibration_errors = _errors_matching(
        errors,
        level=ValidationLevel.ERROR,
        path_contains="Calibration/Raw_data",
    )
    assert calibration_errors == []


def test_validate_json_rejects_invalid_json_payload():
    with pytest.raises(ValueError, match="Invalid JSON format"):
        validate_json("{not-json")


def test_validate_json_rejects_non_object_top_level():
    with pytest.raises(ValueError, match="JSON object"):
        validate_json(json.dumps([1, 2, 3]))


def test_validate_json_reports_missing_brillouin_data_group():
    descriptor = {
        "node_type": "group",
        "attributes": {
            "brim_version": SUPPORTED_VERSIONS[0],
        },
    }

    errors = validate_json(json.dumps(descriptor))

    missing_group_errors = _errors_matching(
        errors,
        err_type=ValidationType.MISSING_GROUP,
        level=ValidationLevel.CRITICAL,
        path_contains="Brillouin_data",
    )
    assert len(missing_group_errors) == 1


def test_validate_root_attrs_rejects_non_string_subtype():
    attrs = {
        "brim_version": SUPPORTED_VERSIONS[0],
        "Subtype": 123,
    }

    errors = validate_root_attrs(attrs)

    subtype_type_errors = _errors_matching(
        errors,
        err_type=ValidationType.INVALID_TYPE,
        level=ValidationLevel.ERROR,
        path_contains="Subtype",
    )
    assert len(subtype_type_errors) == 1


def test_validate_root_attrs_rejects_non_list_subtype_features():
    attrs = {
        "brim_version": SUPPORTED_VERSIONS[0],
        "Subtype": "SinglePoint_VIPA_v0.1",
        "Subtype_features": "2DArray_per_spectrum",
    }

    errors = validate_root_attrs(attrs)

    feature_type_errors = _errors_matching(
        errors,
        err_type=ValidationType.INVALID_TYPE,
        level=ValidationLevel.ERROR,
        path_contains="Subtype_features",
    )
    assert len(feature_type_errors) == 1


def test_validate_json_requires_all_metadata_types_in_brillouin_data_metadata():
    descriptor = {
        "node_type": "group",
        "attributes": {
            "brim_version": SUPPORTED_VERSIONS[0],
        },
        "Brillouin_data": {
            "node_type": "group",
            "attributes": {
                "Metadata": {
                    "Experiment": {},
                },
            },
            "Data_0": {
                "node_type": "group",
                "attributes": {
                    "Sparse": True,
                },
                "PSD": _array((3, 10)),
                "Frequency": _array((10,), attributes={"Units": "GHz"}),
                "Scanning": _group(Cartesian_visualisation=_array((1, 1, 3), dtype="int32")),
                "Analysis_0": _analysis_group(result_shape=(3,)),
            },
        },
    }

    errors = validate_json(json.dumps(descriptor))

    missing_md_errors = _errors_matching(
        errors,
        err_type=ValidationType.MISSING_METADATA,
        level=ValidationLevel.ERROR,
        path_contains="Metadata.",
    )
    # Only Experiment is present; remaining metadata sections should be reported missing.
    assert len(missing_md_errors) == len(MetadataType) - 1


def test_validate_data_group_warns_when_scanning_present_for_non_sparse_data():
    node = _non_sparse_data_group()
    node["Scanning"] = _group(Cartesian_visualisation=_array((2, 3, 4), dtype="int32"))

    errors = validate_data_group(node, path="Brillouin_data/Data_0")

    scanning_warnings = _errors_matching(
        errors,
        err_type=ValidationType.INVALID_VALUE,
        level=ValidationLevel.WARNING,
        path_contains="Scanning",
    )
    assert len(scanning_warnings) == 1


def test_validate_data_group_requires_frequency_units_attribute():
    node = _non_sparse_data_group()
    node["Frequency"] = _array((151,), attributes={})

    errors = validate_data_group(node, path="Brillouin_data/Data_0")

    unit_errors = _errors_matching(
        errors,
        err_type=ValidationType.MISSING_UNITS,
        level=ValidationLevel.ERROR,
        path_contains="Frequency",
    )
    assert len(unit_errors) == 1


def test_validate_data_group_requires_parameters_when_psd_has_extra_dimensions():
    node = _non_sparse_data_group(psd_shape=(2, 3, 4, 7, 151), frequency_shape=(151,))
    node.pop("Parameters", None)
    node["Analysis_0"] = _analysis_group(result_shape=(2, 3, 4, 7))

    errors = validate_data_group(node, path="Brillouin_data/Data_0")

    parameters_errors = _errors_matching(
        errors,
        err_type=ValidationType.MISSING_ARRAY,
        level=ValidationLevel.ERROR,
        path_contains="Data_0",
    )
    assert len(parameters_errors) == 1
    assert "Parameters" in parameters_errors[0].message


def test_validate_json_reports_missing_root_attributes_group():
    descriptor = {
        "node_type": "group",
        "Brillouin_data": {
            "node_type": "group",
            "attributes": {"Metadata": {}},
            "Data_0": _non_sparse_data_group(),
        },
    }

    errors = validate_json(json.dumps(descriptor))

    root_attr_errors = _errors_matching(
        errors,
        err_type=ValidationType.MISSING_ATTRIBUTE,
        level=ValidationLevel.CRITICAL,
    )
    assert root_attr_errors


def test_validate_json_reports_unsupported_version_blocks_brillouin_group_validation():
    descriptor = {
        "node_type": "group",
        "attributes": {"brim_version": "9.9"},
        "Brillouin_data": {
            "node_type": "group",
            "attributes": {"Metadata": {}},
            "Data_0": _non_sparse_data_group(),
        },
    }

    errors = validate_json(json.dumps(descriptor))

    unsupported_errors = _errors_matching(
        errors,
        err_type=ValidationType.INVALID_VALUE,
        level=ValidationLevel.ERROR,
        path_contains="brim_version",
    )
    assert len(unsupported_errors) >= 1
    assert any("Cannot validate 'Brillouin_data'" in err.message for err in unsupported_errors)


def test_validate_root_attrs_warns_when_subtype_features_missing():
    attrs = {
        "brim_version": SUPPORTED_VERSIONS[0],
        "Subtype": "SinglePoint_VIPA_v0.1",
    }

    errors = validate_root_attrs(attrs)

    missing_features_warning = _errors_matching(
        errors,
        err_type=ValidationType.MISSING_ATTRIBUTE,
        level=ValidationLevel.WARNING,
        path_contains="Subtype_features",
    )
    assert len(missing_features_warning) == 1


def test_validate_root_attrs_rejects_non_string_subtype_feature_entries():
    attrs = {
        "brim_version": SUPPORTED_VERSIONS[0],
        "Subtype": "SinglePoint_VIPA_v0.1",
        "Subtype_features": ["2DArray_per_spectrum", 123],
    }

    errors = validate_root_attrs(attrs)

    feature_type_errors = _errors_matching(
        errors,
        err_type=ValidationType.INVALID_TYPE,
        level=ValidationLevel.ERROR,
        path_contains="Subtype_features",
    )
    assert len(feature_type_errors) == 1


def test_validate_data_group_rejects_non_boolean_sparse_attribute():
    node = _non_sparse_data_group()
    node["attributes"]["Sparse"] = "yes"

    errors = validate_data_group(node, path="Brillouin_data/Data_0")

    sparse_type_errors = _errors_matching(
        errors,
        err_type=ValidationType.INVALID_TYPE,
        level=ValidationLevel.ERROR,
        path_contains="Sparse",
    )
    assert len(sparse_type_errors) == 1


def test_validate_data_group_requires_psd_array():
    node = _non_sparse_data_group()
    del node["PSD"]

    errors = validate_data_group(node, path="Brillouin_data/Data_0")

    psd_missing_errors = _errors_matching(
        errors,
        err_type=ValidationType.MISSING_ARRAY,
        level=ValidationLevel.CRITICAL,
        path_contains="Data_0",
    )
    assert any("PSD" in err.message for err in psd_missing_errors)


def test_validate_data_group_requires_frequency_array():
    node = _non_sparse_data_group()
    del node["Frequency"]

    errors = validate_data_group(node, path="Brillouin_data/Data_0")

    freq_missing_errors = _errors_matching(
        errors,
        err_type=ValidationType.MISSING_ARRAY,
        level=ValidationLevel.CRITICAL,
        path_contains="Data_0",
    )
    assert any("Frequency" in err.message for err in freq_missing_errors)


def test_validate_data_group_rejects_spatial_map_with_mismatched_coordinate_lengths():
    node = _sparse_data_group(
        scanning=_group(
            Spatial_map=_group(
                x=_array((5,), dtype="float64"),
                y=_array((6,), dtype="float64"),
            )
        )
    )

    errors = validate_data_group(node, path="Brillouin_data/Data_0")

    spatial_len_errors = _errors_matching(
        errors,
        err_type=ValidationType.INVALID_SHAPE,
        level=ValidationLevel.CRITICAL,
        path_contains="Scanning/Spatial_map",
    )
    assert len(spatial_len_errors) >= 1


def test_validate_data_group_warns_when_cartesian_visualisation_size_differs_from_sparse_psd():
    node = _sparse_data_group(
        psd_shape=(12, 151),
        scanning=_group(
            Cartesian_visualisation=_array((2, 2, 2), dtype="int32")
        ),
    )

    errors = validate_data_group(node, path="Brillouin_data/Data_0")

    mismatch_warnings = _errors_matching(
        errors,
        err_type=ValidationType.INVALID_SHAPE,
        level=ValidationLevel.WARNING,
        path_contains="Cartesian_visualisation",
    )
    assert len(mismatch_warnings) == 1
