"""Unit tests for validation rules in brimfile.validation.main.

The assertions in this file are derived from the BRIM specification:
https://github.com/brillouin-imaging/Brillouin-standard-file/blob/main/docs/brim_file_specs.md
"""

import pytest

from brimfile.validation.main import (
    ValidationLevel,
    ValidationType,
    validate_analysis_group,
    validate_data_group,
    validate_root_attrs,
)


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


@pytest.mark.xfail(
    strict=True,
    reason="Spec defines Fit_model as enum; validator currently checks only presence.",
)
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
