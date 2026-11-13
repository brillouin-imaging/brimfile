"""Unit tests for metadata validation helpers."""

import pytest

from brimfile.metadata import validation
from brimfile.metadata.schema import ScanningStrategy, SpectrometerType, Type
from brimfile.metadata.types import MetadataItem, MetadataItemValidity


def test_validate_single_field_normalizes_known_field_name():
    key, value = validation.validate_single_field(
        Type.Experiment,
        "temperature",
        MetadataItem(22, "C"),
    )

    assert key == "Temperature"
    assert value.value == 22.0
    assert value.units == "C"


def test_validate_single_field_warns_on_normalized_name_when_reporting():
    with pytest.warns(UserWarning, match="normalized to 'Temperature'"):
        key, value = validation.validate_single_field(
            Type.Experiment,
            "temperature",
            MetadataItem(22, "C"),
            report_on_invalid=True,
        )

    assert key == "Temperature"
    assert value.value == 22.0
    assert value.validity == MetadataItemValidity.LIKELY_TYPO


def test_validate_single_field_requires_units_for_schema_field():
    with pytest.raises(ValueError, match="requires units"):
        validation.validate_single_field(
            Type.Experiment,
            "Temperature",
            MetadataItem(22.0),
            report_on_invalid=True,
        )


def test_validate_single_field_marks_missing_units_without_reporting():
    key, value = validation.validate_single_field(
        Type.Experiment,
        "Temperature",
        MetadataItem(22.0),
    )

    assert key == "Temperature"
    assert value.value == 22.0
    assert value.validity == MetadataItemValidity.MISSING_UNITS


def test_validate_single_field_raises_typed_error_for_invalid_primitive():
    with pytest.raises(ValueError, match="Expected float-like value"):
        validation.validate_single_field(
            Type.Experiment,
            "Temperature",
            MetadataItem("hot", "C"),
            report_on_invalid=True,
        )


def test_validate_single_field_marks_invalid_primitive_without_reporting():
    key, value = validation.validate_single_field(
        Type.Experiment,
        "Temperature",
        MetadataItem("hot", "C"),
    )

    assert key == "Temperature"
    assert value.value == "hot"
    assert value.validity == MetadataItemValidity.INVALID_VALUE


def test_validate_single_field_coerces_enum_from_normalized_alias():
    key, value = validation.validate_single_field(
        Type.Acquisition,
        "Scanning_strategy",
        MetadataItem("line-scanning"),
    )

    assert key == "Scanning_strategy"
    assert value.value == ScanningStrategy.line_scanning


def test_validate_single_field_warns_and_coerces_close_enum_match():
    with pytest.warns(UserWarning, match="Interpreting 'line_scannin' as 'line_scanning'"):
        key, value = validation.validate_single_field(
            Type.Acquisition,
            "Scanning_strategy",
            MetadataItem("line_scannin"),
        )

    assert key == "Scanning_strategy"
    assert value.value == ScanningStrategy.line_scanning
    assert value.validity == MetadataItemValidity.VALID


def test_validate_single_field_accepts_enum_member_name():
    key, value = validation.validate_single_field(
        Type.Spectrometer,
        "Type",
        MetadataItem("FP"),
    )

    assert key == "Type"
    assert value.value == SpectrometerType.FP
    assert value.validity == MetadataItemValidity.VALID


def test_validate_single_field_raises_typed_error_for_invalid_enum():
    with pytest.raises(ValueError, match="is not a valid ScanningStrategy"):
        validation.validate_single_field(
            Type.Acquisition,
            "Scanning_strategy",
            MetadataItem("totally-unknown-mode"),
            report_on_invalid=True,
        )


def test_validate_single_field_rejects_unknown_field_with_close_match():
    with pytest.raises(ValueError, match="Did you mean"):
        validation.validate_single_field(
            Type.Experiment,
            "Temprature",
            MetadataItem(22.0, "C"),
            report_on_invalid=True,
        )


def test_validate_single_field_marks_unknown_field_with_close_match_without_reporting():
    key, value = validation.validate_single_field(
        Type.Experiment,
        "Temprature",
        MetadataItem(22.0, "C"),
    )

    assert key == "Temprature"
    assert value.value == 22.0
    assert value.validity == MetadataItemValidity.LIKELY_TYPO


def test_validate_single_field_allows_custom_field_with_warning():
    with pytest.warns(UserWarning, match="Unknown field 'CompletelyCustomField'"):
        key, value = validation.validate_single_field(
            Type.Experiment,
            "CompletelyCustomField",
            MetadataItem("abc"),
            report_on_invalid=True,
        )

    assert key == "CompletelyCustomField"
    assert value.value == "abc"
    assert value.units is None


def test_validate_single_field_marks_custom_field_without_reporting():
    key, value = validation.validate_single_field(
        Type.Experiment,
        "CompletelyCustomField",
        MetadataItem("abc"),
    )

    assert key == "CompletelyCustomField"
    assert value.value == "abc"
    assert value.validity == MetadataItemValidity.UNKNOWN_FIELD


@pytest.mark.parametrize(
    ("field_name", "item", "expected_validity"),
    [
        ("Temperature", MetadataItem(22.0), MetadataItemValidity.MISSING_UNITS),
        ("CompletelyCustomField", MetadataItem("abc"), MetadataItemValidity.UNKNOWN_FIELD),
        ("Temprature", MetadataItem(22.0, "C"), MetadataItemValidity.LIKELY_TYPO),
        ("Temperature", MetadataItem("hot", "C"), MetadataItemValidity.INVALID_VALUE),
    ],
)
def test_validate_single_field_validation_matrix_without_reporting(
    field_name,
    item,
    expected_validity,
):
    _, value = validation.validate_single_field(
        Type.Experiment,
        field_name,
        item,
        report_on_invalid=False,
    )

    assert value.validity == expected_validity


@pytest.mark.parametrize(
    ("field_name", "item", "error_match"),
    [
        ("Temperature", MetadataItem(22.0), "requires units"),
        ("Temprature", MetadataItem(22.0, "C"), "Did you mean"),
        ("Temperature", MetadataItem("hot", "C"), "Expected float-like value"),
    ],
)
def test_validate_single_field_validation_matrix_with_reporting_raises(
    field_name,
    item,
    error_match,
):
    with pytest.raises(ValueError, match=error_match):
        validation.validate_single_field(
            Type.Experiment,
            field_name,
            item,
            report_on_invalid=True,
        )


def test_validate_single_field_validation_matrix_with_reporting_warns_unknown_field():
    with pytest.warns(UserWarning, match="Unknown field 'CompletelyCustomField'"):
        _, value = validation.validate_single_field(
            Type.Experiment,
            "CompletelyCustomField",
            MetadataItem("abc"),
            report_on_invalid=True,
        )

    assert value.validity == MetadataItemValidity.UNKNOWN_FIELD
