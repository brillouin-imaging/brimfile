from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
import re
from math import prod

from ..metadata.types import MetadataItem, MetadataItemValidity
from ..metadata.validation import validate_single_field
from ..metadata.schema import Type as MetadataType
from ..metadata.schema import METADATA_SCHEMA

from ..subtypes.constants import SubType
from ..subtypes.constants import FEATURES as SUBTYPE_FEATURES

from ..constants import brim_obj_names
from ..utils import concatenate_paths
from .utils import get_node_type, get_attributes, get_array_shape_and_dtype, \
                    is_numeric_dtype, generate_attr_path, _NodeType, broadcast_shapes
from .versions import get_supported_versions, get_version_rules, UnsupportedBrimVersionError
from .versions.base import VersionRules

class ValidationLevel(Enum):
    """Severity for validation issues."""
    WARNING = 'warning'
    ERROR = 'error'
    CRITICAL = 'critical'

class ValidationType(Enum):
    UNKNOWN_ERROR = 'unknown error'
    MISSING_GROUP = 'missing group'
    MISSING_ARRAY = 'missing array'
    MISSING_ATTRIBUTE = 'missing attribute'
    MISSING_UNITS = 'missing units'
    MISSING_METADATA = 'missing metadata'
    INVALID_NAME = 'invalid name'
    INVALID_VALUE = 'invalid value'
    INVALID_TYPE = 'invalid type'
    INVALID_SHAPE = 'invalid shape'

@dataclass(frozen=True, slots=True)
class ValidationError:
    level: ValidationLevel
    type: ValidationType
    path: str = None # The full path of the object (group or array) where the error occurred. None if the error is not specific to a particular path.
    message: str = ""


def _validate_singlepoint_vipa_data_group(
    node: dict,
    path: str,
    *,
    sparse: bool,
    PSD_shape: tuple[int, ...] | None,
) -> list[ValidationError]:
    errs: list[ValidationError] = []
    if PSD_shape is None:
        return errs

    expected_spatial_dims = 1 if sparse else 3
    expected_psd_ndim = expected_spatial_dims + 1
    if len(PSD_shape) != expected_psd_ndim:
        errs.append(ValidationError(
            level=ValidationLevel.ERROR,
            type=ValidationType.INVALID_SHAPE,
            path=concatenate_paths(path, 'PSD'),
            message=(
                f"For subtype '{SubType.SinglePoint_VIPA_v0_1.value}', the 'PSD' array must have {expected_psd_ndim} "
                f"dimensions ({expected_spatial_dims} spatial + frequency), found shape {PSD_shape}."
            )
        ))
    spatial_shape = PSD_shape[:-1]

    def _validate_2d_raw_array(raw_array_node: dict, raw_array_path: str, *, expected_prefix: tuple[int, ...]) -> tuple[int | None, bool]:
        """Validate a 2DArray_per_spectrum array shape and dtype.

        The array must match the provided spectrum-prefix dimensions, followed by
        either:
        - one optional replicate axis and image axes (M, N), or
        - only image axes (M, N).

        Args:
            raw_array_node: Candidate array node.
            raw_array_path: Full path used for validation messages.
            expected_prefix: Required leading dimensions derived from the
                corresponding spectra array.

        Returns:
            A tuple ``(replicate_count, shape_is_valid)`` where
            ``replicate_count`` is the replicate axis length when present,
            otherwise ``None``.
        """
        raw_shape, raw_dtype = get_array_shape_and_dtype(raw_array_node)
        if raw_shape is None or raw_dtype is None:
            errs.append(ValidationError(
                level=ValidationLevel.CRITICAL,
                type=ValidationType.MISSING_ATTRIBUTE,
                path=raw_array_path,
                message=f"The '2DArray_per_spectrum' array at '{raw_array_path}' must define 'shape' and 'dtype'."
            ))
            return None, False
        if not is_numeric_dtype(raw_dtype):
            errs.append(ValidationError(
                level=ValidationLevel.ERROR,
                type=ValidationType.INVALID_TYPE,
                path=raw_array_path,
                message=f"The '2DArray_per_spectrum' array at '{raw_array_path}' must have a numeric dtype, found '{raw_dtype}'."
            ))

        with_replicates = len(raw_shape) == len(expected_prefix) + 3
        without_replicates = len(raw_shape) == len(expected_prefix) + 2
        if not (with_replicates or without_replicates):
            errs.append(ValidationError(
                level=ValidationLevel.ERROR,
                type=ValidationType.INVALID_SHAPE,
                path=raw_array_path,
                message=(
                    f"The '2DArray_per_spectrum' array at '{raw_array_path}' has invalid shape {raw_shape}. "
                    f"Expected prefix {expected_prefix}, optionally one replicate axis, then image axes (M, N)."
                )
            ))
            return None, False

        if tuple(raw_shape[:len(expected_prefix)]) != expected_prefix:
            errs.append(ValidationError(
                level=ValidationLevel.ERROR,
                type=ValidationType.INVALID_SHAPE,
                path=raw_array_path,
                message=(
                    f"The '2DArray_per_spectrum' array at '{raw_array_path}' must match prefix {expected_prefix} "
                    f"from the corresponding spectra data, found {raw_shape}."
                )
            ))

        if len(raw_shape) >= 2 and (raw_shape[-2] < 1 or raw_shape[-1] < 1):
            errs.append(ValidationError(
                level=ValidationLevel.ERROR,
                type=ValidationType.INVALID_SHAPE,
                path=raw_array_path,
                message=f"The image dimensions (M, N) in '{raw_array_path}' must be >= 1, found {raw_shape[-2:]}."
            ))

        replicate_count = raw_shape[len(expected_prefix)] if with_replicates else None
        return replicate_count, True

    def _validate_spectral_line_array(
        sl_node: dict,
        sl_path: str,
        *,
        base_count: int | None = None,
        replicate_count: int | None = None,
        spatial_prefix: tuple[int, ...] | None = None,
    ) -> None:
        """Validate Spectral_line shape and dtype for subtype-specific contexts.

        The final axis must always contain 4 coordinates
        (y_start, x_start, y_end, x_end). Prefix dimensions are validated
        according to where the array is stored:
        - analysis-level: either global ``(4,)`` or one value per spatial point
          ``spatial_prefix + (4,)``;
        - calibration raw-data level: optional global ``(4,)``, per-calibration
          ``(base_count, 4)``, or per-calibration-and-replicate
          ``(base_count, replicate_count, 4)``.

        Args:
            sl_node: Candidate Spectral_line array node.
            sl_path: Full path used for validation messages.
            base_count: Number of calibration spectra for the mirrored
                calibration item.
            replicate_count: Number of raw replicates per calibration spectrum,
                when present.
            spatial_prefix: Spatial dimensions of PSD for analysis-level
                Spectral_line validation.
        """
        sl_shape, sl_dtype = get_array_shape_and_dtype(sl_node)
        if sl_shape is None or sl_dtype is None:
            errs.append(ValidationError(
                level=ValidationLevel.CRITICAL,
                type=ValidationType.MISSING_ATTRIBUTE,
                path=sl_path,
                message=f"The 'Spectral_line' array at '{sl_path}' must define 'shape' and 'dtype'."
            ))
            return
        if not is_numeric_dtype(sl_dtype):
            errs.append(ValidationError(
                level=ValidationLevel.ERROR,
                type=ValidationType.INVALID_TYPE,
                path=sl_path,
                message=f"The 'Spectral_line' array at '{sl_path}' must have a numeric dtype, found '{sl_dtype}'."
            ))
        if len(sl_shape) < 1 or sl_shape[-1] != 4:
            errs.append(ValidationError(
                level=ValidationLevel.ERROR,
                type=ValidationType.INVALID_SHAPE,
                path=sl_path,
                message=f"The last dimension of 'Spectral_line' at '{sl_path}' must be 4 (y_start, x_start, y_end, x_end), found shape {sl_shape}."
            ))
            return

        if spatial_prefix is not None:
            if tuple(sl_shape[:-1]) not in (tuple(), spatial_prefix):
                errs.append(ValidationError(
                    level=ValidationLevel.ERROR,
                    type=ValidationType.INVALID_SHAPE,
                    path=sl_path,
                    message=(
                        f"The 'Spectral_line' array at '{sl_path}' must have either shape (4,) or "
                        f"{spatial_prefix + (4,)}, found {sl_shape}."
                    )
                ))
            return

        allowed_prefixes: set[tuple[int, ...]] = {tuple()}
        if base_count is not None:
            allowed_prefixes.add((base_count,))
        if base_count is not None and replicate_count is not None:
            allowed_prefixes.add((base_count, replicate_count))

        if tuple(sl_shape[:-1]) not in allowed_prefixes:
            errs.append(ValidationError(
                level=ValidationLevel.ERROR,
                type=ValidationType.INVALID_SHAPE,
                path=sl_path,
                message=(
                    f"The 'Spectral_line' array at '{sl_path}' has invalid shape {sl_shape}. "
                    f"Allowed prefixes before the final coordinate axis are {sorted(allowed_prefixes)}."
                )
            ))

    has_2d_feature_data = False

    if 'Raw_data' in node:
        raw_data_path = concatenate_paths(path, 'Raw_data')
        raw_data_node = node['Raw_data'] 
        if get_node_type(raw_data_node) != _NodeType.GROUP:
            errs.append(ValidationError(
                level=ValidationLevel.ERROR,
                type=ValidationType.INVALID_TYPE,
                path=raw_data_path,
                message=f"For subtype 'SinglePoint_VIPA_v0.1', '{raw_data_path}' must be a group."
            ))
        else:
            _2DArray_per_spectrum_path = concatenate_paths(raw_data_path, '2DArray_per_spectrum')
            if not '2DArray_per_spectrum' in raw_data_node:
                errs.append(ValidationError(
                    level=ValidationLevel.ERROR,
                    type=ValidationType.MISSING_ARRAY,
                    path=raw_data_path,
                    message=f"For subtype 'SinglePoint_VIPA_v0.1', '{raw_data_path}' must contain a '2DArray_per_spectrum' array."
                ))
            else:
                _validate_2d_raw_array(raw_data_node['2DArray_per_spectrum'], _2DArray_per_spectrum_path, expected_prefix=tuple(spatial_shape))
                has_2d_feature_data = True

    calibration_group = node.get('Calibration', None)
    calibration_raw_group = None
    if calibration_group is not None:
        if get_node_type(calibration_group) != _NodeType.GROUP:
            errs.append(ValidationError(
                level=ValidationLevel.ERROR,
                type=ValidationType.INVALID_TYPE,
                path=concatenate_paths(path, 'Calibration'),
                message=f"The '{concatenate_paths(path, 'Calibration')}' node must be a group when present."
            ))
        if 'Raw_data' in calibration_group:
            calibration_raw_group = calibration_group['Raw_data']
            cal_raw_path = concatenate_paths(path, 'Calibration/Raw_data')
            if get_node_type(calibration_raw_group) != _NodeType.GROUP:
                errs.append(ValidationError(
                    level=ValidationLevel.ERROR,
                    type=ValidationType.INVALID_TYPE,
                    path=cal_raw_path,
                    message=f"For subtype 'SinglePoint_VIPA_v0.1', '{cal_raw_path}' must be a group."
                ))
                calibration_raw_group = None

    if calibration_raw_group is not None:
        calibration_items = [
            k for k in calibration_group.keys()
            if re.match(r'^(\d+)$', k)
        ]
        for item_name in calibration_items:
            cal_item = calibration_group[item_name]
            cal_item_path = concatenate_paths(path, f'Calibration/{item_name}')
            cal_raw_item_path = concatenate_paths(path, f'Calibration/Raw_data/{item_name}')

            if get_node_type(cal_item) != _NodeType.ARRAY:
                continue
            cal_item_shape, _ = get_array_shape_and_dtype(cal_item)
            if cal_item_shape is None or len(cal_item_shape) < 1:
                errs.append(ValidationError(
                    level=ValidationLevel.CRITICAL,
                    type=ValidationType.MISSING_ATTRIBUTE,
                    path=cal_item_path,
                    message=f"The calibration array '{cal_item_path}' must define a non-empty shape."
                ))
                continue

            if item_name not in calibration_raw_group:
                errs.append(ValidationError(
                    level=ValidationLevel.ERROR,
                    type=ValidationType.MISSING_GROUP,
                    path=cal_raw_item_path,
                    message=(
                        f"Missing mirrored group '{cal_raw_item_path}' for calibration item '{cal_item_path}'."
                    )
                ))
                continue
            if get_node_type(calibration_raw_group[item_name]) != _NodeType.GROUP:
                errs.append(ValidationError(
                    level=ValidationLevel.ERROR,
                    type=ValidationType.INVALID_TYPE,
                    path=cal_raw_item_path,
                    message=f"The node '{cal_raw_item_path}' must be a group containing '2DArray_per_spectrum'."
                ))
                continue

            raw_item_group = calibration_raw_group[item_name]
            if '2DArray_per_spectrum' not in raw_item_group:
                errs.append(ValidationError(
                    level=ValidationLevel.ERROR,
                    type=ValidationType.MISSING_ARRAY,
                    path=cal_raw_item_path,
                    message=f"The group '{cal_raw_item_path}' must contain '2DArray_per_spectrum'."
                ))
                continue

            raw_array_path = concatenate_paths(cal_raw_item_path, '2DArray_per_spectrum')
            replicate_count, valid_shape = _validate_2d_raw_array(
                raw_item_group['2DArray_per_spectrum'],
                raw_array_path,
                expected_prefix=tuple(cal_item_shape[:1]),
            )
            has_2d_feature_data = True

            if 'Spectral_line' in raw_item_group and valid_shape:
                _validate_spectral_line_array(
                    raw_item_group['Spectral_line'],
                    concatenate_paths(cal_raw_item_path, 'Spectral_line'),
                    base_count=cal_item_shape[0],
                    replicate_count=replicate_count,
                )

    if not has_2d_feature_data:
        errs.append(ValidationError(
            level=ValidationLevel.ERROR,
            type=ValidationType.MISSING_ARRAY,
            path=path,
            message=(
                f"Subtype 'SinglePoint_VIPA_v0.1' requires feature '2DArray_per_spectrum': "
                f"provide '{concatenate_paths(path, 'Raw_data')}' and/or mirrored arrays under "
                f"'{concatenate_paths(path, 'Calibration/Raw_data')}'."
            )
        ))

    analysis_groups = [
        key for key in node.keys()
        if re.match(brim_obj_names.data.analysis_results + r'_(\d+)$', key)
    ]
    for analysis_name in analysis_groups:
        analysis_path = concatenate_paths(path, analysis_name)
        analysis_group = node[analysis_name]
        if get_node_type(analysis_group) != _NodeType.GROUP:
            continue
        if 'Spectral_line' in analysis_group:
            _validate_spectral_line_array(
                analysis_group['Spectral_line'],
                concatenate_paths(analysis_path, 'Spectral_line'),
                spatial_prefix=spatial_shape,
            )

    return errs


def validate_metadata(
    metadata_type: MetadataType,
    metadata_dict: dict[str],
    *,
    path_prefix: str = 'Brillouin_data',
    require_all_required_fields: bool = True,
) -> list[ValidationError]:
    errs: list[ValidationError] = []

    def generate_metadata_path(field_name: str) -> str:
        path = generate_attr_path(path_prefix, 'Metadata')
        return f"{path}.{metadata_type.value}.{field_name}"
    
    def map_MetadataItemValidity_to_ValidationError(validity: MetadataItemValidity, *,
                                                    canonical_field_name: str,  path: str) -> ValidationError | None:
        if validity != MetadataItemValidity.VALID and validity != MetadataItemValidity.NOT_CHECKED:                
            match validity :
                case MetadataItemValidity.LIKELY_TYPO:
                    level = ValidationLevel.ERROR
                    error_type  = ValidationType.INVALID_NAME
                    message = f"Metadata field '{field_name}' is likely a typo. Did you mean '{canonical_field_name}'?"
                case MetadataItemValidity.UNKNOWN_FIELD:
                    level = ValidationLevel.WARNING
                    error_type  = ValidationType.INVALID_NAME
                    message = f"Metadata field '{field_name}' is not recognized by the schema but may be a valid field outside the schema. The closest match within the schema is '{canonical_field_name}'."
                case MetadataItemValidity.MISSING_UNITS:
                    level = ValidationLevel.ERROR
                    error_type  = ValidationType.MISSING_UNITS
                    message = f"Metadata field '{field_name}' is missing units, but units are required."
                case MetadataItemValidity.INVALID_TYPE:
                    level = ValidationLevel.ERROR
                    error_type  = ValidationType.INVALID_TYPE
                    message = f"Metadata field '{field_name}' has an invalid type. Expected type is {METADATA_SCHEMA[metadata_type].get_field(canonical_field_name).python_type.__name__}."
                case MetadataItemValidity.INVALID_VALUE:
                    level = ValidationLevel.ERROR
                    error_type  = ValidationType.INVALID_VALUE
                    message = f"Metadata field '{field_name}' has an invalid value."
                case _:
                    level = ValidationLevel.ERROR
                    error_type  = ValidationType.UNKNOWN_ERROR
                    message = f"An unknown error occurred while validating the metadata field '{field_name}'"
            return ValidationError(
                    level=level,
                    type=error_type,
                    path=path,
                    message=message
                )
        return None
    
    # validate fields in the file
    for field_name in metadata_dict:
        if field_name.endswith('_units'):
            continue # skip the units fields, they will be checked together with the corresponding value fields
        value = metadata_dict[field_name]
        units = None
        if f"{field_name}_units" in metadata_dict:
            units = metadata_dict[f"{field_name}_units"]
        canonical_field_name, value = validate_single_field(metadata_type, field_name, MetadataItem(value, units))
        validity = value.get_validity()
        err = map_MetadataItemValidity_to_ValidationError(validity, canonical_field_name=canonical_field_name, 
                                                                path = generate_metadata_path(field_name))
        if err is not None:
            errs.append(err)
    # check for missing required fields
    if require_all_required_fields:
        for field in METADATA_SCHEMA[metadata_type]:
            if not field.required:
                # only check for required fields, optional fields can be missing without causing an error
                continue
            field_name = field.name
            if field_name not in metadata_dict:
                errs.append(ValidationError(
                    level=ValidationLevel.ERROR,
                    type=ValidationType.MISSING_METADATA,
                    path=generate_metadata_path(field_name),
                    message=f"The required field '{field_name}' is missing for metadata type '{metadata_type.value}'."
                ))
    return errs


def _validate_nested_data_group_metadata_override(
    node: dict,
    path: str,
    *,
    sparse: bool,
    PSD_shape: tuple[int, ...] | None,
) -> list[ValidationError]:
    errs: list[ValidationError] = []
    attrs = get_attributes(node)
    if attrs is None:
        return errs

    metadata_attr_path = generate_attr_path(path, 'Metadata')
    metadata_attr = attrs.get('Metadata', None)
    if metadata_attr is None:
        return errs

    if not isinstance(metadata_attr, dict):
        errs.append(ValidationError(
            level=ValidationLevel.ERROR,
            type=ValidationType.INVALID_TYPE,
            path=metadata_attr_path,
            message=f"The local 'Metadata' attribute in '{path}' must be a dictionary, found {type(metadata_attr).__name__}."
        ))
        return errs

    arrays_by_type: dict[MetadataType, set[str]] = {}
    known_type_names = {md_type.value for md_type in MetadataType}
    for metadata_type_name, metadata_values in metadata_attr.items():
        type_attr_path = f"{metadata_attr_path}.{metadata_type_name}"
        if metadata_type_name not in known_type_names:
            errs.append(ValidationError(
                level=ValidationLevel.WARNING,
                type=ValidationType.INVALID_NAME,
                path=type_attr_path,
                message=f"Metadata type '{metadata_type_name}' is not recognized by the schema."
            ))
            continue

        metadata_type = MetadataType(metadata_type_name)
        if not isinstance(metadata_values, dict):
            errs.append(ValidationError(
                level=ValidationLevel.ERROR,
                type=ValidationType.INVALID_TYPE,
                path=type_attr_path,
                message=f"The '{metadata_type_name}' entry in local metadata must be a dictionary, found {type(metadata_values).__name__}."
            ))
            continue

        arrays_hint = metadata_values.get('_arrays', None)
        if arrays_hint is not None:
            if not isinstance(arrays_hint, list) or not all(isinstance(field_name, str) for field_name in arrays_hint):
                errs.append(ValidationError(
                    level=ValidationLevel.ERROR,
                    type=ValidationType.INVALID_TYPE,
                    path=f"{type_attr_path}._arrays",
                    message="The '_arrays' hint must be a list of metadata field names (strings)."
                ))
            else:
                arrays_by_type[metadata_type] = set(arrays_hint)

        metadata_values_without_arrays = {
            key: value
            for key, value in metadata_values.items()
            if key != '_arrays'
        }
        errs.extend(validate_metadata(
            metadata_type,
            metadata_values_without_arrays,
            path_prefix=path,
            require_all_required_fields=False,
        ))

    if not arrays_by_type:
        return errs

    data_metadata_group_path = concatenate_paths(path, 'Metadata')
    data_metadata_group = node.get('Metadata', None)
    if data_metadata_group is None:
        errs.append(ValidationError(
            level=ValidationLevel.ERROR,
            type=ValidationType.MISSING_GROUP,
            path=data_metadata_group_path,
            message=(
                f"The '{data_metadata_group_path}' group is required when '_arrays' is declared "
                "in the local metadata override attribute."
            )
        ))
        return errs

    if get_node_type(data_metadata_group) != _NodeType.GROUP:
        errs.append(ValidationError(
            level=ValidationLevel.ERROR,
            type=ValidationType.INVALID_TYPE,
            path=data_metadata_group_path,
            message=f"The '{data_metadata_group_path}' node must be a group."
        ))
        return errs

    expected_shape: tuple[int, ...] | None = None
    if PSD_shape is not None:
        expected_shape = (PSD_shape[0],) if sparse else PSD_shape[:3]

    for metadata_type, array_fields in arrays_by_type.items():
        type_group_path = concatenate_paths(data_metadata_group_path, metadata_type.value)
        type_group = data_metadata_group.get(metadata_type.value, None)
        if type_group is None:
            errs.append(ValidationError(
                level=ValidationLevel.ERROR,
                type=ValidationType.MISSING_GROUP,
                path=type_group_path,
                message=f"Missing metadata arrays group '{type_group_path}' declared via '_arrays'."
            ))
            continue
        if get_node_type(type_group) != _NodeType.GROUP:
            errs.append(ValidationError(
                level=ValidationLevel.ERROR,
                type=ValidationType.INVALID_TYPE,
                path=type_group_path,
                message=f"The node '{type_group_path}' must be a group containing metadata arrays."
            ))
            continue

        for field_name in sorted(array_fields):
            field_path = concatenate_paths(type_group_path, field_name)
            if field_name not in type_group:
                errs.append(ValidationError(
                    level=ValidationLevel.ERROR,
                    type=ValidationType.MISSING_ARRAY,
                    path=field_path,
                    message=f"Missing metadata array '{field_path}' declared in '_arrays'."
                ))
                continue
            if get_node_type(type_group[field_name]) != _NodeType.ARRAY:
                errs.append(ValidationError(
                    level=ValidationLevel.ERROR,
                    type=ValidationType.INVALID_TYPE,
                    path=field_path,
                    message=f"The metadata node '{field_path}' must be an array."
                ))
                continue

            field_shape, _ = get_array_shape_and_dtype(type_group[field_name])
            if field_shape is None:
                errs.append(ValidationError(
                    level=ValidationLevel.CRITICAL,
                    type=ValidationType.MISSING_ATTRIBUTE,
                    path=field_path,
                    message=f"The metadata array '{field_path}' must define 'shape'."
                ))
                continue
            if expected_shape is not None and tuple(field_shape) != tuple(expected_shape):
                errs.append(ValidationError(
                    level=ValidationLevel.ERROR,
                    type=ValidationType.INVALID_SHAPE,
                    path=field_path,
                    message=(
                        f"The metadata array '{field_path}' must match spatial PSD dimensions {expected_shape}, "
                        f"found shape {field_shape}."
                    )
                ))

    return errs


def _validate_data_group_metadata_overrides(
    node: dict,
    path: str,
    *,
    sparse: bool,
    PSD_shape: tuple[int, ...] | None,
    version_rules: VersionRules,
) -> list[ValidationError]:
    errs: list[ValidationError] = []
    attrs = get_attributes(node)
    if attrs is None:
        return errs

    metadata_type_names = [metadata_type.value for metadata_type in MetadataType]

    if version_rules.uses_nested_data_group_metadata_attribute():
        flattened_fields = []
        for attr_name in attrs:
            if any(attr_name.startswith(f"{md_type_name}.") for md_type_name in metadata_type_names):
                flattened_fields.append(attr_name)
        for attr_name in sorted(flattened_fields):
            errs.append(ValidationError(
                level=ValidationLevel.ERROR,
                type=ValidationType.INVALID_VALUE,
                path=generate_attr_path(path, attr_name),
                message=(
                    f"Flattened metadata override '{attr_name}' is not valid for brim_version "
                    f"'{version_rules.version}'. Use the nested 'Metadata' attribute instead."
                )
            ))

        if version_rules.supports_data_group_metadata_arrays_group():
            errs.extend(_validate_nested_data_group_metadata_override(
                node,
                path,
                sparse=sparse,
                PSD_shape=PSD_shape,
            ))

    return errs


def _get_default_version_rules() -> VersionRules:
    """Return a deterministic fallback ruleset for direct validate_data_group calls.

    Full-file validation resolves rules from the root brim_version and passes
    them down explicitly. This fallback avoids hardcoding a specific version for
    callers that validate a data group in isolation.
    """
    supported_versions = get_supported_versions()
    if not supported_versions:
        raise RuntimeError("No brim validation versions are registered.")
    return get_version_rules(supported_versions[0])

def validate_analysis_group(node: dict, path: str, *, sparse=False, PSD_shape=None) -> list[ValidationError]:
    errs: list[ValidationError] = []
    attrs = get_attributes(node)
    if attrs is None or 'Fit_model' not in attrs:
        errs.append(ValidationError(
            level=ValidationLevel.ERROR,
            type=ValidationType.MISSING_ATTRIBUTE,
            path=generate_attr_path(path, 'Fit_model'),
            message=f"The analysis group '{path}' is missing the required 'Fit_model' attribute."
        ))
    else:
        fit_model = attrs['Fit_model']
        allowed_fit_models = {'other', 'Lorentzian', 'DHO', 'Voigt'}
        if not isinstance(fit_model, str):
            errs.append(ValidationError(
                level=ValidationLevel.ERROR,
                type=ValidationType.INVALID_TYPE,
                path=generate_attr_path(path, 'Fit_model'),
                message=(
                    f"The 'Fit_model' attribute in '{path}' must be a string, "
                    f"found '{type(fit_model).__name__}'."
                )
            ))
        elif fit_model not in allowed_fit_models:
            errs.append(ValidationError(
                level=ValidationLevel.ERROR,
                type=ValidationType.INVALID_VALUE,
                path=generate_attr_path(path, 'Fit_model'),
                message=(
                    f"The 'Fit_model' attribute in '{path}' must be one of "
                    f"{sorted(allowed_fit_models)}, found '{fit_model}'."
                )
            ))

    def _check_quantity(name: str) -> bool:
        """Validate all arrays for a given analysis quantity.

        A quantity is considered valid if at least one corresponding peak type
        exists and passes validation checks. Any validation issues are appended
        to ``errs``.

        Args:
            name: Quantity prefix to validate (for example, ``'Shift'``).

        Returns:
            ``True`` if at least one matching and valid peak array is found,
            otherwise ``False``.
        """
        _any_match_found = False
        for qt in node.keys():
            match = re.match(name + r'_(AS|S)_(\d+)$', qt)
            if match:
                if get_node_type(node[qt]) != _NodeType.ARRAY:
                    errs.append(ValidationError(
                        level=ValidationLevel.ERROR,
                        type=ValidationType.INVALID_TYPE,
                        path=concatenate_paths(path, qt),
                        message=f"The '{qt}' node in the analysis group '{path}' must be an array, found '{get_node_type(node[qt])}'."
                    ))
                else:
                    _any_match_found = True
                    qt_shape, qt_dtype = get_array_shape_and_dtype(node[qt])
                    if qt_shape is None or qt_dtype is None:
                        errs.append(ValidationError(
                            level=ValidationLevel.CRITICAL,
                            type=ValidationType.MISSING_ATTRIBUTE,
                            path=concatenate_paths(path, qt),
                            message=f"The '{qt}' array in the analysis group '{path}' must have 'shape' and 'dtype' attributes."
                        ))
                    elif not is_numeric_dtype(qt_dtype):
                        errs.append(ValidationError(
                            level=ValidationLevel.ERROR,
                            type=ValidationType.INVALID_TYPE,
                            path=concatenate_paths(path, qt),
                            message=f"The '{qt}' array in the analysis group '{path}' must have a numeric dtype, found '{qt_dtype}'."
                        ))
                    if qt_shape is not None and PSD_shape is not None:
                        if qt_shape != PSD_shape[:-1]:
                            errs.append(ValidationError(
                                level=ValidationLevel.CRITICAL,
                                type=ValidationType.INVALID_SHAPE,
                                path=concatenate_paths(path, qt),
                                message=f"The '{qt}' array in the analysis group '{path}' has an incompatible shape {qt_shape} with the shape of the 'PSD' array {PSD_shape}."
                            ))
        return _any_match_found
    _check_quantity('Shift')
    _check_quantity('Width')
    _check_quantity('Amplitude')
    _check_quantity('Offset')
    # TODO: check the Fit_error group
    return errs

def validate_data_group(
    node: dict,
    path: str,
    *,
    version_rules: VersionRules | None = None,
    subtype: str | None = None,
    subtype_features: set[str] | None = None,
) -> list[ValidationError]:
    errs: list[ValidationError] = []
    if version_rules is None:
        version_rules = _get_default_version_rules()
    node_type = get_node_type(node)
    if node_type != _NodeType.GROUP:
        errs.append(ValidationError(
            level=ValidationLevel.CRITICAL,
            type=ValidationType.INVALID_TYPE,
            path=path,
            message=f"The data group '{path}' must be a group, found '{node_type}'."
        ))
    # Validate the attributes of the data group
    attrs = get_attributes(node)
    if attrs is None:
        errs.append(ValidationError(
            level=ValidationLevel.ERROR,
            type=ValidationType.MISSING_ATTRIBUTE,
            path=path,
            message=f"The data group '{path}' must define at least the 'Sparse' or the 'element_size' attribute."
        ))
    sparse: bool = False
    if attrs is not None:
        if 'Sparse' in attrs:
            sparse = attrs['Sparse']
        if not isinstance(sparse, bool):
            errs.append(ValidationError(
                level=ValidationLevel.ERROR,
                type=ValidationType.INVALID_TYPE,
                path=generate_attr_path(path, 'Sparse'),
                message=f"The 'Sparse' attribute of the data group '{path}' must be a boolean, found '{type(sparse).__name__}'."
            ))
        if sparse is False and 'element_size' not in attrs:
            errs.append(ValidationError(
                level=ValidationLevel.ERROR,
                type=ValidationType.MISSING_ATTRIBUTE,
                path=path,
                message=f"The data group '{path}' must have an 'element_size' attribute when 'Sparse' is False."
            ))
        if 'element_size' in attrs:
            element_size = attrs['element_size']
            if not (isinstance(element_size, list) and len(element_size) == 3):
                errs.append(ValidationError(
                    level=ValidationLevel.ERROR,
                    type=ValidationType.INVALID_VALUE,
                    path=generate_attr_path(path, 'element_size'),
                    message=f"The 'element_size' attribute of the data group '{path}' must be a list of three numbers (in the order z, y, x), found '{element_size}'. Unused dimensions can be set to None."
                ))
            if 'element_size_units' not in attrs:
                errs.append(ValidationError(
                    level=ValidationLevel.ERROR,
                    type=ValidationType.MISSING_ATTRIBUTE,
                    path=generate_attr_path(path, 'element_size_units'),
                    message=f"The 'element_size_units' attribute of the data group '{path}' is required when 'element_size' is provided, but it is missing."
                ))

    # Validate the arrays in the data group

    # Validate the PSD array
    PSD_shape = None
    if 'PSD' not in node:
        errs.append(ValidationError(
            level=ValidationLevel.CRITICAL,
            type=ValidationType.MISSING_ARRAY,
            path=path,
            message=f"The data group '{path}' must contain a 'PSD' array."
        ))
    else:
        PSD_shape, PSD_dtype = get_array_shape_and_dtype(node['PSD'])
        if PSD_shape is None or PSD_dtype is None:
            errs.append(ValidationError(
                level=ValidationLevel.CRITICAL,
                type=ValidationType.MISSING_ATTRIBUTE,
                path=concatenate_paths(path, 'PSD'),
                message=f"The 'PSD' array in the data group '{path}' must have 'shape' and 'dtype' attributes."
            ))
        elif not is_numeric_dtype(PSD_dtype):
            errs.append(ValidationError(
                level=ValidationLevel.ERROR,
                type=ValidationType.INVALID_TYPE,
                path=concatenate_paths(path, 'PSD'),
                message=f"The 'PSD' array in the data group '{path}' must have a numeric dtype, found '{PSD_dtype}'."
            ))
        if PSD_shape is not None:
            if not sparse and len(PSD_shape) < 4:
                errs.append(ValidationError(
                    level=ValidationLevel.CRITICAL,
                    type=ValidationType.INVALID_SHAPE,
                    path=concatenate_paths(path, 'PSD'),
                    message=f"The 'PSD' array in the data group '{path}' must be at least 4-dimensional for non-sparse data, found shape {PSD_shape}."
                ))
            elif sparse and len(PSD_shape) < 2:
                errs.append(ValidationError(
                    level=ValidationLevel.CRITICAL,
                    type=ValidationType.INVALID_SHAPE,
                    path=concatenate_paths(path, 'PSD'),
                    message=f"The 'PSD' array in the data group '{path}' must be at least 2-dimensional for sparse data, found shape {PSD_shape}."
                ))

    # Validate the frequency array         
    if 'Frequency' not in node:
        errs.append(ValidationError(
            level=ValidationLevel.CRITICAL,
            type=ValidationType.MISSING_ARRAY,
            path=path,
            message=f"The data group '{path}' must contain a 'Frequency' array."
        ))
    else:
        Frequency_shape, Frequency_dtype = get_array_shape_and_dtype(node['Frequency'])
        if Frequency_shape is None or Frequency_dtype is None:
            errs.append(ValidationError(
                level=ValidationLevel.CRITICAL,
                type=ValidationType.MISSING_ATTRIBUTE,
                path=concatenate_paths(path, 'Frequency'),
                message=f"The 'Frequency' array in the data group '{path}' must have 'shape' and 'dtype' attributes."
            ))
        elif not is_numeric_dtype(Frequency_dtype):
            errs.append(ValidationError(
                level=ValidationLevel.ERROR,
                type=ValidationType.INVALID_TYPE,
                path=concatenate_paths(path, 'Frequency'),
                message=f"The 'Frequency' array in the data group '{path}' must have a numeric dtype, found '{Frequency_dtype}'."
            ))
        if PSD_shape is not None and Frequency_shape is not None:
            try:
                broadcast_shapes(PSD_shape, Frequency_shape)
            except ValueError as e:
                errs.append(ValidationError(
                    level=ValidationLevel.CRITICAL,
                    type=ValidationType.INVALID_SHAPE,
                    path=concatenate_paths(path, 'Frequency'),
                    message=f"The 'Frequency' array in the data group '{path}' has an incompatible shape {Frequency_shape} that cannot be broadcast to the shape of the 'PSD' array {PSD_shape}. Error details: {e}"
                ))
        attrs = get_attributes(node['Frequency'])
        if attrs is None or "Units" not in attrs:
            errs.append(ValidationError(
                level=ValidationLevel.ERROR,
                type=ValidationType.MISSING_UNITS,
                path=concatenate_paths(path, 'Frequency'),
                message=f"The 'Frequency' array in the data group '{path}' is missing the required 'Units' attribute."
            ))

    # Validate the Scanning group
    if not sparse and "Scanning" in node:
        errs.append(ValidationError(
            level=ValidationLevel.WARNING,
            type=ValidationType.INVALID_VALUE,
            path=concatenate_paths(path, "Scanning"),
            message=f"The 'Scanning' group in '{path}' is not supported for non-sparse data. It will probably be ignored by most software."
        ))
    if sparse and ("Scanning" not in node or get_node_type(node["Scanning"]) != _NodeType.GROUP):
        errs.append(ValidationError(
            level=ValidationLevel.CRITICAL,
            type=ValidationType.MISSING_ARRAY,
            path=path,
            message=f"The data group '{path}' must contain a 'Scanning' group when 'Sparse' is True."
        ))
    elif "Scanning" in node:
        if get_node_type(node["Scanning"]) != _NodeType.GROUP:
            errs.append(ValidationError(
                level=ValidationLevel.CRITICAL,
                type=ValidationType.INVALID_TYPE,
                path=concatenate_paths(path, "Scanning"),
                message=f"The 'Scanning' node in the data group '{path}' must be a group, found '{get_node_type(node['Scanning'])}'."
            ))
        scanning_group = node["Scanning"]
        if sparse and not ("Spatial_map" in scanning_group or "Cartesian_visualisation" in scanning_group):
            errs.append(ValidationError(
                level=ValidationLevel.CRITICAL,
                type=ValidationType.MISSING_ARRAY,
                path=concatenate_paths(path, "Scanning"),
                message=f"The 'Scanning' group in the data group '{path}' must contain at least a 'Spatial_map' group or a 'Cartesian_visualisation' array when 'Sparse' is True."
            ))
        # Validate the Spatial_map group if it exists
        if "Spatial_map" in scanning_group:
            spatial_map_group = scanning_group["Spatial_map"]
            def _get_coord_len(coor: str) -> int | None:
                if coor in spatial_map_group:
                    coor_shape, coor_dtype = get_array_shape_and_dtype(spatial_map_group[coor])
                    if coor_shape is not None and coor_dtype is not None:
                        if not is_numeric_dtype(coor_dtype):
                            errs.append(ValidationError(
                                level=ValidationLevel.ERROR,
                                type=ValidationType.INVALID_TYPE,
                                path=concatenate_paths(path, f"Scanning/Spatial_map/{coor}"),
                                message=f"The '{coor}' array in the 'Spatial_map' group of the data group '{path}' must have a numeric dtype, found '{coor_dtype}'."
                            ))
                            return None
                        if len(coor_shape) != 1:
                            errs.append(ValidationError(
                                level=ValidationLevel.ERROR,
                                type=ValidationType.INVALID_SHAPE,
                                path=concatenate_paths(path, f"Scanning/Spatial_map/{coor}"),
                                message=f"The '{coor}' array in the 'Spatial_map' group of the data group '{path}' must be 1-dimensional, found shape {coor_shape}."
                            ))
                            return None
                        return coor_shape[0]
                    else:
                        errs.append(ValidationError(
                            level=ValidationLevel.CRITICAL,
                            type=ValidationType.MISSING_ATTRIBUTE,
                            path=concatenate_paths(path, f"Scanning/Spatial_map/{coor}"),
                            message=f"The '{coor}' array in the 'Spatial_map' group of the data group '{path}' must have 'shape' and 'dtype' attributes."
                        ))
                        return None
                else:
                    return None
            x_len = _get_coord_len('x')
            y_len = _get_coord_len('y')
            z_len = _get_coord_len('z')
            coor_len = x_len or y_len or z_len
            if coor_len is None:
                errs.append(ValidationError(
                    level=ValidationLevel.CRITICAL,
                    type=ValidationType.MISSING_ARRAY,
                    path=concatenate_paths(path, "Scanning/Spatial_map"),
                    message=f"The 'Spatial_map' group in the data group '{path}' must contain at least one of the coordinate arrays 'x', 'y' or 'z'."
                ))
            is_valid_len = lambda coor: coor is None or coor == coor_len
            if not is_valid_len(x_len) or not is_valid_len(y_len) or not is_valid_len(z_len):
                errs.append(ValidationError(
                    level=ValidationLevel.CRITICAL,
                    type=ValidationType.INVALID_SHAPE,
                    path=concatenate_paths(path, "Scanning/Spatial_map"),
                    message=f"All the coordinate arrays in the 'Spatial_map' group of the data group '{path}' must have the same length. Found lengths x: {x_len}, y: {y_len}, z: {z_len}."
                ))
            if PSD_shape is not None and coor_len is not None:
                if PSD_shape[0] != coor_len:
                    errs.append(ValidationError(
                        level=ValidationLevel.CRITICAL,
                        type=ValidationType.INVALID_SHAPE,
                        path=concatenate_paths(path, "Scanning/Spatial_map"),
                        message=f"The length of the coordinate arrays in the 'Spatial_map' group of the data group '{path}' must match the size of the first dimension of the 'PSD' array. Found coordinate length: {coor_len}, 'PSD' shape: {PSD_shape}."
                    ))
        # Validate the Cartesian_visualisation array if it exists
        if "Cartesian_visualisation" in scanning_group:
            cart_vis_shape, cart_vis_dtype = get_array_shape_and_dtype(scanning_group["Cartesian_visualisation"])
            if cart_vis_shape is None or cart_vis_dtype is None:
                errs.append(ValidationError(
                    level=ValidationLevel.CRITICAL,
                    type=ValidationType.MISSING_ATTRIBUTE,
                    path=concatenate_paths(path, "Scanning/Cartesian_visualisation"),
                    message=f"The 'Cartesian_visualisation' array in the 'Scanning' group of the data group '{path}' must have 'shape' and 'dtype' attributes."
                ))
            elif not is_numeric_dtype(cart_vis_dtype):
                errs.append(ValidationError(
                    level=ValidationLevel.CRITICAL,
                    type=ValidationType.INVALID_TYPE,
                    path=concatenate_paths(path, "Scanning/Cartesian_visualisation"),
                    message=f"The 'Cartesian_visualisation' array in the 'Scanning' group of the data group '{path}' must have a numeric dtype, found '{cart_vis_dtype}'."
                ))
            elif len(cart_vis_shape) != 3:
                errs.append(ValidationError(
                    level=ValidationLevel.CRITICAL,
                    type=ValidationType.INVALID_SHAPE,
                    path=concatenate_paths(path, "Scanning/Cartesian_visualisation"),
                    message=f"The 'Cartesian_visualisation' array in the 'Scanning' group of the data group '{path}' must be 3-dimensional, found shape {cart_vis_shape}."
                ))
            elif PSD_shape is not None and cart_vis_shape is not None:
                if sparse and prod(cart_vis_shape) != PSD_shape[0]:
                    errs.append(ValidationError(
                        level=ValidationLevel.WARNING,
                        type=ValidationType.INVALID_SHAPE,
                        path=concatenate_paths(path, "Scanning/Cartesian_visualisation"),
                        message=f"The total number of elements in the 'Cartesian_visualisation' array (shape {cart_vis_shape}) is not matching the spatial positions of the 'PSD' array (shape {PSD_shape}). This is valid - e.g. when some spatial positions are missing (which is often the case for sparse data) - but a warning is issued nevertheless."
                    ))
            
    # Validate the Parameters array
    if PSD_shape is not None and \
        ((sparse and len(PSD_shape) > 2) or (not sparse and len(PSD_shape) > 4)):
        if "Parameters" not in node:
            errs.append(ValidationError(
                level=ValidationLevel.ERROR,
                type=ValidationType.MISSING_ARRAY,
                path=path,
                message=f"The data group '{path}' must have a 'Parameters' array when the 'PSD' array has more than 2 dimensions for sparse data or more than 4 dimensions for non-sparse data."
            ))
        else:
            Parameters_shape, Parameters_dtype = get_array_shape_and_dtype(node['Parameters'])
            if Parameters_shape is None or Parameters_dtype is None:
                errs.append(ValidationError(
                    level=ValidationLevel.CRITICAL,
                    type=ValidationType.MISSING_ATTRIBUTE,
                    path=concatenate_paths(path, 'Parameters'),
                    message=f"The 'Parameters' array in the data group '{path}' must have 'shape' and 'dtype' attributes."
                ))
            num_pars = len(PSD_shape) - (2 if sparse else 4)
            if Parameters_shape is not None and len(Parameters_shape) != num_pars+1:
                errs.append(ValidationError(
                    level=ValidationLevel.ERROR,
                    type=ValidationType.INVALID_SHAPE,
                    path=concatenate_paths(path, 'Parameters'),
                    message=f"The 'Parameters' array in the data group '{path}' must have {num_pars+1} dimensions, found shape {Parameters_shape}."
                ))
            if Parameters_shape is not None and Parameters_shape[-1] != num_pars:
                errs.append(ValidationError(
                    level=ValidationLevel.ERROR,
                    type=ValidationType.INVALID_SHAPE,
                    path=concatenate_paths(path, 'Parameters'),
                    message=f"The 'Parameters' array in the data group '{path}' must have {num_pars} elements in the last dimension, found {Parameters_shape[-1]} instead."
                ))

    errs.extend(_validate_data_group_metadata_overrides(
        node,
        path,
        sparse=sparse,
        PSD_shape=PSD_shape,
        version_rules=version_rules,
    ))
    
    # list the analysis groups in the current data group and validate them
    analysis_groups: list[tuple[str, int]] = []
    for key in node.keys():
        match = re.match(brim_obj_names.data.analysis_results + r"_(\d+)$", key)
        if match:
            analysis_groups.append((key, int(match.group(1))))
    # check that there is at least one analysis group
    if len(analysis_groups) == 0:
        errs.append(ValidationError(
            level=ValidationLevel.WARNING,
            type=ValidationType.MISSING_GROUP,
            path=path,
            message=f"No analysis group was found in {path}. The file is still valid but no image could be extracted from it."
        ))
    else:
        # validate each analysis group
        for dg_name, dg_index in analysis_groups:
            errs.extend(validate_analysis_group(node[dg_name], path=concatenate_paths(path, dg_name),
                                                sparse=sparse, PSD_shape=PSD_shape))

    if subtype == SubType.SinglePoint_VIPA_v0_1.value:
        has_required_feature = False
        if subtype_features is not None:
            has_required_feature = '2DArray_per_spectrum' in subtype_features
        # Validate deep subtype constraints even when declarations are incomplete.
        if has_required_feature or subtype_features is None:
            errs.extend(_validate_singlepoint_vipa_data_group(
                node,
                path,
                sparse=sparse,
                PSD_shape=PSD_shape,
            ))

    return errs

def validate_root_attrs(attrs: dict) -> list[ValidationError]:
    errs: list[ValidationError] = []
    path = 'Brillouin_data'
    # check the version attribute
    attr_name = 'brim_version'
    version = attrs.get(attr_name, None)
    if version is None:
        errs.append(ValidationError(
            level=ValidationLevel.ERROR,
            type=ValidationType.MISSING_ATTRIBUTE,
            path=generate_attr_path(path, attr_name),
            message=f"The root group must have a '{attr_name}' attribute."
        ))
    else:
        try:
            get_version_rules(version)
        except UnsupportedBrimVersionError:
            errs.append(ValidationError(
                level=ValidationLevel.ERROR,
                type=ValidationType.INVALID_VALUE,
                path=generate_attr_path(path, attr_name),
                message=(
                    f"Unsupported brim_version '{version}'. Supported versions are "
                    f"{list(get_supported_versions())}."
                )
            ))
    attr_name = 'Subtype'
    subtype = attrs.get(attr_name, None)
    if subtype is not None:
        if not isinstance(subtype, str):
            errs.append(ValidationError(
                level=ValidationLevel.ERROR,
                type=ValidationType.INVALID_TYPE,
                path=generate_attr_path(path, attr_name),
                message=f"The '{attr_name}' attribute must be a string, found {type(subtype).__name__}."
            ))
        try:
            # this will raise a ValueError if the subtype is not valid
            subtype = SubType(subtype) 
        except ValueError:
            errs.append(ValidationError(
                level=ValidationLevel.ERROR,
                type=ValidationType.INVALID_VALUE,
                path=generate_attr_path(path, attr_name),
                message=f"Unsupported subtype '{subtype}'. Supported subtypes are: {sorted([st.value for st in SubType])}."
            ))

        attr_name = 'Subtype_features'
        subtype_features = attrs.get(attr_name, None)
        if subtype_features is None:
            errs.append(ValidationError(
                level=ValidationLevel.WARNING,
                type=ValidationType.MISSING_ATTRIBUTE,
                path=generate_attr_path(path, attr_name),
                message=f"When 'Subtype' is specified, it is recommended to provide the '{attr_name}' attribute as well."
            ))
        else:
            if not isinstance(subtype_features, (list, tuple)):
                errs.append(ValidationError(
                    level=ValidationLevel.ERROR,
                    type=ValidationType.INVALID_TYPE,
                    path=generate_attr_path(path, attr_name),
                    message=f"The '{attr_name}' attribute must be a list (or tuple) of feature names, found {type(subtype_features).__name__}."
                ))
            if not all(isinstance(feature, str) for feature in subtype_features):
                errs.append(ValidationError(
                    level=ValidationLevel.ERROR,
                    type=ValidationType.INVALID_TYPE,
                    path=generate_attr_path(path, attr_name),
                    message=f"All entries in '{attr_name}' must be strings."
                ))
        if isinstance(subtype, SubType) and isinstance(subtype_features, (list, tuple)) and \
            all(isinstance(feature, str) for feature in subtype_features):
            subtype_enum = SubType(subtype)
            declared_features = set(subtype_features)
            required_features = {feature.name for feature in SUBTYPE_FEATURES[subtype_enum] if feature.required}
            optional_features = {feature.name for feature in SUBTYPE_FEATURES[subtype_enum] if not feature.required}
            allowed_features = required_features | optional_features

            missing_features = required_features - declared_features
            if missing_features:
                errs.append(ValidationError(
                    level=ValidationLevel.ERROR,
                    type=ValidationType.INVALID_VALUE,
                    path=generate_attr_path(path, attr_name),
                    message=f"Subtype '{subtype}' requires feature(s) {sorted(missing_features)} to be listed in '{attr_name}'."
                ))

            unknown_features = declared_features - allowed_features
            if unknown_features:
                errs.append(ValidationError(
                    level=ValidationLevel.WARNING,
                    type=ValidationType.INVALID_VALUE,
                    path=generate_attr_path(path, attr_name),
                    message=f"Unknown feature(s) {sorted(unknown_features)} listed in '{attr_name}' for subtype '{subtype}'."
                ))
    return errs

def validate_Brillouin_data_group(
    node: dict,
    *,
    version_rules: VersionRules,
    subtype: str | None = None,
    subtype_features: set[str] | None = None,
) -> list[ValidationError]:
    errs: list[ValidationError] = []
    path = 'Brillouin_data'
    node_type = get_node_type(node)
    if node_type != _NodeType.GROUP:
        errs.append(ValidationError(
            level=ValidationLevel.CRITICAL,
            type=ValidationType.INVALID_TYPE,
            path=path,
            message=f"The 'Brillouin_data' node must be a group, found '{node_type}'."
        ))
    attrs = get_attributes(node)
    if attrs is None:
        errs.append(ValidationError(
            level=ValidationLevel.CRITICAL,
            type=ValidationType.MISSING_ATTRIBUTE,
            path=path,
            message="The 'Brillouin_data' group must have attributes."
        ))
    else:
        # validate the general metadata
        if 'Metadata' not in attrs:
            errs.append(ValidationError(
                level=ValidationLevel.ERROR,
                type=ValidationType.MISSING_ATTRIBUTE,
                path=path,
                message="The 'Brillouin_data' group must contain a 'Metadata' attribute."
            ))
        else:
            metadata_path = generate_attr_path(path, 'Metadata')
            metadata = attrs['Metadata']
            if not isinstance(metadata, dict):
                errs.append(ValidationError(
                    level=ValidationLevel.ERROR,
                    type=ValidationType.INVALID_TYPE,
                    path=metadata_path,
                    message=f"The 'Metadata' attribute must be a dictionary, found {type(metadata).__name__}."
                ))
            else:
                for md_type in MetadataType:
                    if md_type.value in metadata:
                        md_dict = metadata[md_type.value]
                        if not isinstance(md_dict, dict):
                            errs.append(ValidationError(
                                level=ValidationLevel.ERROR,
                                type=ValidationType.INVALID_TYPE,
                                path=f"{metadata_path}.{md_type.value}",
                                message=f"The '{md_type.value}' field in 'Metadata' must be a dictionary, found {type(md_dict).__name__}."
                            ))
                        else:
                            errs.extend(validate_metadata(md_type, md_dict))
                    else:
                        errs.append(ValidationError(
                            level=ValidationLevel.ERROR,
                            type=ValidationType.MISSING_METADATA,
                            path=f"{metadata_path}.{md_type.value}",
                            message=f"The '{md_type.value}' field is missing in 'Metadata'."
                        ))
    # list the data groups in the Brillouin_data group and validate them
    data_groups: list[tuple[str, int]] = []
    for key in node.keys():
        match = re.match(brim_obj_names.data.base_group + r"_(\d+)$", key)
        if match:
            data_groups.append((key, int(match.group(1))))
    # check that there is at least one data group
    if len(data_groups) == 0:
        errs.append(ValidationError(
            level=ValidationLevel.CRITICAL,
            type=ValidationType.MISSING_GROUP,
            path=path,
            message="At least one data group is required in the 'Brillouin_data' group, but none were found."
        ))
    else:
        # validate each data group
        for dg_name, dg_index in data_groups:
            errs.extend(validate_data_group(
                node[dg_name],
                path=concatenate_paths(path, dg_name),
                version_rules=version_rules,
                subtype=subtype,
                subtype_features=subtype_features,
            ))

    return errs

def validate_json(json_descriptor: str) -> list[ValidationError]:
    """Validate a JSON descriptor against the expected structure of a brim file (https://github.com/brillouin-imaging/Brillouin-standard-file/blob/linkml-schema/docs/brim_file_specs.md).

    This function checks that the JSON descriptor contains all required fields
    and that they have the correct types. It raises a ValueError if any
    validation checks fail.

    Args:
        json_descriptor: A JSON string representing the Zarr hierarchy descriptor.
    """
    try:
        descriptor_dict = json.loads(json_descriptor)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}")

    # Perform validation checks on the descriptor_dict structure
    if not isinstance(descriptor_dict, dict):
        raise ValueError("Descriptor must be a JSON object at the top level.")
        
    errs: list[ValidationError] = []

    # check the root
    path = ''
    node_type = get_node_type(descriptor_dict)
    if node_type != _NodeType.GROUP:
        errs.append(ValidationError(
            level=ValidationLevel.CRITICAL,
            type=ValidationType.MISSING_GROUP,
            path=path,
            message=f"There must be a group at the root level, found '{node_type}'."
        ))
    subtype: str | None = None
    subtype_features: set[str] | None = None
    version_rules: VersionRules | None = None
    attrs = get_attributes(descriptor_dict)
    if attrs is None:
        errs.append(ValidationError(
            level=ValidationLevel.CRITICAL,
            type=ValidationType.MISSING_ATTRIBUTE,
            path=path,
            message="The root group must have attributes."
        ))
    else:
        errs.extend(validate_root_attrs(attrs))
        version = attrs.get('brim_version', None)
        if isinstance(version, str):
            try:
                version_rules = get_version_rules(version)
            except UnsupportedBrimVersionError:
                version_rules = None
        raw_subtype = attrs.get('Subtype', None)
        if isinstance(raw_subtype, str):
            subtype = raw_subtype
        raw_subtype_features = attrs.get('Subtype_features', None)
        if isinstance(raw_subtype_features, (list, tuple)) and all(isinstance(x, str) for x in raw_subtype_features):
            subtype_features = set(raw_subtype_features)
    
    # check the Brillouin_data group
    path = 'Brillouin_data'
    brillouin_data_group = descriptor_dict.get('Brillouin_data', None)
    if brillouin_data_group is not None and version_rules is not None:
        errs.extend(validate_Brillouin_data_group(
            brillouin_data_group,
            version_rules=version_rules,
            subtype=subtype,
            subtype_features=subtype_features,
        ))
    elif brillouin_data_group is not None and version_rules is None:
        errs.append(ValidationError(
            level=ValidationLevel.ERROR,
            type=ValidationType.INVALID_VALUE,
            path=generate_attr_path('', 'brim_version'),
            message=(
                f"Cannot validate '{path}' because brim_version is missing or unsupported. "
                f"Supported versions are {list(get_supported_versions())}."
            )
        ))
    else:
        errs.append(ValidationError(
            level=ValidationLevel.CRITICAL,
            type=ValidationType.MISSING_GROUP,
            path=path,
            message="The 'Brillouin_data' group is required but missing."
        ))

    return errs