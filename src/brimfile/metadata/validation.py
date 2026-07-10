from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar
if TYPE_CHECKING:
    from typing import Any

import warnings

from difflib import get_close_matches, SequenceMatcher
from enum import Enum
from numbers import Real
from .schema import Type, MetadataEnum, MetadataField, METADATA_SCHEMA
from .types import MetadataItem, MetadataValue, MetadataItemValidity

T = TypeVar('T')

def _normalize_token(value: str) -> str:
    """Normalize free-form tokens to a canonical snake-case key.

    This is used to match user inputs against enum aliases while tolerating
    differences in separators and casing.
    """
    return (
        value.strip()
        .lower()
        .replace(' ', '_')
        .replace('-', '_')
        .replace('&', '_and_')
    )


def _find_close_normalized_matches(
    value: str,
    candidates: dict[str, T],
    *,
    n: int,
    cutoff: float,
) -> tuple[str, list[tuple[str, T, float]]]:
    """Return normalized close matches and similarity scores.

    Args:
        value: Value to match (no normalization is applied before matching).
        candidates: Mapping from normalized token to candidate payload.
        n: Maximum number of close matches to return.
        cutoff: Similarity cutoff used by ``difflib.get_close_matches``.

    Returns:
        Tuple of normalized input and match entries as
        ``(normalized_candidate, payload, similarity_score)``.
    """
    close_tokens = get_close_matches(
        value,
        candidates.keys(),
        n=n,
        cutoff=cutoff,
    )
    scored_matches = [
        (
            token,
            candidates[token],
            SequenceMatcher(None, value, token).ratio(),
        )
        for token in close_tokens
    ]
    return value, scored_matches


def _coerce_enum(
    enum_type: type[MetadataEnum],
    value: MetadataEnum | Enum |str,
) -> MetadataEnum:
    """Convert input to an enum member using fuzzy matching with difflib.

    Args:
        enum_type: Target enum class.
        value: Input value provided by the user.

    Returns:
        A member of ``enum_type``.

    Raises:
        ValueError: If the input cannot be mapped to a valid enum member.
            The error message includes suggestions for close matches.
    """
    if isinstance(value, enum_type):
        return value
    if isinstance(value, Enum):
        value = str(value.value)
    
    if not isinstance(value, str):
        raise TypeError(
            f"Expected enum value to be a string or Enum of strings, got {type(value).__name__}"
        )
    
    # Try exact match with enum member values first
    for member in enum_type:
        if value == member.value:
            return member
    
    # Try exact match with enum member names
    try:
        return enum_type[value]
    except KeyError:
        pass
    
    # Build a list of all possible matches (member names and values) with normalized keys
    candidates: dict[str, MetadataEnum] = {}
    for member in enum_type:
        # Add normalized member name
        normalized_name = _normalize_token(member.name)
        candidates[normalized_name] = member
        # Add normalized member value
        normalized_value = _normalize_token(str(member.value))
        candidates[normalized_value] = member

    normalized_input = _normalize_token(value)
    
    # Try exact match with normalized tokens
    if normalized_input in candidates:
        return candidates[normalized_input]
    
    # Try fuzzy matching using difflib
    normalized_input, close_matches = _find_close_normalized_matches(
        normalized_input,
        candidates,
        n=3,
        cutoff=0.4,
    )
    if close_matches:
        best_match, _, best_score = close_matches[0]
        if best_score >= 0.8:
            warnings.warn(
                f"Interpreting '{value}' as '{best_match}' for {enum_type.__name__} (similarity: {best_score:.2f})."
            )
            return candidates[best_match]
        else:
            raise ValueError(
                f"Value '{value}' is not a valid {enum_type.__name__}. "
                f"Did you mean '{[sugg for sugg, _, _ in close_matches]}'? "
            )
    else:
        raise ValueError(
            f"Value '{value}' is not a valid {enum_type.__name__}, and no close matches were found."
        )

def _coerce_primitive(expected_type: type, value: Any) -> Any:
    """Coerce and validate primitive values for metadata fields.

    Float fields accept any real number and are converted to ``float``.
    String fields require a ``str`` value.
    """
    if expected_type is float:
        if not isinstance(value, Real):
            raise TypeError(f"Expected float-like value, got {type(value).__name__}")
        return float(value)
    if expected_type is str:
        if not isinstance(value, str):
            raise TypeError(f"Expected string value, got {type(value).__name__}")
        return value
    if not isinstance(value, expected_type):
        raise TypeError(f"Expected {expected_type.__name__}, got {type(value).__name__}")
    return value


def validate_single_field(
    metadata_type: Type,
    field_name: str,
    value: MetadataItem, *,
    report_on_invalid: bool = False
) -> tuple[str, MetadataItem]:
    """Validate and normalize a single metadata field value.

    Args:
        metadata_type: Section to validate e.g. Experiment, Optics, etc... (``Type`` or string equivalent).
        field_name: Name of the field to validate.
        value: Input value for the field.
        report_on_invalid: Whether to raise an exception or print a warning (depending on the severity) if the field is invalid.
    Returns:
        The canonical field name as a string, and the coerced value as a MetadataItem. The coerced value contains the validity field as well, which indicates whether the value is valid or not.
        The original field name and value are returned if they are valid or the field name is not recognized but there are no close matches.
    """
    validity : MetadataItemValidity = MetadataItemValidity.VALID

    # Get the metadata field from the schema if it exists
    field: MetadataField = next((f for f in METADATA_SCHEMA[metadata_type] if _normalize_token(f.name) == _normalize_token(field_name)), None)
    if field is not None:
        if field.name != field_name:
            validity = MetadataItemValidity.LIKELY_TYPO
            if report_on_invalid:
                # if the name is in the schema but it is not normalized, raise a warning to inform the user that we are normalizing the field name to the canonical name in the schema
                warnings.warn(
                    f"Field name '{field_name}' normalized to '{field.name}' for metadata type '{metadata_type.value}'."
                )
        field_name = field.name  # use the canonical field name from the schema
    else:
        # the provided field_name is not in the schema, let's try to find close matches and suggest them in a warning
        available_field_names = [f.name for f in METADATA_SCHEMA[metadata_type]]
        normalized_to_name = {
            _normalize_token(name): name
            for name in available_field_names
        }
        _, close_matches = _find_close_normalized_matches(
            _normalize_token(field_name),
            normalized_to_name,
            n=3,
            cutoff=0.6,
        )
        suggested_names = [name for _, name, _ in close_matches]
        if suggested_names:
            validity = MetadataItemValidity.LIKELY_TYPO
            if report_on_invalid:
                # if the name is not in the schema but there are close matches, raise an error with suggestions
                # (it is likely that the user made a typo and we don't want to silently accept an unknown field name in this case)
                raise ValueError(
                    f"Unknown field '{field_name}' for metadata type '{metadata_type.value}'. "
                    f"Did you mean {suggested_names}? "
                )
        else:
            validity = MetadataItemValidity.UNKNOWN_FIELD
            if report_on_invalid:
                # if the name is not in the schema and there are no close matches, we can accept it 
                # as it is likely that the user is trying to add a custom field that is not in the schema, but we will raise a warning to make sure they are aware that the field name is not recognized
                warnings.warn(
                    f"Unknown field '{field_name}' for metadata type '{metadata_type.value}'."
                    f" Note that '{field_name}' was added to the metadata of the file nevertheless."
                    )
    
    if field is not None and field.units_required and value.units is None:
        validity = MetadataItemValidity.MISSING_UNITS
        if report_on_invalid:
            raise ValueError(
                f"Metadata attribute {metadata_type.value}.{field.name} requires units."
            )
    coerced_value: MetadataValue = value.value
    if field is not None:
        if field.enum_type is not None:
            try:
                coerced_value = _coerce_enum(field.enum_type, value.value)
            except (ValueError, TypeError) as e:
                validity = MetadataItemValidity.INVALID_VALUE
                if report_on_invalid:
                    raise ValueError(str(e)) from e
        else:
            try:
                coerced_value = _coerce_primitive(field.python_type, value.value)
            except TypeError as e:
                validity = MetadataItemValidity.INVALID_VALUE
                if report_on_invalid:
                    raise ValueError(str(e)) from e

    return field_name, MetadataItem(coerced_value, value.units, validity=validity)