from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import TypeAlias


JSONScalar: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"] | Enum
MetadataValue: TypeAlias = JSONValue

class MetadataItemValidity(Enum):
    """Validation status assigned to a `MetadataItem`."""

    NOT_CHECKED = 'not checked' # Validity has not been checked for this item
    VALID = 'valid'
    LIKELY_TYPO = 'likely typo' # The value is not valid but there are close matches that suggest it might be a typo
    UNKNOWN_FIELD = 'unknown field' # The field name is not recognized in the schema
    MISSING_FIELD = 'missing field' # The field is required but missing in the input metadata
    MISSING_UNITS = 'missing units' # The field requires units but they are missing
    INVALID_TYPE = 'invalid type' # The value has an invalid type
    INVALID_VALUE = 'invalid value' # The value is not valid for the field

class MetadataItem:
    """
    Container for one metadata value, optional units, and validation status.

    `value` stores the metadata payload, while `units` stores its units string
    when applicable (for example `'GHz'`, `'nm'`, or `'C'`).
    """

    # units should be a str. If None, no units is defined
    def __init__(self, value: MetadataValue , units: str | None = None, *,
                 validity: MetadataItemValidity | None = None):
        """Initialize a metadata item.
        value: the value of the metadata item. It can be any JSON-serializable value or an Enum.
        units: the units of the metadata item. It should be a string representing the units (e.g. 'nm', 'GHz', 'C', etc.). If None, no units is defined for this metadata item.
        validity: the validity of the metadata item. If no validity check was done, it can be set to None (default).
        """
        self.value = value
        self.units = units
        if validity is not None:
            self.validity = validity

    def get_validity(self) -> MetadataItemValidity:
        """
        Return the current validity flag.

        Returns:
            MetadataItemValidity: Stored validity if available, otherwise
                `MetadataItemValidity.NOT_CHECKED`.
        """
        if hasattr(self, 'validity'):
            return self.validity 
        else:
            return MetadataItemValidity.NOT_CHECKED