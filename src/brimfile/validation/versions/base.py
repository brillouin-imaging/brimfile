from __future__ import annotations


class VersionRules:
    """Version-specific validation behavior toggles.

    Shared validation logic remains in validation.main; this class only exposes
    behaviors that differ across BRIM spec versions.
    """

    version: str = "0.1"

    def uses_flattened_data_group_metadata_overrides(self) -> bool:
        """Whether /Data_{n} metadata overrides are stored as Type.Field attrs."""
        return True

    def uses_nested_data_group_metadata_attribute(self) -> bool:
        """Whether /Data_{n} uses a nested Metadata object attribute."""
        return False

    def supports_data_group_metadata_arrays_group(self) -> bool:
        """Whether /Data_{n}/Metadata per-position arrays are part of the spec."""
        return False


class V0_1Rules(VersionRules):
    version = "0.1"
