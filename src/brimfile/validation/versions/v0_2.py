from __future__ import annotations

from .base import V0_1Rules


class V0_2Rules(V0_1Rules):
    version = "0.2"

    def uses_flattened_data_group_metadata_overrides(self) -> bool:
        return False

    def uses_nested_data_group_metadata_attribute(self) -> bool:
        return True

    def supports_data_group_metadata_arrays_group(self) -> bool:
        return True
