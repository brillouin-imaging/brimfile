from __future__ import annotations

from functools import lru_cache
import importlib

from .base import VersionRules


class UnsupportedBrimVersionError(ValueError):
    """Raised when no validation rules are registered for a brim_version."""


VERSION_RULES_REGISTRY: dict[str, str] = {
    "0.1": "brimfile.validation.versions.base:V0_1Rules",
    "0.2": "brimfile.validation.versions.v0_2:V0_2Rules",
}


def get_supported_versions() -> tuple[str, ...]:
    return tuple(sorted(VERSION_RULES_REGISTRY.keys()))


@lru_cache(maxsize=None)
def get_version_rules(version: str) -> VersionRules:
    target = VERSION_RULES_REGISTRY.get(version)
    if target is None:
        supported = ", ".join(get_supported_versions())
        raise UnsupportedBrimVersionError(
            f"Unsupported brim_version '{version}'. Supported versions are: [{supported}]."
        )

    module_path, class_name = target.split(":", maxsplit=1)
    module = importlib.import_module(module_path)
    rules_cls = getattr(module, class_name)
    rules = rules_cls()
    if not isinstance(rules, VersionRules):
        raise TypeError(
            f"Registered rules class '{class_name}' for version '{version}' is not a VersionRules subclass."
        )
    return rules
