from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any

from .file_abstraction import FileAbstraction, sync


async def of_object(file: FileAbstraction, obj: Any) -> str:
    """
    Get the units of a given object in a file.

    Args:
        file (FileAbstraction)
        obj: The object for which to retrieve the units, as accepted by FileAbstraction.get_attr.

    Returns:
        str: The units of the object. If the units are not found, an empty string is returned.
    """
    try:
        units: str = await file.get_attr(obj, 'Units')
        return units
    except Exception:
        # If the attribute 'units' is not found, return an empty string
        return ''


def add_to_object(file: FileAbstraction, obj: Any, units: str) -> None:
    """
    Set the units of a given object in a file.

    Args:
        file (FileAbstraction)
        obj: The object for which to retrieve the units, as accepted by FileAbstraction.get_attr.
        units (str): The units to set for the object.
    """
    sync(file.create_attr(obj, 'Units', units))


async def of_attribute(file: FileAbstraction, obj: Any, attr_name: str) -> str | None:
    """
    Get the units of an attribute attached to a given object in a file.

    Args:
        file (FileAbstraction)
        obj: The object to which the attribute is attached, as accepted by FileAbstraction.get_attr.
        attr_name (str): The name of the attribute for which to retrieve the units.

    Returns:
        str | None: The units of the attribute. If the units are not found, None is returned.
    """
    try:
        units: str = await file.get_attr(obj, f"{attr_name}_units")
        return units
    except Exception:
        # If the attribute f"{attr_name}_units" is not found, return None
        return None


def add_to_attribute(file: FileAbstraction, obj: Any, attr_name: str, units: str) -> None:
    """
    Set the units of a given attribute in a file.

    Args:
        file (FileAbstraction)
        obj: The object for which to retrieve the units, as accepted by FileAbstraction.get_attr.
        units (str): The units to set for the object.
        attr_name (str): The name of the attribute for which to set the units.
    """
    sync(file.create_attr(obj, f"{attr_name}_units", units))
