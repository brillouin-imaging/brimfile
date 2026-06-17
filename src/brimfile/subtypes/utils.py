from ..file_abstraction import sync, FileAbstraction
from .constants import SubType

def get_subtype(f: FileAbstraction) -> SubType:
    """
    Get the subtype of the data group, returning SubType.none if no subtype is defined.
    """
    try:
        subtype_str = sync(f.get_attr('/', 'Subtype'))
        return SubType(subtype_str)
    except KeyError:
        return SubType.none
    except ValueError:
        raise ValueError(f"Invalid subtype: {subtype_str}. Expected one of {[s.value for s in SubType]}")

def _check_or_create_subtype(f: FileAbstraction, subtype: SubType):
    """
    Check that the data group subtype is 'subtype', creating it if missing.
    """
    try:
        # Check if the subtype already stored in the file is correct
        stored_subtype = sync(f.get_attr('/', 'Subtype'))
        if stored_subtype != subtype.value:
            raise ValueError(f"Invalid subtype: {stored_subtype}. Expected {subtype.value}")
    except KeyError:
        # If the Subtype attribute does not exist, create it
        sync(f.create_attr('/', 'Subtype', subtype.value))

def _check_or_create_subtype_feature(f: FileAbstraction, feature: str):
    """
    Check that the given feature is declared in the Subtype_features attribute, creating it if missing.
    """
    try:
        subtype_features = sync(f.get_attr('/', 'Subtype_features'))
        if not isinstance(subtype_features, (list, tuple)):
            raise ValueError(f"Invalid Subtype_features attribute: expected a list or tuple, found {type(subtype_features).__name__}")
        if feature not in subtype_features:
            subtype_features = list(subtype_features)
            subtype_features.append(feature)
            sync(f.create_attr('/', 'Subtype_features', subtype_features))
    except KeyError:
        # If the Subtype_features attribute does not exist, create it with the given feature
        sync(f.create_attr('/', 'Subtype_features', [feature]))