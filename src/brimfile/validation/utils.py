from __future__ import annotations

import re
from enum import StrEnum
from operator import index

_NUMERIC_DTYPE_ALIASES = {
    'int', 'uint', 'float', 'complex',
    'intp', 'uintp',
    'int8', 'int16', 'int32', 'int64',
    'uint8', 'uint16', 'uint32', 'uint64',
    'float16', 'float32', 'float64', 'float96', 'float128',
    'complex64', 'complex128', 'complex192', 'complex256',
    'byte', 'ubyte', 'short', 'ushort', 'longlong', 'ulonglong',
    'half', 'single', 'double', 'longdouble', 'longfloat',
    'csingle', 'cdouble', 'clongdouble',
}

class _NodeType(StrEnum):
    GROUP = 'group'
    ARRAY = 'array'

def broadcast_shapes(*shapes: object) -> tuple[int, ...]:
    """Broadcast multiple shapes using NumPy broadcasting rules."""
    if not shapes:
        return ()
    
    def _normalize_shape(shape: object, *, arg_index: int) -> tuple[int, ...]:
        """Validate and normalize a single shape argument."""
        try:
            dims = tuple(shape)
        except TypeError as exc:
            raise TypeError(
                f"shape argument {arg_index} must be an iterable of integers, "
                f"got {type(shape).__name__}"
            ) from exc

        normalized_dims: list[int] = []
        for dim in dims:
            try:
                normalized_dim = index(dim)
            except TypeError as exc:
                raise TypeError(
                    f"shape argument {arg_index} has non-integer dimension {dim!r}"
                ) from exc

            if normalized_dim < 0:
                raise ValueError(
                    f"shape argument {arg_index} has negative dimension {normalized_dim}"
                )

            normalized_dims.append(normalized_dim)

        return tuple(normalized_dims)

    normalized_shapes = [
        _normalize_shape(shape, arg_index=arg_index)
        for arg_index, shape in enumerate(shapes)
    ]

    max_ndim = max(len(shape) for shape in normalized_shapes)
    broadcasted_dims = [1] * max_ndim
    dim_owners: list[int | None] = [None] * max_ndim

    for arg_index, shape in enumerate(normalized_shapes):
        offset = max_ndim - len(shape)
        for axis_index, dim in enumerate(shape):
            axis = offset + axis_index
            current_dim = broadcasted_dims[axis]

            if current_dim == 1:
                broadcasted_dims[axis] = dim
                if dim != 1:
                    dim_owners[axis] = arg_index
                continue

            if dim == 1 or dim == current_dim:
                continue

            mismatch_arg_index = dim_owners[axis]
            if mismatch_arg_index is None:
                mismatch_arg_index = arg_index

            raise ValueError(
                "shape mismatch: objects cannot be broadcast to a single shape.  "
                f"Mismatch is between arg {mismatch_arg_index} "
                f"with shape {normalized_shapes[mismatch_arg_index]} "
                f"and arg {arg_index} with shape {shape}."
            )

    return tuple(broadcasted_dims)


def is_numeric_dtype(dtype: str) -> bool:
    """Return whether a dtype string represents a numeric dtype."""
    if not isinstance(dtype, str):
        return False

    normalized = dtype.strip().lower()
    if not normalized:
        return False

    if normalized.startswith("<class '") and normalized.endswith("'>"):
        normalized = normalized[8:-2]

    if normalized.startswith('numpy.'):
        normalized = normalized[6:]

    if normalized and normalized[0] in {'<', '>', '=', '|'}:
        normalized = normalized[1:]

    if normalized in _NUMERIC_DTYPE_ALIASES:
        return True

    if re.fullmatch(r'(?:u?int|float|complex)\d+', normalized):
        return True

    if re.fullmatch(r'[iufc]\d*', normalized):
        return True

    return False


def get_node_type(node: dict) -> _NodeType | None:
    return node.get('node_type', None)


def get_array_shape_and_dtype(node: dict) -> tuple[tuple[int, ...] | None, str | None]:
    return node.get('shape', None), node.get('dtype', None)


def get_attributes(node: dict) -> dict | None:
    return node.get('attributes', None)


def generate_attr_path(parent_path: str, attr_name: str) -> str:
    """
    Generate a path string for an attribute based on the parent path and the attribute name.
    Used to fill the 'path' field of ValidationError when the error is related to a specific attribute of a group or array.
    """
    return f"{parent_path}:{attr_name}"