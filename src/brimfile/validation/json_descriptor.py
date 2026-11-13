from __future__ import annotations

import sys
from typing import TYPE_CHECKING
from ..file_abstraction import sync, _AbstractFile
from .utils import _Node

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import Any, TypeAlias
    _ZarrGroup: TypeAlias = 'zarr.Group | zarr.AsyncGroup'
    _ZarrNode: TypeAlias = zarr.Array[Any] | zarr.AsyncArray[Any] | zarr.Group | zarr.AsyncGroup | None

# N.B. `TYPE_CHECKING` is always False at runtime, so the condition below is equivalent to
# `"pyodide" not in sys.modules`. It only tells static type checkers to analyze the zarr-based implementation.
if TYPE_CHECKING or "pyodide" not in sys.modules:
    import zarr
    import json

    def list_keys(group: _ZarrGroup) -> Iterable[str]:
        if isinstance(group, zarr.AsyncGroup):
            async def list_keys_async() -> list[str]:
                return [key async for key in group.keys()]
            return sync(list_keys_async())
        else:
            return group.keys()
        
    def get_child(
            group: _ZarrGroup,
            key: str
            ) -> _ZarrNode:
        if isinstance(group, zarr.AsyncGroup):
            async def get_child_async() -> zarr.AsyncArray[Any] | zarr.AsyncGroup | None:
                return await group.get(key)
            return sync(get_child_async())
        else:
            return group[key]

    def recurse_nodes(node: _ZarrNode) -> _Node:
        node_dict: _Node = _Node({})
        if node is None:
            return node_dict
        attrs = node.attrs
        if hasattr(attrs, 'asdict'):
            attrs = attrs.asdict()
        node_dict['attributes'] = attrs
        if isinstance(node, zarr.Group) or isinstance(node, zarr.AsyncGroup):
            node_dict['node_type'] = 'group'
            # Recursively process children
            for key in list_keys(node):
                child_dict = recurse_nodes(get_child(node, key))
                node_dict[key] = child_dict
        else:
            node_dict['node_type'] = 'array'
            node_dict['shape'] = node.shape
            node_dict['dtype'] = str(node.dtype)        
        
        return node_dict

    def generate_json_descriptor(file: _AbstractFile) -> str:
        """Generate a JSON descriptor for a Zarr hierarchy.

        The descriptor recursively captures each node's attributes and type.
        Group nodes include their children, while array nodes include shape and
        data type information.

        Args:
            file: The _AbstractFile instance containing the Zarr hierarchy.

        Returns:
            A pretty-printed JSON string describing the full hierarchy.
        """

        root = file._root
        out_dict = recurse_nodes(root)

        json_output = json.dumps(out_dict, indent=4)
        return json_output
else:
    def generate_json_descriptor(file: _AbstractFile) -> str:
        """Generate a JSON descriptor for a Zarr hierarchy.

        The descriptor recursively captures each node's attributes and type.
        Group nodes include their children, while array nodes include shape and
        data type information.

        Args:
            file: The _AbstractFile instance containing the Zarr hierarchy.

        Returns:
            A pretty-printed JSON string describing the full hierarchy.
        """

        return sync(file._zarr_js.generateJsonDescriptor())
    