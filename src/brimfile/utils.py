import re
import numpy as np
from typing import Any
from numpy.typing import NDArray

from .file_abstraction import FileAbstraction, sync

__docformat__ = "google"


def concatenate_paths(*paths):
    """
    Concatenate multiple paths into a single path.

    Args:
        *paths: Paths to concatenate.

    Returns:
        str: The concatenated path. The concatenated path will have a leading '/' but no trailing '/'.
    """
    con_path = ''
    for path in paths:
        if path.endswith('/'):
            path = path[:-1]
        if path.startswith('/'):
            path = path[1:]
        con_path += '/' + path
    return con_path


async def list_objects_matching_pattern_async(file: FileAbstraction, parent_obj, regexp: str) -> list:
    """
    Lists objects within a parent object that match a given regular expression pattern.
    Args:
        file (FileAbstraction): An abstraction representing the file system or storage 
            where objects are listed.
        parent_obj: The parent object containing the objects to be matched.
        regexp (str): A regular expression pattern to match object names.
    Returns:
        list: A list of tuples containing the matched object names and their corresponding capturing groups as string.
    """

    pattern = re.compile(regexp)
    # n_par = pattern.groups

    matched_objects = []
    for obj_name in await file.list_objects(parent_obj):
        match = pattern.match(obj_name)
        if match:
            matched_objects.append((obj_name,) + match.groups())
    return matched_objects


async def get_object_name(file: FileAbstraction, obj_path: str) -> str:
    """
    Returns the name of the object.

    The name is retrieved from the 'Name' attribute attached to the object.
    If the attribute is not found, the last part of the path is returned instead.
    """
    try:
        name = await file.get_attr(obj_path, 'Name')
        if name is None or name == '':
            raise Exception("Name attribute is None")
    except Exception as e:
        name = obj_path.split('/')[-1]
    return name


def set_object_name(file: FileAbstraction, obj, name: str):
    """
    Set the name of the object.

    The name is set by attaching a 'Name' attribute to the object.
    """
    sync(file.create_attr(obj, 'Name', name))

def var_to_singleton(var):
    """
    If `var` is not a list or a tuple, convert it to a single-ton (list with one element).

    Args:
        var: The variable to convert.

    Returns:
        list: A single-ton containing the variable.
    """    
    if (not isinstance(var, list)) and (not isinstance(var, tuple)):
        var = [var,]
    return var

def np_array_to_smallest_int_type(arr):
    """
    Convert a numpy array containing integers to the smallest integer type that can hold its values.

    Args:
        arr (numpy.ndarray): The input numpy array.

    Returns:
        numpy.ndarray: The converted array with the smallest integer type.
    """
    def type_bug_fix (dt):
        # explicitely pass an object of type np.dtype
        # as what is returned by `min_scalar_type` seems
        # to break type matching in the zarr library
        match dt:
            case np.int8:
                return np.int8
            case np.uint8:
                return np.uint8
            case np.int16:
                return np.int16
            case np.uint16:
                return np.uint16
            case np.int32:
                return np.int32
            case np.uint32:
                return np.uint32
            case np.int64:
                return np.int64
            case np.uint64:
                return np.uint64
            case _:
                return dt
        return dt
    if not isinstance(arr, np.ndarray):
        raise TypeError("Input must be a numpy array.")
    if not np.issubdtype(arr.dtype, np.integer):
        raise TypeError("Input array must contain integer values.")
    if np.any(arr < 0):
        # set mx to a negative number so that the smallest integer type is signed
        mx = -np.max(np.abs(arr)).astype(np.int64)
    else:
        mx = arr.max().astype(np.uint64)    
    # convert arr to the smallest integer type that can hold the values
    return arr.astype(type_bug_fix(np.min_scalar_type(mx)))

def _guess_chunks(
    shape: tuple,
    typesize: int,
    dset_size: int = None,
    *,
    increment_bytes: int = 256 * 1024,
    min_bytes: int = 128 * 1024,
    max_bytes: int = 64 * 1024 * 1024,
) -> tuple:
    """
    Iteratively guess an appropriate chunk layout for an array, given its shape and
    the size of each element in bytes, and size constraints expressed in bytes. 
    This function is based on https://github.com/zarr-developers/zarr-python/blob/71cc0c2e38ccd4ea1cfc2bd5f49dd66d4557484e/src/zarr/core/chunk_grids.py#L34

    Parameters
    ----------
    shape : tuple
        The chunk shape.
    typesize : int
        The size, in bytes, of each element of the chunk.
    dset_size : int
        The size, in bytes, of the whole dataset. If None, it is calculated as the product of the shape and typesize.
    increment_bytes : int = 256 * 1024
        The number of bytes used to increment or decrement the target chunk size in bytes.
    min_bytes : int = 128 * 1024
        The soft lower bound on the final chunk size in bytes.
    max_bytes : int = 64 * 1024 * 1024
        The hard upper bound on the final chunk size in bytes.

    Returns
    -------
    tuple

    """
    if isinstance(shape, int):
        shape = (shape,)

    if typesize == 0:
        return shape

    ndims = len(shape)
    # require chunks to have non-zero length for all dimensions
    chunks = np.maximum(np.array(shape, dtype="=f8"), 1)

    # Determine the optimal chunk size in bytes using a PyTables expression.
    # This is kept as a float.
    if dset_size is None:
        dset_size = np.prod(chunks) * typesize
    target_size = increment_bytes * (2 ** np.log10(dset_size / (1024.0 * 1024)))

    if target_size > max_bytes:
        target_size = max_bytes
    elif target_size < min_bytes:
        target_size = min_bytes

    idx = 0
    while True:
        # Repeatedly loop over the axes, dividing them by 2.  Stop when:
        # 1a. We're smaller than the target chunk size, OR
        # 1b. We're within 50% of the target chunk size, AND
        # 2. The chunk is smaller than the maximum chunk size

        chunk_bytes = np.prod(chunks) * typesize

        if (
            chunk_bytes < target_size or abs(chunk_bytes - target_size) / target_size < 0.5
        ) and chunk_bytes < max_bytes:
            break

        if np.prod(chunks) == 1:
            break  # Element size larger than max_bytes

        chunks[idx % ndims] = np.ceil(chunks[idx % ndims] / 2.0)
        idx += 1

    return tuple(int(x) for x in chunks)

def _determine_chunk_size(arr: NDArray[Any], n_unsplit_dims: int = 1) -> tuple:
    """
    Use the same heuristic as the zarr library to determine chunk sizes, while
    keeping the trailing ``n_unsplit_dims`` dimensions unchanged.
    """
    shape = arr.shape
    typesize = arr.itemsize

    if n_unsplit_dims < 0:
        raise ValueError("n_unsplit_dims must be >= 0")

    # If all dimensions are preserved (or there are no split dimensions), do not chunk.
    if len(shape) <= max(1, n_unsplit_dims):
        return tuple(shape)

    n_split_dims = len(shape) - n_unsplit_dims
    split_shape = shape[:n_split_dims]
    unsplit_shape = shape[n_split_dims:]

    unsplit_elements = int(np.prod(unsplit_shape)) if unsplit_shape else 1
    target_sizes = _guess_chunks.__kwdefaults__
    # Scale target chunk bytes based on preserved trailing dimensions.
    target_sizes = {
        k: max(1, target_sizes[k] // unsplit_elements)
        for k in target_sizes.keys()
    }

    chunks = _guess_chunks(split_shape, typesize, arr.nbytes, **target_sizes)
    return chunks + unsplit_shape
