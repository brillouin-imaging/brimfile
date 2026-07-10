from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from enum import Enum
import numpy as np  
import asyncio

from typing import TypeAlias
Version: TypeAlias = tuple[int, int, int] | None


class FileAbstraction(ABC):
    """
    Abstract base class that rapresents a general interface to work with brim files.

    This class defines a common interface for file operations, such as creating attributes,
    retrieving attributes, and managing groups and datasets. It is designed to be extended
    by specific file implementations, such as HDF5 or Zarr.

    All the methods which require a path to an exixsting object in the file, will accept
    either the object itself (as defined by the specific implementation) or its path as a string.
    """

    # -------------------- Public attributes --------------------
    version: Version = None

    # -------------------- Attribute Management --------------------

    @abstractmethod
    async def create_attr(self, obj, name: str, data, **kwargs):
        """
        Create an attribute in the file.

        Args:
            obj: object that supports the creation of an attribute (e.g. group or dataset) or its path as a string.
            name (str): Name of the attribute.
            data: Data for the attribute.
            **kwargs: Additional arguments for attribute creation.
        """
        pass

    @abstractmethod
    async def get_attr(self, obj, name: str):
        """
        Return the data of an attribute in the file.

        Args:
            obj: object that supports the creation of an attribute (e.g. group or dataset) or its path as a string.
            name (str): Name of the attribute.
        Raises:
            KeyError: If the attribute does not exist.
        """
        pass

    # -------------------- Group Management --------------------

    @abstractmethod
    async def open_group(self, full_path: str, **kwargs):
        """
        Open a group in the file.

        Args:
            full_path (str): Path to the group.
            **kwargs: Additional arguments for opening the group.
        """
        pass

    @abstractmethod
    async def create_group(self, full_path: str, **kwargs):
        """
        Create a group in the file.

        Args:
            full_path (str): Path to the group.
            **kwargs: Additional arguments for creating the group.
        """
        pass

    # -------------------- Dataset Management --------------------

    class Compression:
        """
        Compression options for datasets.
        """
        NONE = None
        DEFAULT = 1
        ZLIB = 2
        LZF = 3

        def __init__(self, type=DEFAULT, level=None):
            self.type = type
            self.level = level

    @abstractmethod
    async def open_dataset(self, full_path: str):
        """
        Open a dataset in the file.

        Args:
            full_path (str): Path to the dataset.

        Returns:
            Dataset object which must support numpy indexing and slicing.
        """
        pass

    @abstractmethod
    async def create_dataset(self, parent_group, name: str, data, chunk_size=None, compression: 'FileAbstraction.Compression' = None):
        """
        Create a dataset in the file.

        Args:
            parent_group: Group in which to create the dataset or its path as a string.
            name (str): Name of the dataset.
            data: Data for the dataset.
            chunk_size (tuple, optional): Chunk size for the dataset. If None the automatically computed size will be used.
            compression (FileAbstraction.Compression, optional): Compression options for the dataset.
        """
        pass

    # -------------------- Listing --------------------

    @abstractmethod
    async def list_objects(self, obj) -> list:
        """
        Lists the objects (groups or datasets) contained within one hierarchical level below the given object.

        Args:
            obj: parent object or its path as a string.

        Returns:
            list: List of strings representing the names of the objects.
        """
        pass

    @abstractmethod
    async def object_exists(self, full_path) -> bool:
        """
        Check if an object exists in the file.

        Args:
            full_path (str): Path to the object.

        Returns:
            bool: True if the object exists, False otherwise.
        """
        pass

    @abstractmethod
    async def list_attributes(self, obj) -> list:
        """
        Lists the attributes attached to the specified object.

        Args:
            obj: object or its path as a string.

        Returns:
            list: List of strings representing the names of the attributes.
        """
        pass

    # -------------------- File Management --------------------

    def close(self):
        """
        Close the file.
        """
        pass

    # -------------------- Properties --------------------

    async def is_read_only(self) -> bool:
        """
        Check if the file is read-only.

        Returns:
            bool: True if the file is read-only, False otherwise.
        """
        return True

class StoreType(Enum):
    """
    Enum to represent the type of store used by the Zarr file.
    """
    ZIP = 'zip'
    """We recommend using zip only for reading files. Writing will work, but at the cost of duplicating entries
    inside the archive (see [GitHub issue](https://github.com/zarr-developers/zarr-python/issues/1695)).
    Consider using zarr store instead and zipping it at the end of writing."""
    ZARR = 'zarr'
    S3 = 'S3'
    AUTO = 'auto' 
    """Automatically determine the store type based on the filename (i.e. extension or url schema)"""

async def _async_getitem(obj, indices: tuple):
        """
        Asynchronously get a slice of an object that supports indexing and slicing.
        
        Args:
            obj: Object that supports indexing and slicing (e.g., zarr.AsyncArray).
            indices (tuple): Tuple of indices or slices to retrieve.
        Returns:
            The sliced data from the object.
        
        N.B. this function is a quick workaround to transition from the sync to async paradigm.
             Consider rethinking the whole structure it in the future!
        """
        if isinstance(indices, list):
            indices = tuple(indices)
        elif not isinstance(indices, tuple):
            indices = (indices,)

        if isinstance(obj, _ZarrAsyncArray):
            #N.B. it is important to check first if obj is a _ZarrAsyncArray,
            # since in pyodide there might be no distinction between zarr.Array and zarr.AsyncArray
            # and the async call should have priority
            return await obj.getitem(indices)
        elif isinstance(obj, np.ndarray) or isinstance(obj,  _ZarrArray):
            return obj[indices]        
        else:
            raise ValueError(f"Object of type '{type(obj)}' does not support indexing and slicing.")

def _gather_sync(*aws, return_exceptions: bool = False):
    """
    Sync version of asyncio.gather.
    Args: same as asyncio.gather
    """
    async def _f():
        return await asyncio.gather(*aws, return_exceptions=return_exceptions)
    return sync(_f())

import sys
if "pyodide" in sys.modules:  # using javascript based zarr library
    import js
    
    async def _awaitable_wrapper(coro):
        return await coro
    def sync(coro):
        """"
        Synchronously run an asynchronous coroutine.
        """
        loop = asyncio.get_event_loop()
        task = loop.create_task(_awaitable_wrapper(coro))
        # In Pyodide, we can't block the event loop, so instead we yield back
        # control until the task is done.
        while not task.done():
            loop.run_until_complete(asyncio.sleep(0))
        return task.result()

    class _zarrFile:
        class ZarrArray:
            def __init__(self, zarr_js, dts):
                self._zarr_js = zarr_js
                self.dts = dts            
            def __str__(self):
                return str(self.dts)
            def __array__(self, dtype=None, copy=None):
                #TODO: implement dtype and copy
                # see https://numpy.org/doc/stable/user/basics.interoperability.html#dunder-array-interface
                return self[...]
            
            async def getitem(self, index):
                def index_to_js_slice(i):
                    if isinstance(i, slice):
                        return [i.start, i.stop]
                    elif isinstance(i, type(Ellipsis)):
                        raise ValueError("INTERNAL: Ellipsis should have been already substituted")
                    else:
                        return [i, i+1]
                if type(index) is not tuple:
                    index = (index,)
                # check that only one Ellipsis is present
                num_ellipsis = sum(isinstance(i, type(Ellipsis)) for i in index)
                if num_ellipsis>1:
                    raise ValueError("Only one Ellipsis is allowed!")
                if num_ellipsis == 1:
                    n_dim = len(self.shape)
                    if len(index)-1>n_dim:
                        raise ValueError(f"Expected at most {n_dim} indices and got {len(index)} instead!")
                    n_null_indices = 1 + n_dim-len(index)
                    new_index = ()
                    for i in index:
                        if isinstance(i, type(Ellipsis)):
                            new_index += (slice(None),)*n_null_indices
                        else:
                            new_index += (i,)
                    index = new_index     
                js_indices = []
                for i in index:
                    js_indices.append(index_to_js_slice(i))
                
                res = await self._zarr_js.get_array_slice(str(self.dts), js_indices)
                data = _zarrFile.JsProxy_to_py(res.data)
                shape = _zarrFile.JsProxy_to_py(res.shape)
                data = np.array(data)
                data = np.reshape(data, shape)
                
                # remove singleton dimensions
                singleton_dims = ()
                for i, ind in enumerate(index):
                    if isinstance(ind, int):
                        singleton_dims += (i,)
                if len(singleton_dims)>0:
                    data = np.squeeze(data, axis=singleton_dims)
                return data            
            def __getitem__(self, index):
                return sync(self.getitem(index))
            
            @property
            def shape(self):
                if hasattr(self, '_shape'):
                    return self._shape
                else:
                    res = sync(self._zarr_js.get_array_shape(str(self.dts)))
                    self._shape = _zarrFile.JsProxy_to_py(res)
                    return self._shape
            @property
            def size(self):
                return np.prod(self.shape)

            @property
            def ndim(self):
                return len(self.shape)
            

        def __init__(self, zarr_js, filename:str,  *, version: Version = None):
            self._zarr_js = zarr_js
            self.filename = filename
            self.version = version

        @staticmethod
        def JsProxy_to_py(jsproxy):
            #alternatively one can use isinstance(jsproxy, pyodide.ffi.JsProxy)
            if hasattr(jsproxy, "to_py"):
                return jsproxy.to_py()
            return jsproxy
        
        # -------------------- Attribute Management --------------------
        async def get_attr(self, full_path, attr_name):  
            res = await self._zarr_js.get_attribute(str(full_path), str(attr_name))
            return _zarrFile.JsProxy_to_py(res)
        
        # -------------------- Group Management --------------------
        async def open_group(self, full_path: str):
            res = await self._zarr_js.open_group(str(full_path))
            return res
        
        # -------------------- Dataset Management --------------------
        async def open_dataset(self, full_path: str):
            dts = await self._zarr_js.open_dataset(str(full_path))
            return _zarrFile.ZarrArray(self._zarr_js, dts)
        
        # -------------------- Listing --------------------
        async def list_objects(self, full_path) -> list:
            res = await self._zarr_js.list_objects(str(full_path))
            return _zarrFile.JsProxy_to_py(res)
        async def object_exists(self, full_path) -> bool:
            res = await self._zarr_js.object_exists(str(full_path))
            return _zarrFile.JsProxy_to_py(res)
        async def list_attributes(self, full_path) -> list:
            res = await self._zarr_js.list_attributes(str(full_path))
            return _zarrFile.JsProxy_to_py(res)
        
         # -------------------- File Management --------------------

        def close(self):
            pass

        # -------------------- Properties --------------------
        async def is_read_only(self) -> bool:
            return True
    
    # used by _async_getitem
    _ZarrAsyncArray = _zarrFile.ZarrArray
    _ZarrArray = _zarrFile.ZarrArray
else:
    import zarr
    import numcodecs

    import importlib.util

    import zarr.api.asynchronous as zarr_async
    def sync(coro):
        """"
        Synchronously run an asynchronous coroutine.
        """
        res =  zarr.core.sync.sync(coro)
        return res
    
    # used by _async_getitem
    _ZarrAsyncArray = zarr.AsyncArray
    _ZarrArray = zarr.Array
    
    def _parse_storage_url(url):
        from urllib.parse import urlparse

        parsed = urlparse(url)
        scheme = parsed.scheme
        netloc = parsed.netloc
        path = parsed.path.lstrip('/')

        # Case 1: Amazon S3 (virtual-hosted-style or path-style)
        if netloc == "amazonaws.com" or netloc.endswith(".amazonaws.com"):
            parts = netloc.split('.')
            if parts[0] != 's3':  # virtual-hosted-style
                bucket = parts[0]
                endpoint = '.'.join(parts[1:])
                object_path = path
            else:  # path-style
                path_parts = path.split('/', 1)
                bucket = path_parts[0]
                endpoint = netloc
                object_path = path_parts[1] if len(path_parts) > 1 else ''
        # Case 2: Google Cloud Storage
        elif netloc == "storage.googleapis.com" or netloc.endswith(".storage.googleapis.com"):
            if netloc == "storage.googleapis.com":
                # path-style: https://storage.googleapis.com/bucket-name/object
                path_parts = path.split('/', 1)
                bucket = path_parts[0]
                endpoint = netloc
                object_path = path_parts[1] if len(path_parts) > 1 else ''
            else:
                # virtual-hosted-style: https://bucket-name.storage.googleapis.com/object
                bucket = netloc.split('.')[0]
                endpoint = '.'.join(netloc.split('.')[1:])
                object_path = path
        # Case 3: Custom endpoint or S3-compatible storage (MinIO, etc.)
        else:
            path_parts = path.split('/', 1)
            bucket = path_parts[0]
            endpoint = netloc
            object_path = path_parts[1] if len(path_parts) > 1 else ''

        return {
            'protocol': scheme,
            'bucket': bucket,
            'endpoint': endpoint,
            'object_path': object_path
        }
    

    class _zarrFile (FileAbstraction):
        def __init__(self, filename: str, mode: str = 'r',
                     store_type: StoreType = StoreType.AUTO, *, version: Version = None):
            """
            Initialize the Zarr file.

            Args:
                filename (str): Path to the Zarr file.
                mode: {'r', 'r+', 'a', 'w', 'w-'} the mode for opening the file (default is 'r' for read-only).
                        'r' means read only (must exist); 'r+' means read/write (must exist);
                        'a' means read/write (create if doesn't exist); 'w' means create (overwrite if exists); 'w-' means create (fail if exists).
                store_type (str): Type of the store to use. Default is 'AUTO'.
                version (Version): Version of the file format to use. Default is None.
            """
            st = StoreType

            if store_type == st.ZIP:
                if not filename.endswith('.zip'):
                    filename += '.zip'
            elif store_type == st.ZARR:
                if not filename.endswith('.zarr'):
                    filename += '.zarr'
            elif store_type == st.AUTO:
                if filename.startswith('http') or filename.startswith('s3'):
                    store_type = st.S3
                elif filename.endswith('.zip'):
                    store_type = st.ZIP
                elif filename.endswith('.zarr'):
                    store_type = st.ZARR
                else:
                    raise ValueError(
                        "When using 'auto' store_type, the filename must end with '.zip' or '.zarr' or start with 'http' or 's3'.")

            if mode not in ['r', 'r+', 'a', 'w', 'w-']:
                raise ValueError(
                    f"Invalid mode '{mode}'. Supported modes are 'r', 'r+', 'a', 'w', and 'w-'.")

            match store_type:
                case st.ZIP:
                    mode_zip = mode
                    if mode == 'w-':
                        mode_zip = 'x'
                    elif mode == 'r+':
                        mode_zip = 'a'
                    store = zarr.storage.ZipStore(filename, mode=mode_zip)
                case st.ZARR:
                    store = zarr.storage.LocalStore(filename)
                case st.S3:
                    if importlib.util.find_spec('fsspec') is None:
                        raise ModuleNotFoundError(
                            "The fsspec module is required for using S3 storage")
                    import fsspec
                    parsed_url = _parse_storage_url(filename)                           

                    fs = fsspec.filesystem('s3', anon=True, asynchronous=True,
                                        client_kwargs={'endpoint_url': f"{parsed_url['protocol']}://{parsed_url['endpoint']}"})

                    store = zarr.storage.FsspecStore(fs, path = f"{parsed_url['bucket']}/{parsed_url['object_path']}",
                                                    read_only=(mode == 'r'))
                case _:
                    raise ValueError(
                        f"Unsupported store type '{store_type}'. Supported types are 'zip', 'zarr', and 'remote'.")
            self._root = sync(zarr_async.open_group(store=store, mode=mode))
            self._store = store
            self.filename = filename
            self.version = version

        # -------------------- Attribute Management --------------------

        async def create_attr(self, obj, name: str, data, **kwargs):
            for k in kwargs.keys():
                warnings.warn(
                    f"'{k}' argument not supported by 'create_attr' in zarr")
            if isinstance(obj, str):
                obj = await self._root.getitem(obj)
            """
            if isinstance(obj, zarr.AsyncGroup):
                obj = zarr.Group(obj)
            elif isinstance(obj, zarr.AsyncArray):
                obj = zarr.Array(obj)
            """
            attrs = obj.attrs
            attrs[name] = data
            await obj.update_attributes(attrs) 

        async def get_attr(self, obj, name: str):
            if isinstance(obj, str):
                obj = await self._root.getitem(obj)
            return obj.attrs[name]

        # -------------------- Group Management --------------------

        async def open_group(self, full_path: str, **kwargs):
            for k in kwargs.keys():
                warnings.warn(
                    f"'{k}' argument not supported by 'open_group' in zarr")
            g = await self._root.getitem(full_path)
            return g

        async def create_group(self, full_path: str):
            g = await self._root.create_group(full_path)
            return g

        # -------------------- Dataset Management --------------------

        @staticmethod
        def _to_ZarrArray(obj: zarr.AsyncArray):
            """"
            Add attributes to Zarr.AsyncArray object to support numpy indexing and slicing.
            
            N.B. this is a temporary fix to make existing code compatible with zarr.AsyncArray
                 Don't add any new functionality here and consider changing it in the future!
            """ 
            class _ZarrArray(zarr.AsyncArray):
                def __array__(self, dtype=None, copy=None):
                # N.B. this is calling `sync` internally, so it shouldn't be used in async functions!!
                    return zarr.Array(self).__array__(dtype=dtype, copy=copy)
                async def to_np_array(self, dtype=None, copy=None):
                # same as __array__ but using async code
                    return np.array(await self.getitem(...))
                def __getitem__(self, index):
                # N.B. this is calling `sync` internally, so it shouldn't be used in async functions!!
                    return zarr.Array(self).__getitem__(index)
            # since @dataclass(frozen=True), we need to use object.__setattr__
            object.__setattr__(obj, '__class__', _ZarrArray)
            return obj

        async def open_dataset(self, full_path: str):
            ds = await self._root.getitem(full_path)
            # "upgrade" the object to a _ZarrArray
            ds=_zarrFile._to_ZarrArray(ds)
            return ds

        async def create_dataset(self, parent_group, name: str, data, chunk_size=None, compression: 'FileAbstraction.Compression' = FileAbstraction.Compression()):
            if isinstance(parent_group, str):
                parent_group = await self.open_group(parent_group)
            compressor = None
            if chunk_size is None:
                chunk_size = 'auto'
            if compression is not None:
                if compression.type == FileAbstraction.Compression.DEFAULT:
                    # see https://zarr.readthedocs.io/en/stable/api/zarr/index.html#zarr.create_array
                    compressor = 'auto'
                elif compression.type == FileAbstraction.Compression.ZLIB:
                    compressor = zarr.codecs.BloscCodec(
                        cname='zlib', clevel=compression.level)
                elif compression.type == FileAbstraction.Compression.LZF:
                    compressor = numcodecs.LZF()
                else:
                    compression = None
                    warnings.warn(
                        f"Compression type '{compression.type}' not supported by zarr. Using no compression.")
            ds = await parent_group.create_array(
                name=name, data=data,
                chunks=chunk_size, compressors=compressor)
            # "upgrade" the object to a _ZarrArray
            ds = _zarrFile._to_ZarrArray(ds)
            return ds

        # -------------------- Listing --------------------

        async def list_objects(self, obj):
            if isinstance(obj, str):
                obj = await self._root.getitem(obj)
            return tuple([str(el) async for el in obj.keys()])

        async def object_exists(self, full_path) -> bool:
            return await self._root.contains(full_path)

        async def list_attributes(self, obj):
            if isinstance(obj, str):
                obj = await self._root.getitem(obj)
            return (str(attr) for attr in obj.attrs.keys())

        # -------------------- File Management --------------------

        def close(self):
            self._store.close()

        # -------------------- Properties --------------------

        async def is_read_only(self) -> bool:
            return self._store.read_only

_AbstractFile = _zarrFile
