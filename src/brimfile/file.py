from __future__ import annotations

import numpy as np
import warnings

from .data import Data

from .utils import concatenate_paths
from .constants import brim_obj_names
from . import units
from . import subtypes

from .file_abstraction import FileAbstraction, StoreType, sync
from .validation import validate_json, ValidationError, ValidationLevel
from .validation.json_descriptor import generate_json_descriptor

# don't import _AbstractFile if running in pyodide (it is defined in js)
import sys
if "pyodide" not in sys.modules:
    from .file_abstraction import _AbstractFile

__docformat__ = "google"

class File:
    """
    Represents a brim file with Brillouin data, extending h5py.File.
    """

    if "pyodide" in sys.modules:
        def __init__(self, file):
            self._file = file
            self._file.version = self._get_file_version_tuple()
            if not self.is_valid():
                raise ValueError("The brim file is not valid!")
    else:
        def __init__(self, filename: str, mode: str = 'r',
                     store_type: StoreType = StoreType.AUTO, * ,
                     validate: bool = False) -> None:
            """
            Initialize the File object.

            Args:
                filename (str): Path to the brim file.
                mode: {'r', 'r+', 'a', 'w', 'w-'} the mode for opening the file (default is 'r' for read-only).
                            See the definition of `mode` in `brimfile.file_abstraction._zarrFile.__init__()` for more details.
                            'r' means read only (must exist); 'r+' means read/write (must exist);
                            'a' means read/write (create if doesn't exist); 'w' means create (overwrite if exists); 'w-' means create (fail if exists).
                store_type (StoreType): Type of the store to use, as defined in `brimfile.file_abstraction.StoreType`. Default is 'AUTO'.
                validate (bool): Whether to validate the file upon initialization. Default is False.
            """
            self._file = _AbstractFile(
                filename, mode=mode, store_type=store_type)
            self._file.version = self._get_file_version_tuple()
            if not self.is_valid():
                raise ValueError("The brim file is not valid!")
            if validate:
                validation_errors: list[ValidationError] = self.validate()
                for err in validation_errors:
                    if err.level == ValidationLevel.WARNING or err.level == ValidationLevel.ERROR:
                        warnings.warn(f"Validation warning at {err.path}: {err.message}")
                    elif err.level == ValidationLevel.CRITICAL:
                        raise ValueError(f"Validation error at {err.path}: {err.message}")

    @staticmethod
    def _parse_version_tuple3(version) -> tuple[int | str, int | str, int | str]:
        """Normalize a version value into a 3-item tuple (major, minor, patch)."""
        if isinstance(version, str):
            parts = [part for part in version.split('.') if part != '']
        elif isinstance(version, (list, tuple)):
            parts = list(version)
        else:
            raise ValueError(f"Invalid version format: {version}")

        # convert the first 3 parts to int if they are digits (str), otherwise keep them as they are
        normalized: list[int | str] = []
        for part in parts[:3]:
            if isinstance(part, str) and part.isdigit():
                normalized.append(int(part))
            else:
                try:
                    part = int(part)
                    normalized.append(part)
                except (ValueError, TypeError):
                    raise ValueError(f"Invalid version part: {part}")                

        # add zeros to the end of the list until it has 3 elements
        while len(normalized) < 3:
            normalized.append(0)

        return (normalized[0], normalized[1], normalized[2])

    def _get_file_version_tuple(self) -> tuple[int | str, int | str, int | str]:
        """
        Read brim_version from file root attributes and return a 3-item tuple.
        Returns:
            tuple[int | str, int | str, int | str]: A tuple representing the version of the brim file in the format (major, minor, patch).
            If the version can't be read from the file, returns None.
        """
        try:
            version = sync(self._file.get_attr('/', 'brim_version'))
        except Exception:
            return None
        return self._parse_version_tuple3(version)

    def validate(self) -> list[ValidationError]:
        """
        Validate the brim file and return a list of validation errors.

        Returns:
            list[ValidationError]: A list of validation errors found in the brim file.
            If the list is empty, the file is valid.
        """
        json_descriptor = generate_json_descriptor(self._file)
        validation_errors: list[ValidationError] = validate_json(json_descriptor)
        return validation_errors
    
    def __del__(self):
        try:
            if hasattr(self, '_file'):
                self.close()
        except Exception as e:            
            # don't throw an error if the file cannot be closed
            warnings.warn(f"Cannot close the file: {e}")

    def close(self) -> None:
        self._file.close()

    def is_read_only(self) -> bool:
        return sync(self._file.is_read_only())

    def is_valid(self) -> bool:
        """
        Check if the file is a valid brim file.

        Returns:
            bool: True if the file is valid, False otherwise.
        """
        # TODO validate file against https://github.com/brillouin-imaging/Brillouin-standard-file/blob/main/docs/brim_file_specs.md
        return True

    @classmethod
    def create(cls, filename: str, store_type: StoreType = StoreType.AUTO, *, 
               brim_version: str = '0.2') -> 'File':
        """
        Create a new brim file with the specified filename. If the file exists already it will generate an error.

        Args:
            filename (str): Path to the brim file to be created.
            store_type (StoreType): Type of the store to use, as defined in `brimfile.file_abstraction.StoreType`. Default is 'AUTO'.
            brim_version (str): Version of the brim file format to use. Default is '0.2'.

        Returns:
            File: An instance of the File class representing the newly created brim file.
            store_type (str): Type of the store to use, as defined in `brimfile.file_abstraction.StoreType`. Default is 'AUTO'.
        """
        f = cls(filename, mode='w-', store_type=store_type)

        # File version
        f._file.version = cls._parse_version_tuple3(brim_version)
        sync(f._file.create_attr('/', 'brim_version', brim_version))

        # Root Brillouin_data group
        sync(f._file.create_group(brim_obj_names.Brillouin_base_path))

        return f

    def create_data_group(self, PSD: np.ndarray, frequency: np.ndarray, px_size_um: tuple, *, index: int = None,
                          name: str = None, compression: FileAbstraction.Compression = FileAbstraction.Compression()) -> 'Data':
        """
        Adds a new data entry to the file.
        Parameters:
            PSD (np.ndarray): The Power Spectral Density (PSD) data to be added. It must be 4D with dimensions z, y, x, spectrum
            frequency (np.ndarray): The frequency data corresponding to the PSD. It must be broadcastable to the PSD shape (the most common case is frequency being 1D, in which case the frequency axis is assumed the same for all the spatial coordinates)
            px_size_um (tuple): A tuple of 3 elements, in the order z,y,x, corresponding to the pixel size in um. Unused dimensions can be set to None.
            index (int, optional): The index for the new data group. If None, the next available index is used. Defaults to None.
            name (str, optional): The name for the new data group. Defaults to None.
            compression (FileAbstraction.Compression, optional): The compression method to use for the data. Defaults to FileAbstraction.Compression.DEFAULT.
        Returns:
            Data: The newly created Data object.
        Raises:
            IndexError: If the specified index already exists in the dataset.
            ValueError: If any of the data provided is not valid or consistent
        """
        if PSD.ndim != 4:
            raise ValueError(
                "'PSD' must have 4 dimensions (z, y, x, spectrum)")
        try:
            np.broadcast_shapes(tuple(frequency.shape), tuple(PSD.shape))
        except ValueError as e:
            raise ValueError(f"frequency (shape: {frequency.shape}) is not broadcastable to PSD (shape: {PSD.shape}): {e}")
        if len(px_size_um) != 3:
            raise ValueError("'px_size_um' must have 3 elements (z,y,x); unused dimensions can be set to None")

        return self._create_data_group_raw(PSD, frequency, scanning = None, sparse = False, px_size_um=px_size_um, 
                                             index=index, name=name, compression=compression)

    def create_data_group_sparse(self, PSD: np.ndarray, frequency: np.ndarray, scanning: dict, *, timestamp: np.ndarray = None,
                                index: int = None, name: str = None, compression: FileAbstraction.Compression = FileAbstraction.Compression()) -> 'Data':
        """
        Adds a new [sparse data entry](https://github.com/brillouin-imaging/Brillouin-standard-file/blob/main/docs/brim_file_specs.md) to the file.
        
        Sparse data allows storage of spectra in a flattened format (first dimension is the spectrum index),
        with spatial mapping provided separately. This is efficient for data with irregular sampling or missing pixels.
        
        Parameters:
            PSD (np.ndarray): The Power Spectral Density (PSD) data to be added. First dimension is spectrum index,
                last dimension contains the spectral data. Shape: (n_spectra, ..., n_freq_points).
            frequency (np.ndarray): The frequency data corresponding to the PSD. Must be broadcastable to the PSD array.
            scanning (dict): Dictionary defining the spatial mapping. Must include at least 'Spatial_map' or 'Cartesian_visualisation'.
                See `brimfile.data.Data._add_data` docstring for detailed structure of the scanning dictionary.
            timestamp (np.ndarray, optional): Timestamps in milliseconds for the data. Defaults to None.
            index (int, optional): The index for the new data group. If None, the next available index is used. Defaults to None.
            name (str, optional): The name for the new data group. Defaults to None.
            compression (FileAbstraction.Compression, optional): The compression method to use for the data. Defaults to FileAbstraction.Compression.DEFAULT.
        Returns:
            Data: The newly created Data object.
        Raises:
            IndexError: If the specified index already exists in the dataset.
            ValueError: If any of the data provided is not valid or consistent
        """
        return self._create_data_group_raw(PSD, frequency, scanning=scanning, timestamp=timestamp, sparse=True, index=index, name=name, compression=compression)   
    
    def _create_data_group_raw(self, PSD: np.ndarray, frequency: np.ndarray, *, scanning: dict = None, px_size_um = None, timestamp: np.ndarray = None, sparse: bool = False,
                                index: int = None, name: str = None, compression: FileAbstraction.Compression = FileAbstraction.Compression()) -> 'Data':
        """
        Adds a new data entry to the file. Check the documentation for `brimfile.data.Data._add_data` for more details on the parameters.
        Parameters:
            PSD (np.ndarray): The Power Spectral Density (PSD) data to be added. The last dimension contains the spectra.
            frequency (np.ndarray): The frequency data corresponding to the PSD. Must be broadcastable to the PSD array.
            scanning (dict, optional): Spatial mapping metadata. Required for sparse=True, optional for sparse=False.
                See `brimfile.data.Data._add_data` docstring for detailed structure.
            px_size_um (tuple, optional): A tuple of 3 elements (z, y, x) for pixel size in μm. For non-sparse data only.
            timestamp (np.ndarray, optional): Timestamps in milliseconds for the data. Defaults to None.
            sparse (bool): Whether the data is sparse. See https://github.com/brillouin-imaging/Brillouin-standard-file/blob/main/docs/brim_file_specs.md for details. Defaults to False.
            index (int, optional): The index for the new data group. If None, the next available index is used. Defaults to None.
            name (str, optional): The name for the new data group. Defaults to None.
            compression (FileAbstraction.Compression, optional): The compression method to use for the data. Defaults to FileAbstraction.Compression.DEFAULT.
        Returns:
            Data: The newly created Data object.
        Raises:
            IndexError: If the specified index already exists in the dataset.
            ValueError: If any of the data provided is not valid or consistent
        """
        if index is not None:
            if sync(Data._get_existing_group_name_async(self._file, index)) is not None:
                raise IndexError(
                    f"Data {index} already exists in {self._file.filename}")
        else:
            data_groups = self.list_data_groups()
            indices = [dg['index'] for dg in data_groups]
            indices.sort()
            index = indices[-1] + 1 if indices else 0  # Next available index

        # create the data group
        d = Data._create_new(self._file, index, sparse, name)
        # add the pixel size as an attribute of the data group
        if px_size_um is not None:
            sync(self._file.create_attr(d._group, 'element_size', tuple(px_size_um)))
            units.add_to_attribute(self._file, d._group, 'element_size', 'um')
        elif not sparse:
            warnings.warn("Pixel size is not provided for non-sparse data. It is recommended to provide it for proper spatial calibration and visualization.")
        # add the data to the data group
        d._add_data(PSD, frequency, scanning = scanning,
                   timestamp=timestamp, compression=compression)
        return d

    def list_data_groups(self, retrieve_custom_name=False) -> list:
        """
        List all data groups in the brim file.

        Returns:
            See documentation of brimfile.data.Data.list_data_groups
        """
        return Data.list_data_groups(self._file, retrieve_custom_name)

    def get_data(self, index: int = 0) -> 'Data':
        """
        Retrieve a Data object for the specified index.

        Args:
            index (int): The index of the data group to retrieve.

        Returns:
            Data: The Data object corresponding to the specified index.
        Raises:
            IndexError: If the specified index does not exist in the dataset.
        """        
        return sync(Data.from_existing_async(self._file, index))

    @property
    def filename(self) -> str:
        """
        Get the filename of the brim file.

        Returns:
            str: The filename of the brim file.
        """
        return self._file.filename
    
    @property
    def subtype(self) -> subtypes.SubType:
        """
        Get the subtype of the brim file.

        Returns:
            subtypes.SubType: The subtype of the brim file.
        """
        return subtypes.get_subtype(self._file)
