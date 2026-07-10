from __future__ import annotations

import numpy as np
import asyncio
import warnings

from .file_abstraction import FileAbstraction, _async_getitem, sync, _gather_sync
from .utils import concatenate_paths, list_objects_matching_pattern_async
from . import units
from .metadata.types import MetadataItem

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Any
    # import Data only for type checking to avoid circular imports
    from .data import Data

# do not include 'Same_as' in the list of standard attributes, as it needs to be handled separately in the code
_STANDARD_ATTRIBUTES = ['Datetime', 'Description', 'Temperature', 'FSR']

class Calibration:
    """
    Access calibration spectra and shift metadata for a data group.

    A calibration group may contain one or more calibration materials. Each
    material stores spectra and a reference shift (typically in GHz), and may
    optionally include an index mapping from image coordinates to calibration
    spectra.
    """

    def __init__(self, file: FileAbstraction, full_path: str, *, 
                 data_group: Data, _initialize: bool = True):
        """
        Initialize the Calibration object.

        Args:
            file (FileAbstraction): Parent file abstraction.
            full_path (str): Path to the group storing calibration datasets.
            data_group (Data): Data group associated with this calibration group.
            _initialize (bool): FOR INTERNAL USE ONLY. Whether to automatically initialize the calibration datasets. 
                Set to False if you want to initialize them manually later using the _init_async() method. Default is True.
        """
        self._file = file
        self._path = full_path
        self._data_group = data_group

        if _initialize:
            sync(self._init_async())        
        
    async def _init_async(self) -> None:
        """
        Asynchronous initialization method to open the calibration datasets.

        This method is called internally by the constructor and should not be called directly.
        """
        self._index = None # dataset containing the indices of the calibration data
        self._calibration_arrays: dict[int, Any] = {} # dictionary to store the calibration data arrays, with numeric keys corresponding to the index of the calibration material
        index, spectra_arrs = await asyncio.gather(
            self._file.open_dataset(concatenate_paths(self._path, 'Index')),
            list_objects_matching_pattern_async(self._file, self._path, r"^(\d+)$"),
            return_exceptions=True
        )
        # open the calibration datasets
        if isinstance(spectra_arrs, Exception):
            raise ValueError(f"No calibration data found in {self._path}: {spectra_arrs}")
        spectra_arrs = [name for name, _ in spectra_arrs]
        coros = [self._file.open_dataset(concatenate_paths(self._path, name)) for name in spectra_arrs]
        cal_arrs = await asyncio.gather(*coros)
        self._calibration_arrays = {int(name): arr for name, arr in zip(spectra_arrs, cal_arrs)}
        # sort the calibration arrays by their numeric keys
        self._calibration_arrays = {m: self._calibration_arrays[m] for m in sorted(self._calibration_arrays.keys())}

        if not isinstance(index, Exception):
            self._index = index
            # TODO: check that the shape of the index dataset is consistent with the shape of the PSD.
            if self._data_group._sparse:
                if self._index.ndim != 1:
                    raise ValueError(f"Calibration index shape {self._index.shape} should be 1D for sparse data groups")
            else:
                if self._index.ndim != 3:
                    raise ValueError(f"Calibration index shape {self._index.shape} should be 3D for non-sparse data groups")
        
        for m, arr in self._calibration_arrays.items():
            if arr.ndim!=2:
                raise ValueError(f"Calibration array {m} should be 2D, but has shape {arr.shape}")
            if arr.shape[0] > 1 and self._index is None:
                raise ValueError(f"Calibration array {m} has more than one spectrum but no index dataset found")


    def get_spectrum_at_coor(self, coor: tuple, m: int = 0) -> tuple:
        """
        Retrieve the calibration spectrum for a given spatial coordinate and material.

        Args:
            coor (tuple): Spatial coordinate as ``(z, y, x)``.
            m (int): Calibration material index.

        Returns:
            tuple: ``(spectrum, shift)``, where ``spectrum`` is a 1D NumPy array and
            ``shift`` is a :class:`MetadataItem` containing the material shift value
            and its units.

        Raises:
            ValueError: If `coor` does not contain three coordinates, or if the
                selected spectrum/shift cannot be retrieved.
            IndexError: If calibration material `m` does not exist.
        """
        if len(coor) != 3:
            raise ValueError("coor must contain 3 values for z, y, x")
        
        if m not in self._calibration_arrays:
            raise IndexError(f"Calibration material {m} not found in calibration group {self._path}")
        cal_arr_m = self._calibration_arrays[m]

        i = 0
        if self._index is not None:
            if self._data_group._sparse:
                index = int(self._data_group._spatial_map[coor])
                i = int(self._index[index])
            else:
                i = int(self._index[coor])
        
        coros = [_async_getitem(cal_arr_m, (i, slice(None))),
                 self._file.get_attr(cal_arr_m, 'Shift'),
                 units.of_attribute(self._file, cal_arr_m, 'Shift')]                         
        spectrum, shift, shift_units = _gather_sync(*coros, return_exceptions=True)
        if spectrum is None or isinstance(spectrum, Exception):
            raise ValueError(f"Could not retrieve calibration spectrum for material {m} at coordinates {coor}: {spectrum}")
        spectrum = np.array(spectrum)
        if isinstance(shift, Exception) or shift is None:
            raise ValueError(f"Could not retrieve shift for calibration material {m}: {shift}")
        if isinstance(shift_units, Exception) or shift_units is None:
            shift_units = 'GHz' # default units for the shift if not specified
            warnings.warn(f"Shift units not found for calibration material {m}, defaulting to 'GHz'")
            
        return spectrum, MetadataItem(shift, shift_units)
    
    def list_calibration_materials(self) -> list[int]:
        """
        List the available calibration materials in this calibration group.

        Returns:
            list[int]: A list of calibration material indices.
        """
        return list(self._calibration_arrays.keys())
    