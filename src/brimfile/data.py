import numpy as np
import asyncio

import warnings
from typing import Any
from numpy.typing import NDArray

from .file_abstraction import FileAbstraction, sync, _async_getitem, _gather_sync
from .utils import concatenate_paths, list_objects_matching_pattern_async, get_object_name, set_object_name
from .utils import np_array_to_smallest_int_type, _determine_chunk_size

from .metadata import Metadata
from .metadata.types import MetadataItem

from numbers import Number

from . import units
from .analysis_results import AnalysisResults
from .calibration import Calibration
from .constants import brim_obj_names

__docformat__ = "google"


class Data:
    """
    Represents a data group within the brim file.
    """
    # make AnalysisResults available as an attribute of Data
    AnalysisResults = AnalysisResults

    def __init__(self, file: FileAbstraction, path: str, *, 
                 newly_created = False, _initialize = True):
        """
        Initialize the Data object. This constructor should not be called directly.

        Args:
            file (File): The parent File object.
            path (str): The path to the data group within the file.
            newly_created (bool): Whether this data group is being created as new.
                            If True, the constructor will not attempt to load spatial mapping.
            _initialize (bool): FOR INTERNAL USE ONLY. Whether to automatically initialize the calibration datasets. 
                Set to False if you want to initialize them manually later using the _init_async() method. Default is True.
        """
        self._file = file
        self._path = path
        

        if _initialize:
            sync(self._init_async(newly_created=newly_created))        
    
    async def _init_async(self, newly_created = False) -> None:
        """
        See __init__() for the description of the arguments.
        """
        self._group = await self._file.open_group(self._path)

        self._sparse = await self._load_sparse_flag_async()
        # the _spatial_map is None for non sparse data but the _spatial_map_px_size should always be valid
        self._spatial_map, self._spatial_map_px_size = await self._load_spatial_mapping_async() if not newly_created else (None, None)

    def get_name(self):
        """
        Returns the name of the data group.
        """
        return sync(get_object_name(self._file, self._path))
    
    def get_index(self):
        """
        Returns the index of the data group.
        """
        return int(self._path.split('/')[-1].split('_')[-1])

    async def _load_sparse_flag_async(self) -> bool:
        """
        Load the 'Sparse' flag for the data group.

        Returns:
            bool: The value of the 'Sparse' flag, or False if the attribute is not found or invalid.
        """
        try:
            sparse = await self._file.get_attr(self._group, 'Sparse')
            if isinstance(sparse, bool):
                return sparse
            else:
                warnings.warn(
                    f"Invalid value for 'Sparse' attribute in {self._path}. Expected a boolean, got {type(sparse)}. Defaulting to False.")
                return False
        except Exception:
            # if the attribute is not found, return the default value False
            return False

    async def _load_spatial_mapping_async(self, load_in_memory: bool=True) -> tuple:
        """
        Load a spatial mapping in the same format as 'Cartesian visualisation',
        irrespectively on whether 'Spatial_map' is defined instead.
        -1 is used for "empty" pixels in the image
        Args:
            load_in_memory (bool): Specify whether the map should be forced to load in memory or just opened as a dataset.
        Returns:
            The spatial map and the corresponding pixel size as a tuple of 3 Metadata.Item, both in the order z, y, x.
            If the spatial mapping is not defined in the file, returns None for the spatial map.
            The pixel size is read from the data group for non-sparse data.
        """
        cv = None
        px_size = 3*(Metadata.Item(value=1, units=None),)

        cv_path = concatenate_paths(
            self._path, brim_obj_names.data.cartesian_visualisation)
        sm_path = concatenate_paths(
            self._path, brim_obj_names.data.spatial_map)
        
        if await self._file.object_exists(cv_path):
            cv = await self._file.open_dataset(cv_path)

            #read the pixel size from the 'Cartesian visualisation' dataset
            px_size_val = None
            px_size_units = None
            try:
                px_size_val = await self._file.get_attr(cv, 'element_size')
                if px_size_val is None or len(px_size_val) != 3:
                    raise ValueError(
                        "The 'element_size' attribute of 'Cartesian_visualisation' must be a tuple of 3 elements")
            except Exception:
                px_size_val = 3*(1,)
                warnings.warn(
                    "No pixel size defined for Cartesian visualisation")            
            px_size_units = await units.of_attribute(
                    self._file, cv, 'element_size')
            px_size = ()
            for i in range(3):
                # if px_size_val[i] is not a number, set it to 1 and px_size_units to None
                if isinstance(px_size_val[i], Number):
                    px_size += (Metadata.Item(px_size_val[i], px_size_units), )
                else:
                    px_size += (Metadata.Item(1, None), )
                    

            if load_in_memory:
                cv = await cv.to_np_array()  # load the spatial map in memory as a numpy array
                cv = np_array_to_smallest_int_type(cv)

        elif await self._file.object_exists(sm_path):
            async def load_spatial_map_from_file(self):
                async def load_coordinate_from_sm(coord: str):
                    res = np.empty(0)  # empty array
                    try:
                        res = await self._file.open_dataset(
                            concatenate_paths(sm_path, coord))
                        res = await res.to_np_array()
                        res = np.squeeze(res)  # remove single-dimensional entries
                    except Exception as e:
                        # if the coordinate does not exist, return an empty array
                        pass
                    if len(res.shape) > 1:
                        raise ValueError(
                            f"The 'Spatial_map/{coord}' dataset is not a 1D array as expected")
                    return res

                def check_coord_array(arr, size):
                    if arr.size == 0:
                        return np.zeros(size)
                    elif arr.size != size:
                        raise ValueError(
                            "The 'Spatial_map' dataset is invalid")
                    return arr

                x, y, z = await asyncio.gather(
                    load_coordinate_from_sm('x'),
                    load_coordinate_from_sm('y'),
                    load_coordinate_from_sm('z')
                    )
                size = max([x.size, y.size, z.size])
                if size == 0:
                    raise ValueError("The 'Spatial_map' dataset is empty")
                x = check_coord_array(x, size)
                y = check_coord_array(y, size)
                z = check_coord_array(z, size)
                return x, y, z

            def calculate_step(x):
                n = len(np.unique(x))
                if n == 1:
                    d = None
                else:
                    d = (np.max(x)-np.min(x))/(n-1)
                return n, d

            x, y, z = await load_spatial_map_from_file(self)

            # TODO extend the reconstruction to non-cartesian cases

            nX, dX = calculate_step(x)
            nY, dY = calculate_step(y)
            nZ, dZ = calculate_step(z)

            indices = np_array_to_smallest_int_type(np.lexsort((x, y, z)))
            cv = np.reshape(indices, (nZ, nY, nX))

            px_size_units = await units.of_object(self._file, sm_path)
            px_size = ()
            for i in range(3):
                px_sz = (dZ, dY, dX)[i]
                px_unit = px_size_units
                if px_sz is None:
                    px_sz = 1
                    px_unit = None
                px_size += (Metadata.Item(px_sz, px_unit),)
        elif not self._sparse:
            try:
                px_sz = await self._file.get_attr(self._group, 'element_size')
                if len(px_sz) != 3:
                    raise ValueError(
                        "The 'element_size' attribute must be a tuple of 3 elements")
                px_unit = None
                try:
                    px_unit = await units.of_attribute(self._file, self._group, 'element_size')
                except Exception:
                    warnings.warn("Pixel size unit is not provided for non-sparse data.")
                px_size = tuple(Metadata.Item(el, px_unit) for el in px_sz)
            except Exception:
                warnings.warn("Pixel size is not provided for non-sparse data.")

        return cv, px_size

    def get_PSD(self) -> tuple:
        """
        LOW LEVEL FUNCTION

        Retrieve the Power Spectral Density (PSD) and frequency from the current data group.
        Note: this function exposes the internals of the brim file and thus the interface might change in future versions.
        Use only if more specialized functions are not working for your application!
        Returns:
            tuple: (PSD, frequency, PSD_units, frequency_units)
                - PSD: A 2D (or more) numpy array containing all the spectra (see [specs](https://github.com/brillouin-imaging/Brillouin-standard-file/blob/main/docs/brim_file_specs.md) for more details).
                - frequency: A numpy array representing the frequency data (see [specs](https://github.com/brillouin-imaging/Brillouin-standard-file/blob/main/docs/brim_file_specs.md) for more details).
                - PSD_units: The units of the PSD.
                - frequency_units: The units of the frequency.
        """
        warnings.warn(
            "Data.get_PSD is deprecated and will be removed in a future release. "
            "Use Data.get_PSD_as_spatial_map instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        PSD, frequency = _gather_sync(
            self._file.open_dataset(concatenate_paths(
                self._path, brim_obj_names.data.PSD)),
            self._file.open_dataset(concatenate_paths(
                self._path, brim_obj_names.data.frequency))
        )
        # retrieve the units of the PSD and frequency
        PSD_units, frequency_units = _gather_sync(
            units.of_object(self._file, PSD),
            units.of_object(self._file, frequency)
        )

        return PSD, frequency, PSD_units, frequency_units
    
    def get_PSD_as_spatial_map(self, *, broadcast_frequency: bool = True) -> tuple:
        """
        Retrieve the Power Spectral Density (PSD) as a spatial map and the frequency from the current data group.
        Arguments:
            broadcast_frequency (bool): Whether to broadcast the frequency array to match the shape of the PSD if they have different shapes. 
                This is useful when the frequency is the same for all spectra and thus stored as a 1D array, while the PSD has a spatial dimension. 
                If False, the function will return a 1D array for the frequency, if the frequency is the same for all spectra.
        Returns:
            tuple: (PSD, frequency, PSD_units, frequency_units)
                - PSD: A 4D (or more) numpy array containing all the spectra. Dimensions are z, y, x, [parameters], spectrum.
                - frequency: A numpy array representing the frequency data, which has the same shape as PSD or a 1D array (see `broadcast_frequency`).
                - PSD_units: The units of the PSD.
                - frequency_units: The units of the frequency.
        """
        PSD, frequency = _gather_sync(
            self._file.open_dataset(concatenate_paths(
                self._path, brim_obj_names.data.PSD)),        
            self._file.open_dataset(concatenate_paths(
                self._path, brim_obj_names.data.frequency))
            )        
        # retrieve the units of the PSD and frequency
        PSD_units, frequency_units = _gather_sync(
            units.of_object(self._file, PSD),
            units.of_object(self._file, frequency)
        )

        # ensure PSD and frequency are numpy arrays
        PSD = np.array(PSD)  
        frequency = np.array(frequency)  # ensure it's a numpy array
        
        # if the frequency is not the same for all spectra, broadcast it to match the shape of PSD
        # if it is the same for all spectra, broadcast_frequency determines whether to return it as a 1D array or broadcast it to match the shape of PSD
        if frequency.ndim > 1 or (broadcast_frequency and frequency.shape != PSD.shape):
            frequency = np.broadcast_to(frequency, PSD.shape)
        
        if self._sparse:
            if self._spatial_map is None:
                raise ValueError("The data is defined as sparse, but no spatial mapping is provided.")
            sm = np.array(self._spatial_map)
            # reshape the PSD and frequency to have the spatial dimensions first      
            PSD = PSD[sm, ...]
            # reshape the frequency only if it is not the same for all spectra
            if frequency.ndim > 1:
                frequency = frequency[sm, ...]

        return PSD, frequency, PSD_units, frequency_units

    def _get_spectrum(self, index: int | tuple[int, int, int]) -> tuple:
        """
        Synchronous wrapper for `_get_spectrum_async` (see doc for `brimfile.data.Data._get_spectrum_async`)
        """
        return sync(self._get_spectrum_async(index))
    async def _get_spectrum_async(self, index: int | tuple[int, int, int]) -> tuple:
        """
        Retrieve a spectrum from the data group by its index or coordinates.

        Args:
            index (int | tuple[int, int, int]): The index (for sparse data) or z, y, x coordinates (for non-sparse data) of the spectrum to retrieve.

        Returns:
            tuple: (PSD, frequency, PSD_units, frequency_units) for the specified index. 
                    PSD can be 1D or more (if there are additional parameters);
                    frequency has the same size as PSD
        Raises:
            IndexError: If the index is out of range for the PSD dataset.
        """
        if self._sparse and not isinstance(index, int):
            raise ValueError("For sparse data, index must be an integer.")
        elif not self._sparse and not (isinstance(index, tuple) and len(index) == 3):
            raise ValueError("For non-sparse data, index must be a tuple of (z, y, x) coordinates.")
            
        # index = -1 corresponds to no spectrum
        if self._sparse and index < 0:
            return None, None, None, None
        elif not self._sparse and any(i < 0 for i in index):
            return None, None, None, None
        PSD, frequency = await asyncio.gather(
            self._file.open_dataset(concatenate_paths(
                self._path, brim_obj_names.data.PSD)),                       
            self._file.open_dataset(concatenate_paths(
                self._path, brim_obj_names.data.frequency))
            )
        if self._sparse and index >= PSD.shape[0]:
            raise IndexError(
                f"index {index} out of range for PSD with shape {PSD.shape}")
        elif not self._sparse and any(i >= PSD.shape[j] for j, i in enumerate(index)):
            raise IndexError(
                f"index {index} out of range for PSD with shape {PSD.shape}")
        # retrieve the units of the PSD and frequency
        PSD_units, frequency_units = await asyncio.gather(
            units.of_object(self._file, PSD),
            units.of_object(self._file, frequency)
        )
        # add ellipsis to the index to select the spectrum and the corresponding frequency
        if self._sparse:
            index = (index, ...)
        else:
            index = index + (..., )
        # map index to the frequency array, considering the broadcasting rules
        index_frequency = index
        if frequency.ndim < PSD.ndim:
            if self._sparse:
                # given the definition of the brim file format,
                # if the frequency has less dimensions that PSD,
                # it can only be because it is the same for all the spatial position (first dimension)
                index_frequency = (..., )
            else:
                unassigned_indices = PSD.ndim - frequency.ndim
                if unassigned_indices == 3:
                    # if the frequency has no spatial dimension, it is the same for all the spatial positions
                    index_frequency = (..., )
                else:
                    # if the frequency has some spatial dimensions but not all, we need to add the corresponding indices to the index of the frequency
                    index_frequency = index[-unassigned_indices:] + (..., )
        #get the spectrum and the corresponding frequency at the specified index
        PSD, frequency = await asyncio.gather(
            _async_getitem(PSD, index),
            _async_getitem(frequency, index_frequency)
        )
        #broadcast the frequency to match the shape of PSD if needed
        if frequency.ndim < PSD.ndim:
            frequency = np.broadcast_to(frequency, PSD.shape)
        return PSD, frequency, PSD_units, frequency_units

    def get_spectrum_in_image(self, coor: tuple) -> tuple:
        """
        Retrieve a spectrum from the data group using spatial coordinates.

        Args:
            coor (tuple): A tuple containing the z, y, x coordinates of the spectrum to retrieve.

        Returns:
            tuple: A tuple containing the PSD, frequency, PSD_units, frequency_units for the specified coordinates. See `Data._get_spectrum_async` for details.
        """
        if len(coor) != 3:
            raise ValueError("coor must contain 3 values for z, y, x")

        if self._sparse:
            index = int(self._spatial_map[coor])
            return self._get_spectrum(index)
        else:
            return self._get_spectrum(coor)
          
    async def get_spectrum_and_all_quantities_in_image_async(self, ar: 'Data.AnalysisResults', coor: tuple, index_peak: int = 0) -> tuple[tuple, dict]:
        """
        Retrieve the spectrum and all available quantities from the analysis results at a specific spatial coordinate.

        Args:
            ar (Data.AnalysisResults): The analysis results object to retrieve quantities from.
            coor (tuple): A tuple containing the z, y, x coordinates in the image.
            index_peak (int, optional): The index of the peak to retrieve (for multi-peak fits). Defaults to 0.

        Returns:
            tuple: A tuple containing:
                - spectrum (tuple): (PSD, frequency, PSD_units, frequency_units) at the specified coordinate
                - quantities (dict): Dictionary of Metadata.Item in the form result[quantity.name][peak.name]
        """
        if len(coor) != 3:
            raise ValueError("coor must contain 3 values for z, y, x")
        index = coor
        if self._sparse:
            index = int(self._spatial_map[coor])
        spectrum, quantities = await asyncio.gather(
            self._get_spectrum_async(index),
            ar._get_all_quantities_at_index(index, index_peak)
        )
        return spectrum, quantities
    def get_spectrum_and_all_quantities_in_image(self, ar: 'Data.AnalysisResults', coor: tuple, index_peak: int = 0) -> tuple[tuple, dict]:
        """
        Synchronous wrapper for `get_spectrum_and_all_quantities_in_image_async` (see doc for `brimfile.data.Data.get_spectrum_and_all_quantities_in_image_async`)
        """
        return sync(self.get_spectrum_and_all_quantities_in_image_async(ar, coor, index_peak))

    def get_metadata(self):
        """
        Returns the metadata associated with the current Data group
        Note that this contains both the general metadata stored in the file (which might be redifined by the specific data group)
        and the ones specific for this data group
        """
        return Metadata(self._file, self._path)

    def get_num_parameters(self) -> tuple:
        """
        Retrieves the number of parameters

        Returns:
            tuple: The shape of the parameters if they exist, otherwise an empty tuple.
        """
        pars, _ = self.get_parameters()
        return pars.shape if pars is not None else ()

    def get_parameters(self) -> list:
        """
        Retrieves the parameters  and their associated names.

        If PSD.ndims > 2, the parameters are stored in a separate dataset.

        Returns:
            list: A tuple containing the parameters and their names if there are any, otherwise None.
        """
        pars_full_path = concatenate_paths(
            self._path, brim_obj_names.data.parameters)
        if sync(self._file.object_exists(pars_full_path)):
            pars = sync(self._file.open_dataset(pars_full_path))
            pars_names = sync(self._file.get_attr(pars, 'Name'))
            return (pars, pars_names)
        return (None, None)

    def create_calibration_group(self, *, index: NDArray[np.integer] | None = None, calibration_data: list[dict[str, Any]] | None = None,
                                 timestamp: list[np.array] | None = None, same_as: int | None = None, attributes: dict[str, MetadataItem] = None,
                                 compression: FileAbstraction.Compression = FileAbstraction.Compression()) -> Calibration:
        """
        Create a new calibration group in the current data group.
        For more details on the expected format of the calibration data, see https://github.com/brillouin-imaging/Brillouin-standard-file/blob/main/docs/brim_file_specs.md.

        Parameters:
            index (np.array | None, optional): Index array for the calibration spectra. For sparse data,
                this must be 1D; for non-sparse data, this must be 3D.  
                It can be omitted if each element in `calibration_data` contains only one spectrum.
            calibration_data (list[dict[str, Any]] | None, optional): Calibration entries to store.
                Each dictionary must contain `spectra` and `shift` keys, and may provide `shift_units`.                
            timestamp (list[np.array] | None, optional): Timestamp arrays corresponding to each calibration
                entry. If provided, its length must match `calibration_data`. Defaults to None.
            same_as (int | None, optional): If provided, links this calibration group to an existing
                calibration via the `Same_as` attribute. When set, the other data arguments are ignored.
                Defaults to None.
            attributes (dict[str, MetadataItem], optional): Additional attributes to attach to the calibration group.
                Can be one of ('Datetime', 'Description', 'Temperature', 'FSR') with the relative units (when relevant).
            compression (FileAbstraction.Compression, optional): Compression settings used for created
                datasets. Defaults to FileAbstraction.Compression().

        Returns:
            Calibration: The newly created calibration group.

        Raises:
            ValueError: If the provided calibration data, index, or timestamp is invalid or inconsistent.
        """
        calibration_path = concatenate_paths(self._path, brim_obj_names.data.calibration)
        calibration_group = sync(self._file.create_group(calibration_path))

        # if same_as is provided, create the 'Same_as' attribute to link the calibration group to an existing one
        if same_as is not None:
            sync(self._file.create_attr(calibration_group, 'Same_as', same_as))
        else: # if same_as is provided, the other parameters are ignored
            # check that calibration_data is provided and valid
            if calibration_data is None:
                raise ValueError("'calibration_data' is required when 'same_as' is not provided")
            if not isinstance(calibration_data, (list, tuple)):
                calibration_data = [calibration_data,]
            # check that index is valid if provided
            if index is not None:
                # TODO: check of the shape of 'index' is compatible with PSD
                if self._sparse and index.ndim != 1:
                    raise ValueError("'index' must be a 1D array for sparse data")
                if not self._sparse and index.ndim != 3:
                    raise ValueError("'index' must be a 3D array for non-sparse data")
            # check that timestamp is valid if provided
            if timestamp is not None: 
                if not isinstance(timestamp, (list, tuple)):
                    timestamp = [timestamp,]
                if len(timestamp) != len(calibration_data):
                    raise ValueError("If 'timestamp' is provided, it must have the same length as 'calibration_data'")            

            for m, calib in enumerate(calibration_data):
                # check that each element in calibration_data is a dictionary containing 'spectra' and 'shift' keys
                if not isinstance(calib, dict):
                    raise ValueError("Each element in 'calibration_data' must be a dictionary")
                if 'spectra' not in calib.keys() or 'shift' not in calib.keys():
                    raise ValueError("Each calibration data dictionary must contain 'spectra' and 'shift' keys")
                # retrieve the spectra, shift and shift_units from the calibration data and check that they are valid
                cal_spectra = np.array(calib['spectra'])
                if cal_spectra.ndim != 2:
                    raise ValueError("'spectra' in calibration data must be a 2D array. If only one spectrum is provided, set the first dimension to 1.")
                cal_shift = calib['shift']
                cal_shift_units = calib.get('shift_units', None)
                if cal_shift_units is None:
                    cal_shift_units = 'GHz'
                    warnings.warn("No units provided for 'shift' in calibration data, defaulting to GHz")
                # check that index is compatible with the shape of the spectra
                if index is None and cal_spectra.shape[0] != 1:
                    raise ValueError("If 'index' is not provided, each element in 'calibration_data' must contain only one spectrum (i.e. have shape (1, n))")
                if index is not None and np.max(index) >= cal_spectra.shape[0]:
                    raise ValueError("If 'index' is provided, its maximum value must be less than the number of spectra in each calibration data element")
                # add the m arrays together with their attributes to the file
                spectra_dataset = sync(self._file.create_dataset(calibration_group, f'{m}', cal_spectra, chunk_size=_determine_chunk_size(cal_spectra), compression=compression))
                sync(self._file.create_attr(spectra_dataset, 'Shift', cal_shift))
                units.add_to_attribute(self._file, spectra_dataset, 'Shift', cal_shift_units)
                if timestamp is not None:
                    timestamp_array = np.array(timestamp[m])
                    if timestamp_array.ndim != 1 or timestamp_array.shape[0] != cal_spectra.shape[0]:
                        raise ValueError("Each element in 'timestamp' must be a 1D array with the same length as the number of spectra in each calibration data element")
                    sync(self._file.create_dataset(calibration_group, f'Timestamp_{m}', timestamp_array, compression=compression))
            # add the index array to the file
            if index is not None:
                sync(self._file.create_dataset(calibration_group, 'Index', index, compression=compression))
        
        from .calibration import _STANDARD_ATTRIBUTES
        # add any additional attributes to the calibration group, checking that they do not overwrite the standard
        if attributes is not None:
            for key, value in attributes.items():
                if key not in _STANDARD_ATTRIBUTES:
                    warnings.warn(f"Attribute '{key}' is not a standard attribute for calibration groups.\
                                   Standard attributes are: {', '.join(_STANDARD_ATTRIBUTES)}. \
                                   Make sure this is intentional!")
                if not isinstance(value, MetadataItem):
                    value = MetadataItem(value)
                sync(self._file.create_attr(calibration_group, key, value.value))
                if value.units is not None:
                    units.add_to_attribute(self._file, calibration_group, key, value.units)

        return Calibration(self._file, calibration_path, data_group=self)
    
    def get_calibration(self) -> Calibration:
        """
        Synchronous wrapper for `get_calibration_async` (see doc for `brimfile.data.Data.get_calibration_async`)
        """
        return sync(self.get_calibration_async())

    async def get_calibration_async(self) -> Calibration:
        """
        Retrieve the calibration group associated with the current data group.

        Returns:
            Calibration: The calibration group associated with the current data group.

        Raises:
            ValueError: If no calibration group is found in the current data group or the referenced calibration group does not exist.
        """
        calibration_path = concatenate_paths(self._path, brim_obj_names.data.calibration)
        if not await self._file.object_exists(calibration_path):
            raise ValueError(f"No calibration group found in {self._path}")
        same_as = None
        try:
            same_as = await self._file.get_attr(calibration_path, 'Same_as')
        except Exception:
            pass #  same_as attribute is optional, if it does not exist we just ignore it
        # if the 'Same_as' attribute exists, find the calibration group with the corresponding index
        if same_as is not None:
            try:
                d_m = await Data.from_existing_async(self._file, same_as)
                return await d_m.get_calibration_async()
            except IndexError:
                raise ValueError(f"Calibration group in {self._path} references non-existing calibration index {same_as} in the file")
        cal_group = Calibration(self._file, calibration_path, data_group=self, _initialize=False)
        await cal_group._init_async()
        return cal_group

    def create_analysis_results_group(self, data_AntiStokes, data_Stokes=None, *,
                                          index: int = None, name: str = None, fit_model: 'Data.AnalysisResults.FitModel' = None) -> AnalysisResults:
        """
        Adds a new AnalysisResults entry to the current data group.
        Parameters:
            data_AntiStokes (dict or list[dict]): see documentation for `brimfile.analysis_results.AnalysisResults.add_data`
            data_Stokes (dict or list[dict]): same as data_AntiStokes for the Stokes peaks.
            index (int, optional): The index for the new data entry. If None, the next available index is used. Defaults to None.
            name (str, optional): The name for the new Analysis group. Defaults to None.
            fit_model (Data.AnalysisResults.FitModel, optional): The fit model used for the analysis. Defaults to None (no attribute is set).
        Returns:
            AnalysisResults: The newly created AnalysisResults object.
        Raises:
            IndexError: If the specified index already exists in the dataset.
            ValueError: If any of the data provided is not valid or consistent
        """
        if index is not None:
            try:
                self.get_analysis_results(index)
            except IndexError:
                pass
            else:
                # If the group already exists, raise an error
                raise IndexError(
                    f"Analysis {index} already exists in {self._path}")
        else:
            ar_groups = self.list_AnalysisResults()
            indices = [ar['index'] for ar in ar_groups]
            indices.sort()
            index = indices[-1] + 1 if indices else 0  # Next available index

        ar = Data.AnalysisResults._create_new(self, index=index, sparse=self._sparse)
        if name is not None:
            set_object_name(self._file, ar._path, name)
        ar.add_data(data_AntiStokes, data_Stokes, fit_model=fit_model)

        return ar

    def list_AnalysisResults(self, retrieve_custom_name=False) -> list:
        """
        List all AnalysisResults groups in the current data group. The list is ordered by index.

        Returns:
            list: A list of dictionaries, each containing:
                - 'name' (str): The name of the AnalysisResults group.
                - 'index' (int): The index extracted from the group name.
                - 'custom_name' (str, optional): if retrieve_custom_name==True, it contains the name of the AnalysisResults group as returned from utils.get_object_name.
        """

        analysis_results_groups = []

        matched_objs = sync(list_objects_matching_pattern_async(
            self._file, self._group, brim_obj_names.data.analysis_results + r"_(\d+)$"))
        async def _make_dict_item(matched_obj, retrieve_custom_name):
            name = matched_obj[0]
            index = int(matched_obj[1])
            curr_obj_dict = {'name': name, 'index': index}
            if retrieve_custom_name:
                ar_path = concatenate_paths(self._path, name)
                custom_name = await get_object_name(self._file, ar_path)
                curr_obj_dict['custom_name'] = custom_name
            return curr_obj_dict
        coros = [_make_dict_item(matched_obj, retrieve_custom_name) for matched_obj in matched_objs]
        dicts = _gather_sync(*coros)
        for dict_item in dicts:
            analysis_results_groups.append(dict_item)
        # Sort the data groups by index
        analysis_results_groups.sort(key=lambda x: x['index'])

        return analysis_results_groups

    def get_analysis_results(self, index: int = 0) -> AnalysisResults:
        """
        Returns the AnalysisResults at the specified index

        Args:
            index (int)                

        Raises:
            IndexError: If there is no analysis with the corresponding index
        """
        name = None
        ls = self.list_AnalysisResults()
        for el in ls:
            if el['index'] == index:
                name = el['name']
                break
        if name is None:
            raise IndexError(f"Analysis {index} not found")
        path = concatenate_paths(self._path, name)
        return Data.AnalysisResults(self._file, path, data_group_path=self._path,
                                    spatial_map=self._spatial_map, spatial_map_px_size=self._spatial_map_px_size, sparse=self._sparse)

    def _add_data(self, PSD: np.ndarray, frequency: np.ndarray, *, scanning: dict = None, freq_units='GHz',
                  timestamp: np.ndarray = None, compression: FileAbstraction.Compression = FileAbstraction.Compression()):
        """
        Add data to the current data group.

        This method adds the provided PSD, frequency, and scanning data to the HDF5 group 
        associated with this `Data` object. It validates the inputs to ensure they meet 
        the required specifications before adding them.

        Args:
            PSD (np.ndarray): A 2D numpy array representing the Power Spectral Density (PSD) data. The last dimension contains the spectra.
            frequency (np.ndarray): A 1D or 2D numpy array representing the frequency data. 
                It must be broadcastable to the shape of the PSD array.
            scanning (dict, optional): A dictionary containing scanning-related data. 
                Required for sparse data (sparse=True), optional for non-sparse data.
                For sparse data, must include at least one of 'Spatial_map' or 'Cartesian_visualisation'.
                It may include the following keys:
                - 'Spatial_map' (optional): A dictionary containing coordinate arrays:
                    - 'x', 'y', 'z' (optional): 1D numpy arrays of same length with coordinate values
                    - 'units' (optional): string with the unit (e.g., 'um')
                - 'Cartesian_visualisation' (optional): A 3D numpy array (z, y, x) with integer values 
                   mapping spatial positions to spectra indices. Values must be -1 (invalid/empty pixel) 
                   or between 0 and PSD.shape[0]-1.
                - 'Cartesian_visualisation_pixel' (recommended with Cartesian_visualisation): 
                   Tuple/list of 3 float values (z, y, x) representing pixel size. Unused dimensions can be None.
                - 'Cartesian_visualisation_pixel_unit' (optional): String for pixel size unit (default: 'um').
            timestamp (np.ndarray, optional): Timestamps in milliseconds for each spectrum.
                Must be a 1D array with length equal to PSD.shape[0].


        Raises:
            ValueError: If any of the data provided is not valid or consistent
        """

        # Check if frequency is broadcastable to PSD
        try:
            np.broadcast_shapes(tuple(frequency.shape), tuple(PSD.shape))
        except ValueError as e:
            raise ValueError(f"frequency (shape: {frequency.shape}) is not broadcastable to PSD (shape: {PSD.shape}): {e}")

        # Check if at least one of 'Spatial_map' or 'Cartesian_visualisation' is present in the scanning dictionary
        # This is required for sparse data to establish the spatial mapping
        has_spatial_mapping = False
        if scanning is not None:
            if 'Spatial_map' in scanning:
                sm = scanning['Spatial_map']
                size = 0

                def check_coor(coor: str):
                    if coor in sm:
                        sm[coor] = np.array(sm[coor])
                        size1 = sm[coor].size
                        if size1 != size and size != 0:
                            raise ValueError(
                                f"'{coor}' in 'Spatial_map' is invalid!")
                        return size1
                    return size
                size = check_coor('x')
                size = check_coor('y')
                size = check_coor('z')
                if size == 0:
                    raise ValueError(
                        "'Spatial_map' should contain at least one x, y or z")
                has_spatial_mapping = True
            if 'Cartesian_visualisation' in scanning:
                cv = scanning['Cartesian_visualisation']
                if not isinstance(cv, np.ndarray) or cv.ndim != 3:
                    raise ValueError(
                        "Cartesian_visualisation must be a 3D numpy array")
                if not np.issubdtype(cv.dtype, np.integer) or np.min(cv) < -1 or np.max(cv) >= PSD.shape[0]:
                    raise ValueError(
                        "Cartesian_visualisation values must be integers between -1 and PSD.shape[0]-1")
                if 'Cartesian_visualisation_pixel' in scanning:
                    if len(scanning['Cartesian_visualisation_pixel']) != 3:
                        raise ValueError(
                            "Cartesian_visualisation_pixel must always contain 3 values for z, y, x (set to None if not used)")
                else:
                    warnings.warn(
                        "It is recommended to include 'Cartesian_visualisation_pixel' in the scanning dictionary to define pixel size for proper spatial calibration")
                has_spatial_mapping = True
        if not has_spatial_mapping and self._sparse:
            raise ValueError("For sparse data, 'scanning' must be provided and must contain at least one of 'Spatial_map' or 'Cartesian_visualisation'")

        if timestamp is not None:
            if not isinstance(timestamp, np.ndarray) or timestamp.ndim != 1 or len(timestamp) != PSD.shape[0]:
                raise ValueError("timestamp is not compatible with PSD")

        # TODO: add and validate additional datasets (i.e. 'Parameters', 'Calibration_index', etc.)

        # Add datasets to the group
        sync(self._file.create_dataset(
            self._group, brim_obj_names.data.PSD, data=PSD,
            chunk_size=_determine_chunk_size(PSD), compression=compression))
        freq_ds = sync(self._file.create_dataset(
            self._group,  brim_obj_names.data.frequency, data=frequency,
            chunk_size=_determine_chunk_size(frequency), compression=compression))
        units.add_to_object(self._file, freq_ds, freq_units)

        if scanning is not None:
            if 'Spatial_map' in scanning:
                sm = scanning['Spatial_map']
                sm_group = sync(self._file.create_group(concatenate_paths(
                    self._path, brim_obj_names.data.spatial_map)))
                if 'units' in sm:
                    units.add_to_object(self._file, sm_group, sm['units'])

                def add_sm_dataset(coord: str):
                    if coord in sm:
                        sync(self._file.create_dataset(
                            sm_group, coord, data=sm[coord], compression=compression))

                add_sm_dataset('x')
                add_sm_dataset('y')
                add_sm_dataset('z')
            if 'Cartesian_visualisation' in scanning:
                # convert the Cartesian_visualisation to the smallest integer type
                cv_arr = np_array_to_smallest_int_type(scanning['Cartesian_visualisation'])
                cv = sync(self._file.create_dataset(self._group, brim_obj_names.data.cartesian_visualisation,
                                            data=cv_arr, compression=compression))
                if 'Cartesian_visualisation_pixel' in scanning:
                    sync(self._file.create_attr(
                        cv, 'element_size', scanning['Cartesian_visualisation_pixel']))
                    if 'Cartesian_visualisation_pixel_unit' in scanning:
                        px_unit = scanning['Cartesian_visualisation_pixel_unit']
                    else:
                        warnings.warn(
                            "No unit provided for Cartesian_visualisation_pixel, defaulting to 'um'")
                        px_unit = 'um'
                    units.add_to_attribute(self._file, cv, 'element_size', px_unit)

        self._spatial_map, self._spatial_map_px_size = sync(self._load_spatial_mapping_async())

        if timestamp is not None:
            sync(self._file.create_dataset(
                self._group, 'Timestamp', data=timestamp, compression=compression))

    @staticmethod
    def list_data_groups(file: FileAbstraction, retrieve_custom_name=False) -> list:
        """
        Synchronous wrapper for `list_data_groups_async` (see doc for `brimfile.data.Data.list_data_groups_async`)
        """
        return sync(Data.list_data_groups_async(file, retrieve_custom_name))

    @staticmethod
    async def list_data_groups_async(file: FileAbstraction, retrieve_custom_name=False) -> list:
        """
        List all data groups in the brim file. The list is ordered by index.

        Returns:
            list: A list of dictionaries, each containing:
                - 'name' (str): The name of the data group in the file.
                - 'index' (int): The index extracted from the group name.
                - 'custom_name' (str, optional): if retrieve_custom_name==True, it contains the name of the data group as returned from utils.get_object_name.
        """

        data_groups = []

        matched_objs = await list_objects_matching_pattern_async(
            file, brim_obj_names.Brillouin_base_path, brim_obj_names.data.base_group + r"_(\d+)$")
        
        async def _make_dict_item(matched_obj, retrieve_custom_name):
            name = matched_obj[0]
            index = int(matched_obj[1])
            curr_obj_dict = {'name': name, 'index': index}
            if retrieve_custom_name:
                path = concatenate_paths(
                    brim_obj_names.Brillouin_base_path, name)
                custom_name = await get_object_name(file, path)
                curr_obj_dict['custom_name'] = custom_name
            return curr_obj_dict
        
        coros = [_make_dict_item(matched_obj, retrieve_custom_name) for matched_obj in matched_objs]
        dicts = await asyncio.gather(*coros)
        for dict_item in dicts:
            data_groups.append(dict_item)        
        # Sort the data groups by index
        data_groups.sort(key=lambda x: x['index'])

        return data_groups

    @staticmethod
    async def _get_existing_group_name_async(file: FileAbstraction, index: int) -> str:
        """
        Get the name of an existing data group by index.

        Args:
            file (File): The parent File object.
            index (int): The index of the data group.

        Returns:
            str: The name of the data group, or None if not found.
        """
        group_name: str = None
        data_groups = await Data.list_data_groups_async(file)
        for dg in data_groups:
            if dg['index'] == index:
                group_name = dg['name']
                break
        return group_name
    
    @classmethod
    async def from_existing_async(cls, file: FileAbstraction, index: int) -> 'Data':
        """ 
        Create a Data object from an existing data group in the file.
        Args:
            file (File): The parent File object.
            index (int): The index of the existing data group.      
        Returns:
            Data: A Data object corresponding to the existing data group.   
        Raises:
            IndexError: If no data group with the specified index is found in the file.
        """
        group_name: str = await cls._get_existing_group_name_async(file, index)
        if group_name is None:
            raise IndexError(f"No data group with index {index} found in the file")
        dg = cls(file, concatenate_paths(brim_obj_names.Brillouin_base_path, group_name), _initialize=False) 
        await dg._init_async()
        return dg
    
    @classmethod
    def _create_new(cls, file: FileAbstraction, index: int, sparse: bool = False, name: str = None) -> 'Data':
        """
        Create a new data group with the specified index.

        Args:
            file (File): The parent File object.
            index (int): The index for the new data group.
            sparse (bool): Whether the data is sparse. See https://github.com/brillouin-imaging/Brillouin-standard-file/blob/main/docs/brim_file_specs.md for details. Defaults to False.
            name (str, optional): The name for the new data group. Defaults to None.

        Returns:
            Data: The newly created Data object.
        """
        group_name = Data._generate_group_name(index)
        group = sync(file.create_group(concatenate_paths(
            brim_obj_names.Brillouin_base_path, group_name)))
        sync(file.create_attr(group, 'Sparse', sparse))
        if name is not None:
            set_object_name(file, group, name)
        return cls(file, concatenate_paths(brim_obj_names.Brillouin_base_path, group_name), newly_created=True)

    @staticmethod
    def _generate_group_name(index: int, n_digits: int = None) -> str:
        """
        Generate a name for a data group based on the index.

        Args:
            index (int): The index for the data group.
            n_digits (int, optional): The number of digits to pad the index with. If None no padding is applied. Defaults to None.

        Returns:
            str: The generated group name.

        Raises:
            ValueError: If the index is negative.
        """
        if index < 0:
            raise ValueError("index must be positive")
        num = str(index)
        if n_digits is not None:
            num = num.zfill(n_digits)
        return f"{brim_obj_names.data.base_group}_{num}"
