from __future__ import annotations

import numpy as np
import asyncio

import warnings

from enum import Enum

from .file_abstraction import FileAbstraction, sync, _async_getitem, _gather_sync
from . import units
from .utils import var_to_singleton, concatenate_paths, get_object_name
from .constants import brim_obj_names

from .fitting_models import FitModel

from .metadata import Metadata
from .physics import Brillouin_shift_water, Brillouin_width_water

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    # import here to avoid circular imports
    from .data import Data

__docformat__ = "google"

class AnalysisResults:
    """
    Rapresents the analysis results associated with a Data object.
    """

    class Quantity(Enum):
        """
        Enum representing the type of analysis results.
        """
        Shift = "Shift"
        # elastic contrast as defined in https://doi.org/10.1007/s12551-020-00701-9
        Elastic_contrast = "Elastic_contrast"
        # viscous contrast as defined in https://doi.org/10.1007/s12551-020-00701-9
        Viscous_contrast = "Viscous_contrast"
        Width = "Width"
        Amplitude = "Amplitude"
        Offset = "Offset"
        R2 = "R2"
        RMSE = "RMSE"
        Cov_matrix = "Cov_matrix"

    class PeakType(Enum):
        AntiStokes = "AS"
        Stokes = "S"
        average = "avg"
    
    FitModel = FitModel

    def __init__(self, file: FileAbstraction, full_path: str, *, data_group_path: str,
                    spatial_map = None, spatial_map_px_size = None, sparse: bool = False):
        """
        Initialize the AnalysisResults object.

        Args:
            file (FileAbstraction): Parent file abstraction object.
            full_path (str): Path of the group storing the analysis results.
            data_group_path (str): Path of the data group associated with the analysis results.
            spatial_map (optional): Spatial map used for sparse analysis results.
            spatial_map_px_size (optional): Pixel size associated with ``spatial_map``.
            sparse (bool): Whether the analysis results are stored in sparse format.

        Raises:
            ValueError: If ``sparse`` is ``True`` and either ``spatial_map`` or
                ``spatial_map_px_size`` is not provided.
        """
        self._file = file
        self._path = full_path
        self._data_group_path = data_group_path
        # self._group = file.open_group(full_path)
        self._spatial_map = spatial_map
        self._spatial_map_px_size = spatial_map_px_size
        self._sparse = sparse
        if sparse:
            if spatial_map is None or spatial_map_px_size is None:
                raise ValueError("For sparse analysis results, the spatial map and pixel size must be provided.")
    def _get_metadata(self) -> Metadata:
        """
        Retrieve the Metadata object associated with the current AnalysisResults.

        Returns:
            Metadata: The Metadata object associated with the current Data group.
        """
        return Metadata(self._file, self._data_group_path)
    def get_name(self):
        """
        Returns the name of the Analysis group.
        """
        return sync(get_object_name(self._file, self._path))

    @classmethod
    def _create_new(cls, data: 'Data', *, index: int, sparse: bool = False) -> 'AnalysisResults':
        """
        Create a new AnalysisResults group.

        Args:
            file (FileAbstraction): The file.
            index (int): The index for the new AnalysisResults group.

        Returns:
            AnalysisResults: The newly created AnalysisResults object.
        """
        group_name = f"{brim_obj_names.data.analysis_results}_{index}"
        ar_full_path = concatenate_paths(data._path, group_name)
        sync(data._file.create_group(ar_full_path))
        return cls(data._file, ar_full_path, data_group_path=data._path,
                    spatial_map=data._spatial_map, spatial_map_px_size=data._spatial_map_px_size,
                    sparse=sparse)

    def add_data(self, data_AntiStokes=None, data_Stokes=None, *,
                    fit_model: 'AnalysisResults.FitModel' = None):
        """
        Adds data for the analysis results for AntiStokes and Stokes peaks to the file.
        
        Args:
            data_AntiStokes (dict or list[dict]): A dictionary containing the analysis results for AntiStokes peaks.
                In case multiple peaks were fitted, it might be a list of dictionaries with each element corresponding to a single peak.
            
                Each dictionary may include the following keys (plus the corresponding units,  e.g. 'shift_units'):
                    - 'shift': The shift value.
                    - 'width': The width value.
                    - 'amplitude': The amplitude value.
                    - 'offset': The offset value.
                    - 'R2': The R-squared value.
                    - 'RMSE': The root mean square error value.
                    - 'Cov_matrix': The covariance matrix.
                The above arrays must have one less dimension than the PSD dataset, with the same shape as the first n-1 dimensions of the PSD (i.e. all the dimensions except the last (spectral) one).
                The 'Cov_matrix' should have 2 additional last dimensions which define the matrix.
            data_Stokes (dict or list[dict]): same as `data_AntiStokes` for the Stokes peaks.
            fit_model (AnalysisResults.FitModel, optional): The fit model used for the analysis. Defaults to None (no attribute is set).

            Both `data_AntiStokes` and `data_Stokes` are optional, but at least one of them must be provided.
        """

        ar_cls = self.__class__
        ar_group = sync(self._file.open_group(self._path))

        def add_quantity(qt: AnalysisResults.Quantity, pt: AnalysisResults.PeakType, data, index: int = 0):
            # PSD_nonspectral_shape is an closure variable that is used to check the shape of the data being added, if the PSD dataset is already present in the current data group.
            if PSD_nonspectral_shape is not None:
                expected_shape = PSD_nonspectral_shape
                if qt is AnalysisResults.Quantity.Cov_matrix:
                    expected_shape += (data.shape[-2], data.shape[-1])
                if data.shape != expected_shape:
                    raise ValueError(f"The shape of the '{qt.value}' data is {data.shape}, but it should be {expected_shape} to match the shape of the PSD.")
            sync(self._file.create_dataset(
                ar_group, ar_cls._get_quantity_name(qt, pt, index), data))

        def add_data_pt(pt: AnalysisResults.PeakType, data, index: int = 0):
            if 'shift' in data:
                add_quantity(ar_cls.Quantity.Shift,
                                pt, data['shift'], index)
                if 'shift_units' in data:
                    self._set_units(data['shift_units'],
                                    ar_cls.Quantity.Shift, pt, index)
            if 'width' in data:
                add_quantity(ar_cls.Quantity.Width,
                                pt, data['width'], index)
                if 'width_units' in data:
                    self._set_units(data['width_units'],
                                    ar_cls.Quantity.Width, pt, index)
            if 'amplitude' in data:
                add_quantity(ar_cls.Quantity.Amplitude,
                                pt, data['amplitude'], index)
                if 'amplitude_units' in data:
                    self._set_units(
                        data['amplitude_units'], ar_cls.Quantity.Amplitude, pt, index)
            if 'offset' in data:
                add_quantity(ar_cls.Quantity.Offset,
                                pt, data['offset'], index)
                if 'offset_units' in data:
                    self._set_units(
                        data['offset_units'], ar_cls.Quantity.Offset, pt, index)
            if 'R2' in data:
                add_quantity(ar_cls.Quantity.R2, pt, data['R2'], index)
                if 'R2_units' in data:
                    self._set_units(data['R2_units'],
                                    ar_cls.Quantity.R2, pt, index)
            if 'RMSE' in data:
                add_quantity(ar_cls.Quantity.RMSE, pt, data['RMSE'], index)
                if 'RMSE_units' in data:
                    self._set_units(data['RMSE_units'],
                                    ar_cls.Quantity.RMSE, pt, index)
            if 'Cov_matrix' in data:
                add_quantity(ar_cls.Quantity.Cov_matrix,
                                pt, data['Cov_matrix'], index)
                if 'Cov_matrix_units' in data:
                    self._set_units(
                        data['Cov_matrix_units'], ar_cls.Quantity.Cov_matrix, pt, index)

        PSD_nonspectral_shape = None
        try:
            PSD = sync(self._file.open_dataset(concatenate_paths(
                self._data_group_path, brim_obj_names.data.PSD)))
            PSD_nonspectral_shape = PSD.shape[:-1]
        except Exception as e:
            warnings.warn("It is recommended to add the PSD dataset before adding the analysis results, to ensure the correct shape of the analysis results data.")

        if data_AntiStokes is not None:
            data_AntiStokes = var_to_singleton(data_AntiStokes)
            for i, d_as in enumerate(data_AntiStokes):
                add_data_pt(ar_cls.PeakType.AntiStokes, d_as, i)
        if data_Stokes is not None:
            data_Stokes = var_to_singleton(data_Stokes)
            for i, d_s in enumerate(data_Stokes):
                add_data_pt(ar_cls.PeakType.Stokes, d_s, i)
        if fit_model is not None:
            sync(self._file.create_attr(ar_group, 'Fit_model', fit_model.value))

    def get_units(self, qt: Quantity, pt: PeakType = PeakType.AntiStokes, index: int = 0) -> str | None:
        """
        Retrieve the units of a specified quantity from the data file.

        Args:
            qt (Quantity): The quantity for which the units are to be retrieved.
            pt (PeakType, optional): The type of peak (e.g., Stokes or AntiStokes). Defaults to PeakType.AntiStokes.
            index (int, optional): The index of the quantity in case multiple quantities exist. Defaults to 0.

        Returns:
            str | None: The units of the specified quantity as a string, or None if no units are defined.
        """
        if qt in (AnalysisResults.Quantity.Elastic_contrast, AnalysisResults.Quantity.Viscous_contrast):
            return None
        dt_name = AnalysisResults._get_quantity_name(qt, pt, index)
        full_path = concatenate_paths(self._path, dt_name)
        return sync(units.of_object(self._file, full_path))

    def _set_units(self, un: str, qt: Quantity, pt: PeakType = PeakType.AntiStokes, index: int = 0) -> str:
        """
        Set the units of a specified quantity.

        Args:
            un (str): The units to be set.
            qt (Quantity): The quantity for which the units are to be set.
            pt (PeakType, optional): The type of peak (e.g., Stokes or AntiStokes). Defaults to PeakType.AntiStokes.
            index (int, optional): The index of the quantity in case multiple quantities exist. Defaults to 0.

        Returns:
            str: The units of the specified quantity as a string.
        """
        if qt in (AnalysisResults.Quantity.Elastic_contrast, AnalysisResults.Quantity.Viscous_contrast):
            raise ValueError(f"Units for {qt.name} are not settable because this quantity is computed on-the-fly.")
        dt_name = AnalysisResults._get_quantity_name(qt, pt, index)
        full_path = concatenate_paths(self._path, dt_name)
        return units.add_to_object(self._file, full_path, un)

    async def _compute_elastic_contrast_async(self, shift):
        shift_arr = np.asarray(shift)
        try:
            md = self._get_metadata()
            coros = [md._get_wavelength_nm_async(), md._get_temperature_c_async(), md._get_scattering_angle_deg_async()]
            res = await asyncio.gather(*coros, return_exceptions=True)
            wavelength_nm, temperature_c, scattering_angle_deg = res
            if isinstance(wavelength_nm, Exception):
                raise ValueError("Could not retrieve the wavelength for computing Elastic Contrast.")
            if isinstance(temperature_c, Exception):
                temperature_c = 22  # default value
                warnings.warn("Could not retrieve the temperature for computing Elastic Contrast. Using default value of 22 °C.")
            if isinstance(scattering_angle_deg, Exception):
                scattering_angle_deg = 180  # default value
                warnings.warn("Could not retrieve the scattering angle for computing Elastic Contrast. Using default value of 180 deg.")
            water_shift = Brillouin_shift_water(wavelength_nm, temperature_c, scattering_angle_deg)
            if np.nanmean(shift_arr) < 0:
                water_shift = -water_shift
            return shift_arr / water_shift - 1
        except Exception as e:
            raise ValueError(
                f"Could not compute Elastic_contrast from metadata ({e}).")

    async def _compute_viscous_contrast_async(self, width):
        width_arr = np.asarray(width)
        try:
            md = self._get_metadata()
            coros = [md._get_wavelength_nm_async(), md._get_temperature_c_async(), md._get_scattering_angle_deg_async()]
            res = await asyncio.gather(*coros, return_exceptions=True)
            wavelength_nm, temperature_c, scattering_angle_deg = res
            if isinstance(wavelength_nm, Exception):
                raise ValueError("Could not retrieve the wavelength for computing Viscous Contrast.")
            if isinstance(temperature_c, Exception):
                temperature_c = 22  # default value
                warnings.warn("Could not retrieve the temperature for computing Viscous Contrast. Using default value of 22 °C.")
            if isinstance(scattering_angle_deg, Exception):
                scattering_angle_deg = 180  # default value
                warnings.warn("Could not retrieve the scattering angle for computing Viscous Contrast. Using default value of 180 deg.")
            water_width = Brillouin_width_water(wavelength_nm, temperature_c, scattering_angle_deg)
            if np.nanmean(width_arr) < 0:
                water_width = -water_width
            return width_arr / water_width - 1
        except Exception as e:
            raise ValueError(
                f"Could not compute Viscous_contrast from metadata ({e}).")

    @property
    def fit_model(self) -> 'AnalysisResults.FitModel':
        """
        Retrieve the fit model used for the analysis.

        Returns:
            AnalysisResults.FitModel: The fit model used for the analysis.
        """
        if not hasattr(self, '_fit_model'):
            try:
                fit_model_str = sync(self._file.get_attr(self._path, 'Fit_model'))
                self._fit_model = AnalysisResults.FitModel(fit_model_str)
            except Exception as e:
                if isinstance(e, ValueError):
                    warnings.warn(
                        f"Unknown fit model '{fit_model_str}' found in the file.")
                self._fit_model = AnalysisResults.FitModel.Undefined        
        return self._fit_model

    def save_image_to_OMETiff(self, qt: Quantity, pt: PeakType = PeakType.AntiStokes, index: int = 0, filename: str = None) -> str:
        """
        Saves the image corresponding to the specified quantity and index to an OMETiff file.

        Args:
            qt (Quantity): The quantity to retrieve the image for (e.g. shift).
            pt (PeakType, optional): The type of peak to consider (default is PeakType.AntiStokes).
            index (int, optional): The index of the data to retrieve, if multiple are present (default is 0).
            filename (str, optional): The name of the file to save the image to. If None, a default name will be used.

        Returns:
            str: The path to the saved OMETiff file.
        """
        try:
            import tifffile
        except ImportError:
            raise ModuleNotFoundError(
                "The tifffile module is required for saving to OME-Tiff. Please install it using 'pip install tifffile'.")
        
        if filename is None:
            filename = f"{qt.value}_{pt.value}_{index}.ome.tif"
        if not filename.endswith('.ome.tif'):
            filename += '.ome.tif'
        img, px_size = self.get_image(qt, pt, index)
        if img.ndim > 3:
            raise NotImplementedError(
                "Saving images with more than 3 dimensions is not supported yet.")
        with tifffile.TiffWriter(filename, bigtiff=True) as tif:
            metadata = {
                'axes': 'ZYX',
                'PhysicalSizeX': px_size[2].value,
                'PhysicalSizeXUnit': px_size[2].units,
                'PhysicalSizeY': px_size[1].value,
                'PhysicalSizeYUnit': px_size[1].units,
                'PhysicalSizeZ': px_size[0].value,
                'PhysicalSizeZUnit': px_size[0].units,
            }
            tif.write(img, metadata=metadata)
        return filename

    def get_image(self, qt: Quantity, pt: PeakType = PeakType.AntiStokes, index: int = 0) -> tuple:
        """
        Retrieves an image (spatial map) based on the specified quantity, peak type, and index.

        Args:
            qt (Quantity): The quantity to retrieve the image for (e.g. shift).
            pt (PeakType, optional): The type of peak to consider (default is PeakType.AntiStokes).
            index (int, optional): The index of the data to retrieve, if multiple are present (default is 0).

        Returns:
            A tuple containing the image corresponding to the specified quantity and index and the corresponding pixel size.
            The image is a 3D dataset where the dimensions are z, y, x.
            If there are additional parameters, more dimensions are added in the order z, y, x, par1, par2, ...
            The pixel size is a tuple of 3 Metadata.Item in the order z, y, x.
        """
        if qt == AnalysisResults.Quantity.Elastic_contrast:
            shift_img, px_size = self.get_image(AnalysisResults.Quantity.Shift, pt, index)
            return sync(self._compute_elastic_contrast_async(shift_img)), px_size
        if qt == AnalysisResults.Quantity.Viscous_contrast:
            width_img, px_size = self.get_image(AnalysisResults.Quantity.Width, pt, index)
            return sync(self._compute_viscous_contrast_async(width_img)), px_size

        pt_type = AnalysisResults.PeakType
        data = None
        if pt == pt_type.average:
            peaks = self.list_existing_peak_types(index)
            match len(peaks):
                case 0:
                    raise ValueError(
                        "No peaks found for the specified index. Cannot compute average.")
                case 1:
                    data = np.array(sync(self._get_quantity(qt, peaks[0], index)))
                case 2:
                    data1, data2 = _gather_sync(
                        self._get_quantity(qt, peaks[0], index),
                        self._get_quantity(qt, peaks[1], index)
                        )
                    data = (np.abs(data1) + np.abs(data2))/2
        else:
            data = np.array(sync(self._get_quantity(qt, pt, index)))
        if self._sparse:
            sm = np.array(self._spatial_map)
            img = data[sm, ...]
            img[sm<0, ...] = np.nan  # set invalid pixels to NaN
        else:
            img = data
        return img, self._spatial_map_px_size
    def get_quantity_at_pixel(self, coord: tuple, qt: Quantity, pt: PeakType = PeakType.AntiStokes, index: int = 0):
        """
        Synchronous wrapper for `get_quantity_at_pixel_async` (see doc for `brimfile.analysis_results.AnalysisResults.get_quantity_at_pixel_async`)
        """
        return sync(self.get_quantity_at_pixel_async(coord, qt, pt, index))
    async def get_quantity_at_pixel_async(self, coord: tuple, qt: Quantity, pt: PeakType = PeakType.AntiStokes, index: int = 0):
        """
        Retrieves the specified quantity in the image at coord, based on the peak type and index.

        Args:
            coord (tuple): A tuple of 3 elements corresponding to the z, y, x coordinate in the image
            qt (Quantity): The quantity to retrieve the image for (e.g. shift).
            pt (PeakType, optional): The type of peak to consider (default is PeakType.AntiStokes).
            index (int, optional): The index of the data to retrieve, if multiple peaks are present (default is 0).

        Returns:
            The requested quantity, which is a scalar or a multidimensional array (depending on whether there are additional parameters in the current Data group)
        """
        if len(coord) != 3:
            raise ValueError(
                "'coord' must have 3 elements corresponding to z, y, x")
        if qt == AnalysisResults.Quantity.Elastic_contrast:
            shift_value = await self.get_quantity_at_pixel_async(coord, AnalysisResults.Quantity.Shift, pt, index)
            return await self._compute_elastic_contrast_async(shift_value)
        if qt == AnalysisResults.Quantity.Viscous_contrast:
            width_value = await self.get_quantity_at_pixel_async(coord, AnalysisResults.Quantity.Width, pt, index)
            return await self._compute_viscous_contrast_async(width_value)
        if self._sparse:
            i = self._spatial_map[*coord]
            assert i.size == 1
            if i<0:
                return np.nan  # invalid pixel
            i = (int(i), ...)
        else:
            i = coord + (...,)

        pt_type = AnalysisResults.PeakType
        value = None
        if pt == pt_type.average:
            value = None
            peaks = await self.list_existing_peak_types_async(index)
            match len(peaks):
                case 0:
                    raise ValueError(
                        "No peaks found for the specified index. Cannot compute average.")
                case 1:
                    data = await self._get_quantity(qt, peaks[0], index)
                    value = await _async_getitem(data, i)
                case 2:
                    data_p0, data_p1 = await asyncio.gather(
                        self._get_quantity(qt, peaks[0], index),
                        self._get_quantity(qt, peaks[1], index)
                    )
                    value1, value2 = await asyncio.gather(
                        _async_getitem(data_p0, i),
                        _async_getitem(data_p1, i)
                    )
                    value = (np.abs(value1) + np.abs(value2))/2
        else:
            data = await self._get_quantity(qt, pt, index)
            value = await _async_getitem(data, i)
        return value
    def get_all_quantities_in_image(self, coor: tuple, index_peak: int = 0) -> dict:
        """
        Retrieve all available quantities at a specific spatial coordinate.

        Args:
            coor (tuple): A tuple containing the z, y, x coordinates in the image.
            index_peak (int, optional): The index of the data to retrieve, if multiple peaks are present (default is 0).

        Returns:
            dict: A dictionary of Metadata.Item in the form `result[quantity.name][peak.name] = Metadata.Item(value, units)`.
                The dictionary contains all available quantities (e.g., Shift, Width, etc.) for both Stokes and AntiStokes peaks,
                as well as their average values.
        """
        if len(coor) != 3:
            raise ValueError("coor must contain 3 values for z, y, x")
        index = int(self._spatial_map[coor]) if self._sparse else coor
        return sync(self._get_all_quantities_at_index(index, index_peak))
    async def _get_all_quantities_at_index(self, index: int | tuple[int, int, int], index_peak: int = 0) -> dict:
        """
        Retrieve all available quantities for a specific spatial index.
        Args:
            index (int) | tuple[int, int, int]: The spatial index to retrieve quantities for, which can be a tuple for non-sparse data.
            index_peak (int, optional): The index of the data to retrieve, if multiple peaks are present (default is 0).
        Returns:
            dict: A dictionary of Metadata.Item in the form `result[quantity.name][peak.name] = bls.Metadata.Item(value, units)`
        """
        async def _get_existing_quantity_at_index_async(self,  index: int | tuple[int, int, int], pt: AnalysisResults.PeakType = AnalysisResults.PeakType.AntiStokes):
            as_cls = AnalysisResults
            qts_ls = ()
            dts_ls = ()

            qts = [qt for qt in as_cls.Quantity if qt not in (as_cls.Quantity.Elastic_contrast, as_cls.Quantity.Viscous_contrast)]
            coros = [self._file.open_dataset(concatenate_paths(self._path, as_cls._get_quantity_name(qt, pt, index_peak))) for qt in qts]
            
            # open the datasets asynchronously, excluding those that do not exist
            opened_dts = await asyncio.gather(*coros, return_exceptions=True)
            for i, opened_qt in enumerate(opened_dts):
                if not isinstance(opened_qt, Exception):
                    qts_ls += (qts[i],)
                    dts_ls += (opened_dts[i],)
            # get the values at the specified index
            if isinstance(index, tuple):
                index += (..., )
            else:
                index = (index, ...)
            coros_values = [_async_getitem(dt, index) for dt in dts_ls]
            coros_units = [units.of_object(self._file, dt) for dt in dts_ls]
            ret_ls = await asyncio.gather(*coros_values, *coros_units)
            n = len(coros_values)
            value_ls = [Metadata.Item(ret_ls[i], ret_ls[n+i]) for i in range(n)]
            return qts_ls, value_ls
        antiStokes, stokes = await asyncio.gather(
            _get_existing_quantity_at_index_async(self, index, AnalysisResults.PeakType.AntiStokes),
            _get_existing_quantity_at_index_async(self, index, AnalysisResults.PeakType.Stokes)
        )
        res = {}
        # combine the results, including the average
        for qt in (set(antiStokes[0]) | set(stokes[0])):
            res[qt.name] = {}
            pts = ()
            #Stokes
            if qt in stokes[0]:
                res[qt.name][AnalysisResults.PeakType.Stokes.name] = stokes[1][stokes[0].index(qt)]
                pts += (AnalysisResults.PeakType.Stokes,)
            #AntiStokes
            if qt in antiStokes[0]:
                res[qt.name][AnalysisResults.PeakType.AntiStokes.name] = antiStokes[1][antiStokes[0].index(qt)]
                pts += (AnalysisResults.PeakType.AntiStokes,)
            #average getting the units of the first peak
            res[qt.name][AnalysisResults.PeakType.average.name] = Metadata.Item(
                np.mean([np.abs(res[qt.name][pt.name].value) for pt in pts]), 
                res[qt.name][pts[0].name].units
                )
            if not all(res[qt.name][pt.name].units == res[qt.name][pts[0].name].units for pt in pts):
                warnings.warn(f"The units of {pts} are not consistent.")

        if AnalysisResults.Quantity.Shift.name in res:
            ec_name = AnalysisResults.Quantity.Elastic_contrast.name
            res[ec_name] = {}
            for pt_name, item in res[AnalysisResults.Quantity.Shift.name].items():
                ec = await self._compute_elastic_contrast_async(item.value)
                res[ec_name][pt_name] = Metadata.Item(ec, None)

        if AnalysisResults.Quantity.Width.name in res:
            vc_name = AnalysisResults.Quantity.Viscous_contrast.name
            res[vc_name] = {}
            for pt_name, item in res[AnalysisResults.Quantity.Width.name].items():
                vc = await self._compute_viscous_contrast_async(item.value)
                res[vc_name][pt_name] = Metadata.Item(vc, None)
        return res

    @classmethod
    def _get_quantity_name(cls, qt: Quantity, pt: PeakType, index: int) -> str:
        """
        Returns the name of the dataset correponding to the specific Quantity, PeakType and index

        Args:
            qt (Quantity)   
            pt (PeakType)  
            intex (int): in case of multiple peaks fitted, the index of the peak to consider       
        """
        if not pt in (cls.PeakType.AntiStokes, cls.PeakType.Stokes):
            raise ValueError("pt has to be either Stokes or AntiStokes")
        if qt in (cls.Quantity.Elastic_contrast, cls.Quantity.Viscous_contrast):
            raise ValueError(f"{qt.value} is a computed quantity and is not stored in the file.")
        if qt == cls.Quantity.R2 or qt == cls.Quantity.RMSE or qt == cls.Quantity.Cov_matrix:
            name = f"Fit_error_{str(pt.value)}_{index}/{str(qt.value)}"
        else:
            name = f"{str(qt.value)}_{str(pt.value)}_{index}"
        return name

    async def _get_quantity(self, qt: Quantity, pt: PeakType = PeakType.AntiStokes, index: int = 0):
        """
        Retrieve a specific quantity dataset from the file.

        Args:
            qt (Quantity): The type of quantity to retrieve.
            pt (PeakType, optional): The peak type to consider (default is PeakType.AntiStokes).
            index (int, optional): The index of the quantity if multiple peaks are available (default is 0).

        Returns:
            The dataset corresponding to the specified quantity, as stored in the file.

        """

        dt_name = AnalysisResults._get_quantity_name(qt, pt, index)
        full_path = concatenate_paths(self._path, dt_name)
        return await self._file.open_dataset(full_path)

    def list_existing_peak_types(self, index: int = 0) -> tuple:
        """
        Synchronous wrapper for `list_existing_peak_types_async` (see doc for `brimfile.analysis_results.AnalysisResults.list_existing_peak_types_async`)
        """
        return sync(self.list_existing_peak_types_async(index)) 
    async def list_existing_peak_types_async(self, index: int = 0) -> tuple:
        """
        Returns a tuple of existing peak types (Stokes and/or AntiStokes) for the specified index.
        Args:
            index (int, optional): The index of the peak to check (in case of multi-peak fit). Defaults to 0.
        Returns:
            tuple: A tuple containing `PeakType` members (`Stokes`, `AntiStokes`) that exist for the given index.
        """

        as_cls = AnalysisResults
        shift_s_name = as_cls._get_quantity_name(
            as_cls.Quantity.Shift, as_cls.PeakType.Stokes, index)
        shift_as_name = as_cls._get_quantity_name(
            as_cls.Quantity.Shift, as_cls.PeakType.AntiStokes, index)
        ls = ()
        coro_as_exists = self._file.object_exists(concatenate_paths(self._path, shift_as_name))
        coro_s_exists = self._file.object_exists(concatenate_paths(self._path, shift_s_name))
        as_exists, s_exists = await asyncio.gather(coro_as_exists, coro_s_exists)
        if as_exists:
            ls += (as_cls.PeakType.AntiStokes,)
        if s_exists:
            ls += (as_cls.PeakType.Stokes,)
        return ls

    def list_existing_quantities(self,  pt: PeakType = PeakType.AntiStokes, index: int = 0) -> tuple:
        """
        Synchronous wrapper for `list_existing_quantities_async` (see doc for `brimfile.analysis_results.AnalysisResults.list_existing_quantities_async`)
        """
        return sync(self.list_existing_quantities_async(pt, index))
    async def list_existing_quantities_async(self,  pt: PeakType = PeakType.AntiStokes, index: int = 0) -> tuple:
        """
        Returns a tuple of existing quantities for the specified index.
        Args:
            index (int, optional): The index of the peak to check (in case of multi-peak fit). Defaults to 0.
        Returns:
            tuple: A tuple containing `Quantity` members that exist for the given index.
        """
        as_cls = AnalysisResults
        ls = ()

        qts = [qt for qt in as_cls.Quantity if qt not in (as_cls.Quantity.Elastic_contrast, as_cls.Quantity.Viscous_contrast)]
        coros = [self._file.object_exists(concatenate_paths(self._path, as_cls._get_quantity_name(qt, pt, index))) for qt in qts]
        
        qt_exists = await asyncio.gather(*coros)
        for i, exists in enumerate(qt_exists):
            if exists:
                ls += (qts[i],)
        if as_cls.Quantity.Shift in ls:
            ls += (as_cls.Quantity.Elastic_contrast,)
        if as_cls.Quantity.Width in ls:
            ls += (as_cls.Quantity.Viscous_contrast,)
        return ls