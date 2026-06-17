from __future__ import annotations

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Any
    from numbers import Number
    from numpy.typing import NDArray

import numpy as np
import warnings
import asyncio

from .constants import SubType
from .utils import _check_or_create_subtype, _check_or_create_subtype_feature

from .. import Data, Calibration, AnalysisResults
from ..constants import brim_obj_names
from ..utils import concatenate_paths, _determine_chunk_size
from ..file_abstraction import sync, FileAbstraction, _async_getitem

def _get_PSD_nonspectral_shape(data_group: Data) -> tuple[int, ...] | None:
    try:
        PSD = sync(data_group._file.open_dataset(concatenate_paths(
            data_group._path, brim_obj_names.data.PSD)))
        return PSD.shape[:-1]
    except Exception as e:
        warnings.warn("It is recommended to add the PSD dataset before adding the raw data, to ensure the correct shape of the raw data.")
    return None


def add_rawdata(data_group: Data, rawdata: np.ndarray, *,
                compression: FileAbstraction.Compression = FileAbstraction.Compression()):
    """Add raw SinglePoint_VIPA data to the ``Raw_data/2DArray_per_spectrum`` dataset.

    This function ensures that root-level subtype metadata is set for
    ``SinglePoint_VIPA_v0_1`` and that the ``2DArray_per_spectrum`` feature is
    declared. If a PSD dataset is already present, ``rawdata`` is validated so
    its non-spectral dimensions match the PSD non-spectral shape.

    Parameters
    ----------
    data_group : Data
        Target data group where raw data will be stored.
    rawdata : numpy.ndarray
        Raw data array. Expected dimensionality is constrained by the PSD
        non-spectral shape (when available):
        ``len(PSD_non_spectral_shape) + 2`` or ``+ 3`` dimensions.
        See https://github.com/brillouin-imaging/Brillouin-standard-file/blob/main/docs/brim_file_subtypes.md#2darray_per_spectrum for details.
    compression : FileAbstraction.Compression, optional
        Compression settings used when creating the dataset.

    Raises
    ------
    ValueError
        If subtype metadata is incompatible or if ``rawdata`` shape/dimensions
        are not compatible with the existing PSD layout.
    """
    # make sure the attributes at the root level are correctly set for the SinglePoint_VIPA_v0.1 subtype
    _check_or_create_subtype(data_group._file, SubType.SinglePoint_VIPA_v0_1)
    _check_or_create_subtype_feature(data_group._file, '2DArray_per_spectrum')

    # check that the rawdata shape is compatible with the PSD shape if the latter is already present
    PSD_nonspectral_shape = _get_PSD_nonspectral_shape(data_group)
    if PSD_nonspectral_shape is not None:
        if (data_group._sparse and len(PSD_nonspectral_shape) != 1) or \
            (not data_group._sparse and len(PSD_nonspectral_shape) != 3):
            raise ValueError(f"Adding raw data is not supported when the PSD array has additional non-spectral dimensions (PSD non-spectral shape: {PSD_nonspectral_shape}, sparse: {data_group._sparse}).")
        if rawdata.ndim < len(PSD_nonspectral_shape) + 2 or rawdata.ndim > len(PSD_nonspectral_shape) + 3:
            raise ValueError(f"Invalid rawdata shape: expected {len(PSD_nonspectral_shape) + 2} or {len(PSD_nonspectral_shape) + 3} dimensions, found {rawdata.ndim}")
        if rawdata.shape[:len(PSD_nonspectral_shape)] != PSD_nonspectral_shape:
            raise ValueError(f"Invalid rawdata shape: the non-spectral dimensions {rawdata.shape[:len(PSD_nonspectral_shape)]} are not compatible with the PSD non-spectral shape {PSD_nonspectral_shape}")
    
    # create or open the Raw_data group
    raw_data_group = None
    raw_data_path = concatenate_paths(data_group._path, brim_obj_names.data.raw_data)
    try:
        raw_data_group = sync(data_group._file.open_group(raw_data_path))
    except Exception as e:
        # If the group does not exist, create it
        raw_data_group = sync(data_group._file.create_group(raw_data_path))

    # add the raw data to the file
    sync(data_group._file.create_dataset(raw_data_group,
        '2DArray_per_spectrum', data=rawdata,
        chunk_size=_determine_chunk_size(rawdata, 2),
        compression=compression))

def add_rawdata_calibration(calibration_group: Calibration, rawdata: NDArray | dict[int, Any], *,
                compression: FileAbstraction.Compression = FileAbstraction.Compression()):
    """Add raw SinglePoint_VIPA calibration data for each calibration material.

    This function requires that calibration spectra are already stored in the
    calibration group before raw data is added. It validates the provided raw
    data against the declared calibration materials and their array shapes.

    Parameters
    ----------
    calibration_group : Calibration
        Target calibration group where raw data will be stored.
    rawdata : numpy.ndarray or dict[int, Any]
        Raw calibration data. If a single calibration material is present, a
        plain array is accepted and will be mapped automatically. If multiple
        calibration materials are present, a dictionary mapping material indices
        to arrays must be provided.
    compression : FileAbstraction.Compression, optional
        Compression settings used when creating the dataset.

    Raises
    ------
    ValueError
        If no calibration materials are found in the calibration group, if
        multiple materials are present but ``rawdata`` is not a dictionary, if
        a provided material index is not found in the calibration group, or if
        the shape of a raw data array is incompatible with the corresponding
        calibration array.
    """
    # validate the rawdata input based on the calibration materials declared in the calibration group
    cal_mats = calibration_group.list_calibration_materials() 
    if len(cal_mats) == 0:
        raise ValueError("No calibration materials found in the calibration group. Please add at least one calibration material before adding raw data with calibration.")
    if len(cal_mats) > 1 and not isinstance(rawdata, dict):
        raise ValueError(f"Multiple calibration materials found in the calibration group, but rawdata is not provided as a dictionary with material indices as keys. \
                         Please provide rawdata as a dict[int, Any] where the keys are the calibration material indices corresponding to the spectra in the calibration group.")
    if len(cal_mats) == 1 and not isinstance(rawdata, dict):
        rawdata = {cal_mats[0]: rawdata}
    if len(cal_mats) != len(rawdata):
        warnings.warn(f"The number of calibration materials in the calibration group ({len(cal_mats)}) does not match the number of rawdata entries provided ({len(rawdata)}).")
    
    # add the raw data for each calibration material
    for m, data in rawdata.items():
        if m not in cal_mats:
            raise ValueError(f"Calibration material {m} not found in the calibration group. Available calibration materials: {cal_mats}")
        
        cal_arrs_shape = calibration_group._calibration_arrays[m].shape 
        if data.ndim != len(cal_arrs_shape) + 1:
            raise ValueError(f"Invalid rawdata shape for calibration material {m}: expected {len(cal_arrs_shape) + 1} dimensions, found {data.ndim}")
        if data.shape[0] != cal_arrs_shape[0]:
            raise ValueError(f"Invalid rawdata shape for calibration material {m}: the first dimension size {data.shape[0]} does not match the number of spectra in the calibration array {cal_arrs_shape[0]}")

        gr_m = sync(calibration_group._file.create_group(concatenate_paths(calibration_group._path, brim_obj_names.data.raw_data, str(m))))
        sync(calibration_group._file.create_dataset(gr_m, '2DArray_per_spectrum', data=data,
            chunk_size=_determine_chunk_size(data, 2),
            compression=compression))

def add_analysis_results_spectral_line(analysis_results: AnalysisResults, spectral_line: NDArray, *,
                                  linewidth: float | None = None, compression: FileAbstraction.Compression = FileAbstraction.Compression()):
    """Add spectral line information for the analysis results.

    Parameters
    ----------
    analysis_results : AnalysisResults
        Target analysis results where spectral line data will be stored.
    spectral_line : numpy.ndarray
        Spectral line data. The last dimension must have size 4, corresponding to ``(y_start, x_start, y_end, x_end)``.
        The first dimensions must match the spatial dimensions of the PSD (1 or 3 dimensions depending on whether the data is sparse).
        The spatial dimensions can be omitted if the same spectral line applies to all spectra in the analysis results.
    linewidth : float, optional
        Linewidth value to attach as an attribute to the spectral line dataset.
    compression : FileAbstraction.Compression, optional
        Compression settings used when creating the dataset.

    Raises
    ------
    ValueError
        If the last dimension of the spectral_line array is not 4, or if the number of dimensions is not 2 for sparse analysis results or 4 for non-sparse analysis results.
    """
    _check_or_create_subtype_feature(analysis_results._file, 'Spectral_line')

    if spectral_line.shape[-1] != 4:
        raise ValueError(f"The last dimension of the spectral_line array should have size 4, corresponding to (y_start, x_start, y_end, x_end). Found shape {spectral_line.shape}")
    if spectral_line.ndim > 1:
        if analysis_results._sparse and spectral_line.ndim != 2:
            raise ValueError(f"Invalid spectral_line shape: expected 2 dimensions for sparse analysis results, found {spectral_line.ndim}")
        if not analysis_results._sparse and spectral_line.ndim != 4:
            raise ValueError(f"Invalid spectral_line shape: expected 4 dimensions for non-sparse analysis results, found {spectral_line.ndim}")
    
    # TODO: check that the spatial dimensions of the spectral_line array are compatible with the PSD spatial dimensions
    
    sl_dt = sync(analysis_results._file.create_dataset(analysis_results._path, "Spectral_line", data=spectral_line,
            chunk_size=_determine_chunk_size(spectral_line),
            compression=compression))
    if linewidth is not None:
        sync(analysis_results._file.create_attr(sl_dt, 'Linewidth', linewidth))

def add_calibration_spectral_line(calibration_group: Calibration, spectral_line: NDArray | dict[int, Any], *,
                                  linewidth: float | None = None, compression: FileAbstraction.Compression = FileAbstraction.Compression()):
    """Add spectral line information for the calibration spectra.

    This function requires that raw calibration data is already stored in the
    calibration group before spectral line data is added. It validates the
    provided spectral line data against the declared calibration materials and
    their array shapes.

    Parameters
    ----------
    calibration_group : Calibration
        Target calibration group where spectral line data will be stored.
    spectral_line : numpy.ndarray or dict[int, Any]
        Spectral line data. Each entry must have its last dimension equal to 4,
        corresponding to ``(y_start, x_start, y_end, x_end)``. If a single
        calibration material is present, a plain array is accepted and will be
        mapped automatically. If multiple calibration materials are present, a
        dictionary mapping material indices to arrays must be provided. Arrays
        can be 1-D (single line shared across spectra) or 2-D (one line per
        spectrum).
    linewidth : float, optional
        Linewidth value to attach as an attribute to each spectral line dataset.
    compression : FileAbstraction.Compression, optional
        Compression settings used when creating the dataset.

    Raises
    ------
    ValueError
        If raw data has not been added to the calibration group yet, if no
        calibration materials are found, if multiple materials are present but
        ``spectral_line`` is not a dictionary, if a provided material index is
        not found in the calibration group, if the last dimension of a spectral
        line array is not 4, if the array has more than 2 dimensions, or if
        the first dimension size does not match the number of spectra in the
        corresponding calibration array.
    """

    _check_or_create_subtype_feature(calibration_group._file, 'Spectral_line')

    # validate the spectral_line input based on the calibration materials declared in the calibration group
    cal_mats = calibration_group.list_calibration_materials() 
    if len(cal_mats) == 0:
        raise ValueError("No calibration materials found in the calibration group. Please add at least one calibration material before adding raw data with calibration.")
    if len(cal_mats) > 1 and not isinstance(spectral_line, dict):
        raise ValueError(f"Multiple calibration materials found in the calibration group, but spectral_line is not provided as a dictionary with material indices as keys. \
                         Please provide spectral_line as a dict[int, Any] where the keys are the calibration material indices corresponding to the spectra in the calibration group.")
    if len(cal_mats) == 1 and not isinstance(spectral_line, dict):
        spectral_line = {cal_mats[0]: spectral_line}
    if len(cal_mats) != len(spectral_line):
        warnings.warn(f"The number of calibration materials in the calibration group ({len(cal_mats)}) does not match the number of spectral_line entries provided ({len(spectral_line)}).")
    
    # add the spectra line for each calibration material
    for m, sl in spectral_line.items():
        if m not in cal_mats:
            raise ValueError(f"Calibration material {m} not found in the calibration group. Available calibration materials: {cal_mats}")
        if sl.shape[-1] != 4:
            raise ValueError(f"The last dimension of the spectral_line array for calibration material {m} should have size 4, corresponding to (y_start, x_start, y_end, x_end). Found shape {sl.shape}")
        if sl.ndim > 2:
            raise ValueError(f"Invalid spectral_line shape for calibration material {m}: expected at most 2 dimensions, found {sl.ndim}")
        if sl.ndim == 2 and sl.shape[0] != calibration_group._calibration_arrays[m].shape[0]:
            raise ValueError(f"Invalid spectral_line shape for calibration material {m}: the first dimension size {sl.shape[0]} does not match the number of spectra in the calibration array {calibration_group._calibration_arrays[m].shape[0]}")
        try: 
            gr_m = sync(calibration_group._file.open_group(concatenate_paths(calibration_group._path, brim_obj_names.data.raw_data, str(m))))
        except Exception as e:
            raise ValueError(f"Raw data for calibration material {m} must be added before adding spectral line calibration data.") from e
        sl_dt = sync(calibration_group._file.create_dataset(gr_m, 'Spectral_line', data=sl,
            chunk_size=_determine_chunk_size(sl),
            compression=compression))
        if linewidth is not None:
            sync(calibration_group._file.create_attr(sl_dt, 'Linewidth', linewidth))

async def _get_spectral_line_in_image_from_calibration_async(calibration_group: Calibration, index: tuple, m: int = 0) -> tuple[NDArray | None, Number | None]:
    """
    Retrieve the spectral line and linewidth for a given spatial coordinate and calibration material.

    Args:
        calibration_group (Calibration): The calibration group containing the spectral line data.
        index (tuple): The index which can have 1 or 3 elements depending on whether the data is sparse.
        m (int): Calibration material index.

    Returns:
        tuple[NDArray | None, Number | None]: A tuple containing the spectral line array
        (y_start, x_start, y_end, x_end) and the associated linewidth. If no spectral
        line data is found, returns (None, None).
    """
    try:
        sl_arr = await calibration_group._file.open_dataset(concatenate_paths(calibration_group._path, brim_obj_names.data.raw_data, str(m), 'Spectral_line'))
    except Exception as e:
        return None, None
    if sl_arr.shape[-1] != 4:
        raise ValueError(f"The last dimension of the Spectral_line dataset for calibration material {m} should have size 4, corresponding to (y_start, x_start, y_end, x_end). Found shape {sl_arr.shape}")
    if sl_arr.ndim > 1:
        # if there are multiple spectral lines, we need to select the one corresponding to the current spectrum
        if calibration_group._index is None:
            raise ValueError(f"Calibration array for material {m} contains multiple spectra but no index dataset found in the calibration group.")
        spectrum_index = await _async_getitem(calibration_group._index, index)
        spectral_line = await _async_getitem(sl_arr, (int(spectrum_index),...))
    else:
        spectral_line = await _async_getitem(sl_arr, ...)
    linewidth = None
    try:
        linewidth = await calibration_group._file.get_attr(sl_arr, 'Linewidth')
    except Exception as e:
        # Linewidth metadata is optional; keep default `None` when the attribute is unavailable.
        pass
    return np.array(spectral_line), linewidth
    

async def _get_spectral_line_in_image_from_analysis_results_async(analysis_results: AnalysisResults, index: tuple) -> tuple[NDArray | None, Number | None]:
    """
    Retrieve the spectral line and linewidth for a given spatial coordinate

    Args:
        analysis_results (AnalysisResults): The analysis results containing the spectral line data.
        index (tuple): The index which can have 1 or 3 elements depending on whether the data is sparse.

    Returns:
        tuple[NDArray | None, Number | None]: A tuple containing the spectral line array
        (y_start, x_start, y_end, x_end) and the associated linewidth. If no spectral
        line data is found, returns (None, None).
    """
    spectral_line = None
    linewidth = None
    try:
        sl_arr = await analysis_results._file.open_dataset(concatenate_paths(analysis_results._path, "Spectral_line"))
    except Exception as e:
        return None, None
    if sl_arr.shape[-1] != 4:
        raise ValueError(f"The last dimension of the Spectral_line dataset in the analysis results should have size 4, corresponding to (y_start, x_start, y_end, x_end). Found shape {sl_arr.shape}")
    if sl_arr.ndim > 1:
        spectral_line = await _async_getitem(sl_arr, index+(...,))
    else:
        spectral_line = await _async_getitem(sl_arr, ...)
    try:
        linewidth = await analysis_results._file.get_attr(sl_arr, 'Linewidth')
    except Exception:
        # Linewidth metadata is optional; fall back to None when unavailable.
        linewidth = None
    return np.array(spectral_line), linewidth


async def get_raw_spectrum_in_image_async(data_group: Data, coor: tuple, *, 
                                          analysis_results: AnalysisResults = None) -> tuple:
    """
    Retrieve a raw spectrum together with the corresponding spectral line, if available,
    from the data group at the specified spatial coordinates.

    Args:
        coor (tuple): A tuple containing the z, y, x coordinates of the spectrum to retrieve.
        analysis_results (AnalysisResults, optional): Analysis results used to resolve the spectral line.
            If not provided, the function will attempt to retrieve the spectral line from the calibration group instead.

    Returns:
        tuple: (raw_spectrum, spectral_line, linewidth), where spectral_line and linewidth may be None.
        If no spectral line information is available, spectral_line and linewidth are returned as None.

    Raises:
        ValueError: If coor does not contain exactly 3 values.
        IndexError: If the coordinates are out of range for the raw spectrum dataset.
    """

    if len(coor) != 3:
            raise ValueError("coor must contain 3 values for z, y, x")

    index = coor
    if data_group._sparse:
        index = (int(data_group._spatial_map[coor]),)
    
    rawdata_arr = await data_group._file.open_dataset(concatenate_paths(data_group._path, brim_obj_names.data.raw_data, '2DArray_per_spectrum'))
    raw_spectrum_coro = _async_getitem(rawdata_arr, index +(...,))
    spectral_line_coro = None
    if analysis_results is not None:
        spectral_line_coro = _get_spectral_line_in_image_from_analysis_results_async(analysis_results, index)
    else:
        try:
            calibration_group = await data_group.get_calibration_async()
            # TODO: decide what to do if there are multiple calibration materials (which one to select?)
            spectral_line_coro = _get_spectral_line_in_image_from_calibration_async(calibration_group, index)
        except Exception as e:
            # the spectral line information is optional, so we can ignore errors related to its retrieval and just return None for the spectral line and linewidth when it is not available
            pass
    if spectral_line_coro is None:
        async def none_coro():
            return None, None
        spectral_line_coro = none_coro()
    raw_spectrum, spectral_line = await asyncio.gather(raw_spectrum_coro, spectral_line_coro)     
    
    return (raw_spectrum, ) + spectral_line

def get_raw_spectrum_in_image(data_group: Data, coor: tuple, *,
                               analysis_results: AnalysisResults = None) -> tuple:
    """
    Synchronous wrapper for get_raw_spectrum_in_image_async.
    """
    return sync(get_raw_spectrum_in_image_async(data_group, coor, analysis_results=analysis_results))