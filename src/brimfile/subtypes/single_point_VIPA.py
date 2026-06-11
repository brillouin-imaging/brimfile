import numpy as np
import warnings

from typing import Any
from numpy.typing import NDArray

from .constants import SubType, FEATURES

from .. import Data, Calibration
from ..constants import brim_obj_names
from ..utils import concatenate_paths, _determine_chunk_size
from ..file_abstraction import sync, FileAbstraction

def _check_or_create_subtype(f: FileAbstraction):
    """
    Check that the data group subtype is correct, creating it if missing.
    """
    try:
        # Check if the subtype already stored in the file is correct
        subtype = sync(f.get_attr('/', 'Subtype'))
        if subtype != SubType.SinglePoint_VIPA_v0_1.value:
            raise ValueError(f"Invalid subtype: {subtype}. Expected {SubType.SinglePoint_VIPA_v0_1.value}")
    except KeyError:
        # If the Subtype attribute does not exist, create it
        sync(f.create_attr('/', 'Subtype', SubType.SinglePoint_VIPA_v0_1.value))

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
            sync(f.set_attr('/', 'Subtype_features', subtype_features))
    except KeyError:
        # If the Subtype_features attribute does not exist, create it with the given feature
        sync(f.create_attr('/', 'Subtype_features', [feature]))

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
    _check_or_create_subtype(data_group._file)
    _check_or_create_subtype_feature(data_group._file, '2DArray_per_spectrum')

    # check that the rawdata shape is compatible with the PSD shape if the latter is already present
    PSD_nonspectral_shape = _get_PSD_nonspectral_shape(data_group)
    if PSD_nonspectral_shape is not None:
        if (data_group._sparse and len(PSD_nonspectral_shape) != 1) or \
            (not data_group._sparse and len(PSD_nonspectral_shape) != 3):
            raise ValueError(f"Adding raw data is not supported when the PSD array has additional non-spectral dimensions (PSD non-spectral shape: {PSD_nonspectral_shape}, sparse: {data_group._sparse}).")
        if rawdata.ndim < len(PSD_nonspectral_shape) + 2 or rawdata.ndim > len(PSD_nonspectral_shape) + 3:
            raise ValueError(f"Invalid rawdata shape: expected {len(PSD_nonspectral_shape) + 2}  or {len(PSD_nonspectral_shape) + 3} dimensions, found {rawdata.ndim}")
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

        gr_m = sync(calibration_group._file.create_group(concatenate_paths(calibration_group._path, "Raw_data", str(m))))
        sync(calibration_group._file.create_dataset(gr_m, '2DArray_per_spectrum', data=data,
            chunk_size=_determine_chunk_size(data, 2),
            compression=compression))