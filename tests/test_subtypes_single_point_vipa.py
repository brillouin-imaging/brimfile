"""Unit tests for SinglePoint_VIPA subtype helpers."""

import numpy as np
import pytest

import brimfile as brim
from brimfile.file_abstraction import sync
from brimfile.utils import concatenate_paths
from brimfile.constants import brim_obj_names
from brimfile.subtypes.constants import SubType
from brimfile.subtypes import single_point_VIPA as spv


@pytest.mark.parametrize(
    "file_fixture,is_sparse",
    [("simple_brim_file", False), ("simple_brim_file_sparse", True)],
)
def test_add_rawdata_sets_subtype_and_feature(request, file_fixture, is_sparse):
    """add_rawdata stores raw data and initializes subtype attributes."""
    f = brim.File(request.getfixturevalue(file_fixture), mode="r+")
    data = f.get_data()

    PSD = sync(f._file.open_dataset(concatenate_paths(data._path, brim_obj_names.data.PSD)))
    spatial_shape = PSD.shape[:-1]
    if is_sparse:
        rawdata = np.arange(spatial_shape[0] * 6, dtype=np.float32).reshape(spatial_shape[0], 2, 3)
    else:
        rawdata = np.arange(np.prod(spatial_shape) * 6, dtype=np.float32).reshape(*spatial_shape, 2, 3)

    spv.add_rawdata(data, rawdata)

    raw_path = concatenate_paths(data._path, "Raw_data", "2DArray_per_spectrum")
    raw_ds = sync(f._file.open_dataset(raw_path))
    subtype = sync(f._file.get_attr("/", "Subtype"))
    features = sync(f._file.get_attr("/", "Subtype_features"))

    assert subtype == SubType.SinglePoint_VIPA_v0_1.value
    assert "2DArray_per_spectrum" in features
    np.testing.assert_allclose(np.array(raw_ds), rawdata)

    f.close()


@pytest.mark.parametrize(
    "file_fixture,is_sparse",
    [("simple_brim_file", False), ("simple_brim_file_sparse", True)],
)
def test_add_rawdata_rejects_shape_mismatch_against_psd(request, file_fixture, is_sparse):
    """add_rawdata validates non-spectral shape against PSD layout."""
    f = brim.File(request.getfixturevalue(file_fixture), mode="r+")
    data = f.get_data()

    PSD = sync(f._file.open_dataset(concatenate_paths(data._path, brim_obj_names.data.PSD)))
    spatial_shape = PSD.shape[:-1]
    if is_sparse:
        bad_rawdata = np.zeros((spatial_shape[0] - 1, 2, 3), dtype=np.float32)
    else:
        bad_rawdata = np.zeros((spatial_shape[0], spatial_shape[1], spatial_shape[2] - 1, 2, 3), dtype=np.float32)

    with pytest.raises(ValueError, match="non-spectral dimensions"):
        spv.add_rawdata(data, bad_rawdata)

    f.close()


@pytest.mark.parametrize(
    "file_fixture,is_sparse",
    [("simple_brim_file", False), ("simple_brim_file_sparse", True)],
)
def test_add_analysis_results_spectral_line_stores_dataset(request, file_fixture, is_sparse):
    """add_analysis_results_spectral_line supports sparse and non-sparse layouts."""
    f = brim.File(request.getfixturevalue(file_fixture), mode="r+")
    data = f.get_data()
    analysis_results = data.get_analysis_results()

    PSD = sync(f._file.open_dataset(concatenate_paths(data._path, brim_obj_names.data.PSD)))
    spatial_shape = PSD.shape[:-1]
    if is_sparse:
        spectral_line = np.tile(np.array([0, 1, 2, 3], dtype=np.int32), (spatial_shape[0], 1))
    else:
        spectral_line = np.zeros((*spatial_shape, 4), dtype=np.int32)
        spectral_line[1, 2, 3] = np.array([3, 4, 5, 6], dtype=np.int32)

    spv.add_analysis_results_spectral_line(analysis_results, spectral_line, linewidth=0.55)

    sl_path = concatenate_paths(analysis_results._path, "Spectral_line")
    sl_ds = sync(f._file.open_dataset(sl_path))
    linewidth = sync(f._file.get_attr(sl_ds, "Linewidth"))

    np.testing.assert_array_equal(np.array(sl_ds), spectral_line)
    assert linewidth == 0.55

    f.close()


@pytest.mark.parametrize(
    "file_fixture,is_sparse,bad_shape,error_msg",
    [
        ("simple_brim_file", False, (3, 4), "expected 4 dimensions"),
        ("simple_brim_file_sparse", True, (2, 2, 4), "expected 2 dimensions"),
    ],
)
def test_add_analysis_results_spectral_line_validates_ndim(
    request, file_fixture, is_sparse, bad_shape, error_msg
):
    """add_analysis_results_spectral_line enforces sparse/non-sparse dimensionality."""
    f = brim.File(request.getfixturevalue(file_fixture), mode="r+")
    data = f.get_data()
    analysis_results = data.get_analysis_results()

    bad_spectral_line = np.zeros(bad_shape, dtype=np.int32)
    with pytest.raises(ValueError, match=error_msg):
        spv.add_analysis_results_spectral_line(analysis_results, bad_spectral_line)

    f.close()


@pytest.mark.parametrize("is_sparse", [False, True])
def test_add_calibration_rawdata_and_spectral_line(is_sparse, empty_brim_file, sample_data, sample_data_sparse):
    """Calibration raw data and spectral line can be added for both data layouts."""
    f = brim.File(empty_brim_file, mode="r+")

    if is_sparse:
        data = f.create_data_group_sparse(
            sample_data_sparse["PSD"],
            sample_data_sparse["frequency"],
            scanning=sample_data_sparse["scanning"],
        )
        n_points = sample_data_sparse["PSD"].shape[0]
        index = np.arange(n_points, dtype=np.int32) % 2
        spectra = np.stack(
            [sample_data_sparse["PSD"][0, :], sample_data_sparse["PSD"][1, :]],
            axis=0,
        )
    else:
        data = f.create_data_group(
            sample_data["PSD"],
            sample_data["frequency"],
            sample_data["pixel_size"],
        )
        index = np.zeros(sample_data["dimensions"], dtype=np.int32)
        index[1, 2, 3] = 1
        spectra = np.stack(
            [sample_data["PSD"][0, 0, 0, :], sample_data["PSD"][1, 2, 3, :]],
            axis=0,
        )

    data.create_calibration_group(
        index=index,
        calibration_data=[{"spectra": spectra, "shift": 7.0, "shift_units": "GHz"}],
    )

    calibration = data.get_calibration()
    rawdata_cal = np.arange(2 * 6, dtype=np.float32).reshape(2, 2, 3)
    spv.add_rawdata_calibration(calibration, rawdata_cal)

    spectral_line_cal = np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int32)
    spv.add_calibration_spectral_line(calibration, spectral_line_cal, linewidth=0.6)

    base_path = concatenate_paths(calibration._path, "Raw_data", "0")
    raw_ds = sync(f._file.open_dataset(concatenate_paths(base_path, "2DArray_per_spectrum")))
    sl_ds = sync(f._file.open_dataset(concatenate_paths(base_path, "Spectral_line")))
    linewidth = sync(f._file.get_attr(sl_ds, "Linewidth"))

    np.testing.assert_array_equal(np.array(raw_ds), rawdata_cal)
    np.testing.assert_array_equal(np.array(sl_ds), spectral_line_cal)
    assert linewidth == 0.6

    f.close()


@pytest.mark.parametrize(
    "file_fixture,is_sparse,coor",
    [("simple_brim_file", False, (1, 2, 3)), ("simple_brim_file_sparse", True, (0, 0, 1))],
)
def test_get_raw_spectrum_in_image_with_analysis_results(request, file_fixture, is_sparse, coor):
    """get_raw_spectrum_in_image returns raw spectrum and spectral line from analysis results."""
    f = brim.File(request.getfixturevalue(file_fixture), mode="r+")
    data = f.get_data()
    analysis_results = data.get_analysis_results()

    if is_sparse:
        PSD = sync(f._file.open_dataset(concatenate_paths(data._path, brim_obj_names.data.PSD)))
        n_points = PSD.shape[0]
        rawdata = np.zeros((n_points, 2, 3), dtype=np.float32)
        spectral_line = np.zeros((n_points, 4), dtype=np.int32)
        idx = int(data._spatial_map[coor])
        rawdata[idx] = np.array([[10, 11, 12], [13, 14, 15]], dtype=np.float32)
        spectral_line[idx] = np.array([2, 4, 6, 8], dtype=np.int32)
    else:
        PSD = sync(f._file.open_dataset(concatenate_paths(data._path, brim_obj_names.data.PSD)))
        spatial_shape = PSD.shape[:-1]
        rawdata = np.zeros((*spatial_shape, 2, 3), dtype=np.float32)
        spectral_line = np.zeros((*spatial_shape, 4), dtype=np.int32)
        rawdata[coor] = np.array([[20, 21, 22], [23, 24, 25]], dtype=np.float32)
        spectral_line[coor] = np.array([1, 3, 5, 7], dtype=np.int32)

    spv.add_rawdata(data, rawdata)
    spv.add_analysis_results_spectral_line(analysis_results, spectral_line, linewidth=0.7)

    raw_spectrum, line, linewidth = spv.get_raw_spectrum_in_image(
        data,
        coor,
        analysis_results=analysis_results,
    )

    if is_sparse:
        np.testing.assert_array_equal(raw_spectrum, rawdata[idx])
        np.testing.assert_array_equal(line, spectral_line[idx])
    else:
        np.testing.assert_array_equal(raw_spectrum, rawdata[coor])
        np.testing.assert_array_equal(line, spectral_line[coor])
    assert linewidth == 0.7

    f.close()


def test_get_raw_spectrum_in_image_without_spectral_line_source(empty_brim_file, sample_data):
    """When no analysis results or calibration exist, spectral line data should be None."""
    f = brim.File(empty_brim_file, mode="r+")
    data = f.create_data_group(
        sample_data["PSD"],
        sample_data["frequency"],
        sample_data["pixel_size"],
    )

    rawdata = np.zeros((*sample_data["dimensions"], 2, 3), dtype=np.float32)
    coor = (1, 2, 3)
    rawdata[coor] = np.array([[30, 31, 32], [33, 34, 35]], dtype=np.float32)
    spv.add_rawdata(data, rawdata)

    raw_spectrum, line, linewidth = spv.get_raw_spectrum_in_image(data, coor)

    np.testing.assert_array_equal(raw_spectrum, rawdata[coor])
    assert line is None
    assert linewidth is None

    f.close()


def test_get_raw_spectrum_in_image_supports_1d_analysis_results_line(empty_brim_file, sample_data):
    """A 1-D Spectral_line dataset should be returned as a shared line for all spectra."""
    f = brim.File(empty_brim_file, mode="r+")
    data = f.create_data_group(
        sample_data["PSD"],
        sample_data["frequency"],
        sample_data["pixel_size"],
    )

    rawdata = np.zeros((*sample_data["dimensions"], 2, 3), dtype=np.float32)
    coor = (1, 2, 3)
    rawdata[coor] = np.array([[40, 41, 42], [43, 44, 45]], dtype=np.float32)
    spv.add_rawdata(data, rawdata)

    analysis_results = data.create_analysis_results_group(
        {
            "shift": sample_data["shift"],
            "shift_units": "GHz",
            "width": sample_data["width"],
            "width_units": "GHz",
        },
        {
            "shift": sample_data["shift"],
            "shift_units": "GHz",
            "width": sample_data["width"],
            "width_units": "GHz",
        },
        fit_model=brim.Data.AnalysisResults.FitModel.Lorentzian,
    )
    spectral_line = np.array([1, 3, 5, 7], dtype=np.int32)
    spv.add_analysis_results_spectral_line(analysis_results, spectral_line, linewidth=0.9)

    raw_spectrum, line, linewidth = spv.get_raw_spectrum_in_image(
        data,
        coor,
        analysis_results=analysis_results,
    )

    np.testing.assert_array_equal(raw_spectrum, rawdata[coor])
    np.testing.assert_array_equal(line, spectral_line)
    assert linewidth == 0.9

    f.close()


def test_get_raw_spectrum_in_image_with_calibration_fallback(empty_brim_file, sample_data):
    """Without analysis results, spectral line is resolved from calibration raw-data groups."""
    f = brim.File(empty_brim_file, mode="r+")
    data = f.create_data_group(
        sample_data["PSD"],
        sample_data["frequency"],
        sample_data["pixel_size"],
    )

    rawdata = np.zeros((*sample_data["dimensions"], 2, 3), dtype=np.float32)
    coor = (1, 2, 3)
    rawdata[coor] = np.array([[30, 31, 32], [33, 34, 35]], dtype=np.float32)
    spv.add_rawdata(data, rawdata)

    index = np.zeros(sample_data["dimensions"], dtype=np.int32)
    index[coor] = 1
    spectra = np.stack(
        [sample_data["PSD"][0, 0, 0, :], sample_data["PSD"][1, 2, 3, :]],
        axis=0,
    )
    data.create_calibration_group(
        index=index,
        calibration_data=[{"spectra": spectra, "shift": 7.0, "shift_units": "GHz"}],
    )
    calibration = data.get_calibration()

    rawdata_cal = np.arange(2 * 6, dtype=np.float32).reshape(2, 2, 3)
    spv.add_rawdata_calibration(calibration, rawdata_cal)
    spectral_line_cal = np.array([[0, 1, 2, 3], [9, 8, 7, 6]], dtype=np.int32)
    spv.add_calibration_spectral_line(calibration, spectral_line_cal, linewidth=0.8)

    raw_spectrum, line, linewidth = spv.get_raw_spectrum_in_image(data, coor)

    np.testing.assert_array_equal(raw_spectrum, rawdata[coor])
    np.testing.assert_array_equal(line, spectral_line_cal[1])
    assert linewidth == 0.8

    f.close()
