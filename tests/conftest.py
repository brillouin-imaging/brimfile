"""
Pytest configuration and shared fixtures for brimfile tests.
"""

import pytest
import numpy as np
import os
import shutil
import functools
import http.server
import threading
import urllib.parse
from datetime import datetime


import brimfile as brim


class _S3ListingCORSRequestHandler(http.server.SimpleHTTPRequestHandler):
    """Static file server with CORS headers that also emulates just enough of
    the S3 ListObjectsV2 ("list-type=2") API for `zarr_file.js`'s URL-based
    (S3) store listing to work against a plain local directory, since a real
    S3 bucket isn't available in tests.
    """

    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        super().end_headers()

    def log_message(self, format, *args):
        pass

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        query = urllib.parse.parse_qs(parsed.query)
        if 'list-type' in query:
            prefix = query.get('prefix', [''])[0]
            prefix = prefix.lstrip('/')
            self._serve_list_objects_v2(os.path.join(parsed.path.lstrip('/'), prefix))
            return
        super().do_GET()

    def _serve_list_objects_v2(self, prefix: str):
        prefix = prefix.rstrip('/')
        root = os.path.normpath(self.directory)
        target_dir = os.path.normpath(os.path.join(root, prefix))
        prefixes = []
        # guard against the request prefix escaping the served directory
        if (target_dir == root or target_dir.startswith(root + os.sep)) and os.path.isdir(target_dir):
            for name in sorted(os.listdir(target_dir)):
                if os.path.isdir(os.path.join(target_dir, name)):
                    key = f"{prefix}/{name}/" if prefix else f"{name}/"
                    prefixes.append(key)
        common_prefixes_xml = ''.join(
            f"<CommonPrefixes><Prefix>{p}</Prefix></CommonPrefixes>" for p in prefixes
        )
        xml = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            f"<ListBucketResult>{common_prefixes_xml}</ListBucketResult>"
        )
        body = xml.encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'application/xml')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def zarr_http_server(tmp_path):
    """Serve `tmp_path` over local HTTP with CORS + minimal S3-list emulation.

    Yields the base URL (e.g. 'http://127.0.0.1:PORT') that fixtures written
    under `tmp_path` (e.g. `simple_brim_file`) can be reached at.
    """
    handler = functools.partial(_S3ListingCORSRequestHandler, directory=str(tmp_path))
    httpd = http.server.ThreadingHTTPServer(('127.0.0.1', 0), handler)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()

    yield f"http://127.0.0.1:{port}"

    httpd.shutdown()
    thread.join()


@pytest.fixture(scope="session")
def sample_data():
    """Generate sample spectral data for testing."""
    def lorentzian(x, x0, w):
        return 1 / (1 + ((x - x0) / (w / 2)) ** 2)
    
    Nx, Ny, Nz = (7, 5, 3)  # Number of points in x, y, z
    dx, dy, dz = (0.4, 0.5, 2)  # Step sizes (in µm)
    n_points = Nx * Ny * Nz
    
    width_GHz = 0.4
    width_GHz_arr = np.full((Nz, Ny, Nx), width_GHz)
    shift_GHz_arr = np.empty((Nz, Ny, Nx))
    freq_GHz = np.linspace(6, 9, 151)  # 151 frequency points
    PSD = np.empty((Nz, Ny, Nx, len(freq_GHz)))
    
    for i in range(Nz):
        for j in range(Ny):
            for k in range(Nx):
                index = k + Nx * j + Ny * Nx * i
                # Increase shift linearly
                shift_GHz = freq_GHz[0] + (freq_GHz[-1] - freq_GHz[0]) * index / n_points
                spectrum = lorentzian(freq_GHz, shift_GHz, width_GHz)
                shift_GHz_arr[i, j, k] = shift_GHz
                PSD[i, j, k, :] = spectrum
    
    return {
        'PSD': PSD,
        'frequency': freq_GHz,
        'pixel_size': (dz, dy, dx),
        'shift': shift_GHz_arr,
        'width': width_GHz_arr,
        'dimensions': (Nz, Ny, Nx)
    }

@pytest.fixture(scope="session")
def sample_data_sparse(sample_data):
    """Generate sample spectral data for testing."""
    PSD = sample_data['PSD']
    frequency = sample_data['frequency']
    px_size_um = sample_data['pixel_size']
    PSD_flat = np.reshape(PSD, (-1, PSD.shape[3]))
    if frequency.ndim == 4:
        freq_flat = np.reshape(frequency, (-1, frequency.shape[3]))
    else:
        freq_flat = frequency
    def flatten_arr(arr):
        return np.reshape(arr, (-1,))
    shift_GHz_arr = flatten_arr(sample_data['shift'])
    width_GHz_arr = flatten_arr(sample_data['width'])
    indices = np.arange(PSD_flat.shape[0])
    cartesian_vis = np.reshape(indices, PSD.shape[0:3])
    scanning = {'Cartesian_visualisation': cartesian_vis,
                'Cartesian_visualisation_pixel': px_size_um, 'Cartesian_visualisation_pixel_unit': 'um'}
    
    return {
        'PSD': PSD_flat,
        'frequency': freq_flat,
        'pixel_size': px_size_um,
        'scanning': scanning,
        'shift': shift_GHz_arr,
        'width': width_GHz_arr
    }


@pytest.fixture
def simple_brim_file(tmp_path, sample_data):
    """Create a simple brim file for testing."""
    filename = os.path.join(tmp_path, 'test_file.brim.zarr')
    
    f = brim.File.create(filename, store_type=brim.StoreType.AUTO)
    
    # Create data group
    d = f.create_data_group(
        sample_data['PSD'],
        sample_data['frequency'],
        sample_data['pixel_size'],
        name='test_data'
    )
    
    # Add basic metadata
    Attr = brim.Metadata.Item
    datetime_now = datetime.now().isoformat()
    temp = Attr(22.0, 'C')
    md = d.get_metadata()
    md.add(brim.Metadata.Type.Experiment, {'Datetime': datetime_now, 'Temperature': temp})
    md.add(brim.Metadata.Type.Optics, {'Wavelength': Attr(660, 'nm')})
    md.add(brim.Metadata.Type.Brillouin, {'Scattering_angle': Attr(180, 'deg')})
    
    # Create analysis results
    ar = d.create_analysis_results_group(
        {
            'shift': sample_data['shift'],
            'shift_units': 'GHz',
            'width': sample_data['width'],
            'width_units': 'GHz'
        },
        {
            'shift': sample_data['shift'],
            'shift_units': 'GHz',
            'width': sample_data['width'],
            'width_units': 'GHz'
        },
        name='test_analysis',
        fit_model=brim.Data.AnalysisResults.FitModel.Lorentzian
    )
    
    f.close()
    
    yield filename
    
    # Cleanup
    if os.path.exists(filename):
        shutil.rmtree(filename)


@pytest.fixture
def simple_brim_file_sparse(tmp_path, sample_data_sparse):
    """Create a simple brim file with sparse data for testing."""
    filename = os.path.join(tmp_path, 'test_file_sparse.brim.zarr')
    
    f = brim.File.create(filename, store_type=brim.StoreType.AUTO)
    
    # Create sparse data group
    d = f.create_data_group_sparse(
        sample_data_sparse['PSD'],
        sample_data_sparse['frequency'],
        scanning=sample_data_sparse['scanning'],
        name='test_data_sparse'
    )
    
    # Add basic metadata
    Attr = brim.Metadata.Item
    datetime_now = datetime.now().isoformat()
    temp = Attr(22.0, 'C')
    md = d.get_metadata()
    md.add(brim.Metadata.Type.Experiment, {'Datetime': datetime_now, 'Temperature': temp})
    md.add(brim.Metadata.Type.Optics, {'Wavelength': Attr(660, 'nm')})
    md.add(brim.Metadata.Type.Brillouin, {'Scattering_angle': Attr(180, 'deg')})
    
    # Create analysis results for sparse data
    ar = d.create_analysis_results_group(
        {
            'shift': sample_data_sparse['shift'],
            'shift_units': 'GHz',
            'width': sample_data_sparse['width'],
            'width_units': 'GHz'
        },
        {
            'shift': sample_data_sparse['shift'],
            'shift_units': 'GHz',
            'width': sample_data_sparse['width'],
            'width_units': 'GHz'
        },
        name='test_analysis_sparse',
        fit_model=brim.Data.AnalysisResults.FitModel.Lorentzian
    )
    
    f.close()
    
    yield filename
    
    # Cleanup
    if os.path.exists(filename):
        shutil.rmtree(filename)


@pytest.fixture
def empty_brim_file(tmp_path):
    """Create an empty brim file for testing."""
    filename = os.path.join(tmp_path, 'empty_file.brim.zarr')
    f = brim.File.create(filename, store_type=brim.StoreType.AUTO)
    f.close()
    
    yield filename
    
    # Cleanup
    if os.path.exists(filename):
        shutil.rmtree(filename)
