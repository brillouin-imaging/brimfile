"""
## What is brimfile?

*brimfile* is a Python library to read from and write to brim (**Br**illouin **im**aging) files,
which contain both the spectra and analysed data for Brillouin imaging.
More information about the brim file format can be found [here](https://github.com/brillouin-imaging/Brillouin-standard-file).

Briefly, a brim file can contain multiple data groups,
typically corresponding to imaging of the same sample at different timepoints/conditions.
Each data group contains the spectral data as well as the metadata and
the results of the analysis on the spectral data (which can be many in case multiple reconstruction pipelines are performed).

The structure of the *brimfile* library reflects the structure of the brim file and the user can access the data,
metadata and analysis results through their corresponding classes.

- [File](#file): represents a brim file, which can be opened or created.
- [Data](#data): represents a data group in the brim file, which contains the spectral data and metadata.
- [Metadata](#metadata): represents the metadata associated to a data group (or to the whole file).
- [AnalysisResults](#analysisresults): represents the results of the analysis of the spectral data.


## Install brimfile

We recommend installing *brimfile* in a [virtual environment](https://docs.python.org/3/library/venv.html).

After activating the new environment, simply run:

```bash
pip install brimfile
```

If you also need the support for exporting the analyzed data to OME-TIFF files,
you can install the optional dependencies with:

```bash
pip install "brimfile[export-tiff]"
```

For accessing remote data (i.e. S3 buckets), you need `remote-store`:

```bash
pip install "brimfile[remote-store]"
```

## Quickstart

The following code shows how to:
- open a .brim file 
- get an image for the Brillouin shift 
- get the spectrum at a specific pixel
- get the metadata.

```Python
from brimfile import File, Data, Metadata, AnalysisResults
Quantity = AnalysisResults.Quantity
PeakType = AnalysisResults.PeakType

filename = 'path/to/your/file.brim.zarr' 
f = File(filename)

# get the first data group in the file
d = f.get_data()

# get the first analysis results in the data group
ar = d.get_analysis_results()

# get the image for the shift
img, px_size = ar.get_image(Quantity.Shift, PeakType.average)

# get the spectrum at the pixel (pz,py,px)
(pz,py,px) = (0,0,0)
PSD, frequency, PSD_units, frequency_units = d.get_spectrum_in_image((pz,py,px))

# get the metadata 
md = d.get_metadata()
all_metadata = md.all_to_dict()

# close the file
f.close()
```

## Store types

Currently brimfile supports zip, zarr and S3 buckets as a store.
When opening or creating a file, the storage be selected by using the brimfile.file_abstraction.StoreType enum; zip and zarr can be used both for reading and writing while S3 only for reading. 

Although it is possible to write directly to zip, this will create duplicated entries in the archive (see [GitHub issue](https://github.com/zarr-developers/zarr-python/issues/1695)).

A possible workaround is to create a .zarr store instead and zip the folder afterwards.
Importantly the root of the archive should not contain the folder itself, i.e. you should go inside the .zarr folder, select all the elements there, right click on them to create a .zip archive.


## Use brimfile

### File

The main class is `brimfile.file.File`, which represents a brim file.
It can be used to create a new brim file (`brimfile.file.File.create`) or to open an existing one (`brimfile.file.File.__init__`).

```Python
import brimfile as brim

filename = 'path/to/your/file.brim.zarr'

# Open an existing brim file
f = brim.File(filename)

# or create a new one
f = brim.File.create(filename)
```

### Data

You can then get a `brimfile.data.Data` object representing the data group in the brim file
by opening an existing one (`brimfile.file.File.get_data`).

```Python
# Get the first data group in the file
data = f.get_data()
```

To add a new data group to the file, you can use the `brimfile.file.File.create_data_group` method,
which accepts a 4D array for the PSD with dimensions (z, y, x, spectrum),
a frequency array which might have the same size as PSD or be 1D, in case the frequency axis is the same for all the spectra.
```Python
# or create a new one
data = f.create_data_group(PSD, freq_GHz, (dz, dy, dx), name='my_data_group')
```
Alternatively you can use `brimfile.file.File.create_data_group_sparse` for sparse data, which lets you directly assign the correspondence
between the spatial positions and the spectra through the `scanning` dictionary.

Once you have an istance of `brimfile.data.Data`, you can get the spectrum corresponding to a pixel in the image
by calling the `brimfile.data.Data.get_spectrum_in_image` method:
```Python
PSD, frequency, PSD_units, frequency_units = data.get_spectrum_in_image((pz,py,px))    
```

### Metadata

You can then get a `brimfile.metadata.main.Metadata` object by simply calling the `brimfile.data.Data.get_metadata` method on a previously retrieved `Data` object.
The returned Metadata object contains all the metadata associated with the file and the specific data group.
```Python
metadata = data.get_metadata()
```
The list of available metadata is defined [here](https://github.com/brillouin-imaging/Brillouin-standard-file/blob/main/docs/brim_file_metadata.md) and can also be printed in the terminal with the `brimfile.metadata.schema.print_schema` method, which also allows to print the description of each metadata field:
```Python
brim.metadata.print_schema(include_description=True)
```
For metadata fields which require an enum, it can be imported from `brimfile.metadata`, e.g. `from brimfile.metadata import ImmersionMedium`.

New metadata can be added to the current data group (or to the whole file) by calling the `brimfile.metadata.main.Metadata.add` method.
```Python
import datetime

Attr = Metadata.Item
datetime_now = datetime.now().isoformat()
temp = Attr(22.0, 'C')
    
metadata.add(Metadata.Type.Experiment, {'Datetime':datetime_now, 'Temperature':temp},local=True)
```
When adding metadata fields which require an enum, the enum value or the string representation of the enum member (case-insensitive and ignoring underscores and spaces) can be used, e.g. `brim.metadata.SignalType.spontaneous` or 'spontaneous' can be used for the `Signal_type` field.

A single metadata item can be retrieved by indexing the `Metadata` object, which takes a string in the format 'group.object', e.g. 'Experiment.Datetime'.
```Python
datetime = metadata['Experiment.Datetime']
```
A dictionary containing all metadata can be retrieved by calling the `brimfile.metadata.main.Metadata.all_to_dict` method.
```Python
metadata.all_to_dict()
```

### AnalysisResults

The results of the analysis can be accessed through the `brimfile.analysis_results.AnalysisResults` object, obtained by calling the `brimfile.data.Data.get_analysis_results` method on a previously retrieved `Data` object:
``` Python
analysis_results = data.get_analysis_results()
```
or create a new one by calling the `brimfile.data.Data.create_analysis_results_group`:
``` Python
ar = data.create_analysis_results_group({'shift':shift_GHz, 'shift_units': 'GHz',
                                        'width': width_GHz, 'width_units': 'GHz'},
                                        {'shift':shift_GHz, 'shift_units': 'GHz',
                                        'width': width_GHz, 'width_units': 'GHz'},
                                        name = 'my_analysis_results')
```

`AnalysisResults` also exposes a method to retrieve the images of the analysis results (`brimfile.analysis_results.AnalysisResults.get_image`):

``` Python
ar_cls = AnalysisResults
img, px_size = analysis_results.get_image(ar_cls.Quantity.Shift, ar_cls.PeakType.average)
```

## List the contents of a brim file

The *brimfile* library provides methods to list the contents of a brim file.

To list all the data groups in a brim file, you can use the `brimfile.file.File.list_data_groups` method.

Once you have a `Data` object, you can list the analysis results in it by calling the `brimfile.data.Data.list_AnalysisResults` method.

Once you have an `AnalysisResults` object, you can determine:
- if the Stokes and/or anti-Stokes peaks are present by calling the `brimfile.analysis_results.AnalysisResults.list_existing_peak_types` method;
- the available quantities (e.g. shift, linewidth, etc...) in the analysis results by calling the `brimfile.analysis_results.AnalysisResults.list_existing_quantities` method.

## Example code

Here is a simple example which creates a brim file with a data group and some metadata and then reads it back.

We first write a function to generate some dummy data:

``` Python
import numpy as np

def generate_data():
    def lorentzian(x, x0, w):
        return 1/(1+((x-x0)/(w/2))**2)
    Nx, Ny, Nz = (7, 5, 3) # Number of points in x,y,z
    dx, dy, dz = (0.4, 0.5, 2) # Stepsizes (in um)
    n_points = Nx*Ny*Nz  # total number of points

    width_GHz = 0.4
    width_GHz_arr = np.full((Nz, Ny, Nx), width_GHz)
    shift_GHz_arr = np.empty((Nz, Ny, Nx))
    freq_GHz = np.linspace(6, 9, 151)  # 151 frequency points
    PSD = np.empty((Nz, Ny, Nx, len(freq_GHz)))
    for i in range(Nz):
        for j in range(Ny):
            for k in range(Nx):
                index = k + Nx*j + Ny*Nx*i
                #let's increase the shift linearly to have a readout 
                shift_GHz = freq_GHz[0] + (freq_GHz[-1]-freq_GHz[0]) * index/n_points
                spectrum = lorentzian(freq_GHz, shift_GHz, width_GHz)
                shift_GHz_arr[i,j,k] = shift_GHz 
                PSD[i, j, k,:] = spectrum

    return PSD, freq_GHz, (dz,dy,dx), shift_GHz_arr, width_GHz_arr
```

Now we can use this function to create a brim file with a data group and some metadata:

``` Python
    import brimfile as brim
    from brimfile import File, Data, Metadata, StoreType
    from datetime import datetime

    filename = 'path/to/your/file.brim.zarr' 

    f = File.create(filename, store_type=StoreType.AUTO)

    PSD, freq_GHz, (dz,dy,dx), shift_GHz, width_GHz = generate_data()
    
    d0 = f.create_data_group(PSD, freq_GHz, (dz,dy,dx), name='test1')
    
    # Create the metadata
    Attr = Metadata.Item
    datetime_now = datetime.now().isoformat()
    temp = Attr(22.0, 'C')
    md = d0.get_metadata()
    
    md.add(Metadata.Type.Experiment, {'Datetime':datetime_now, 'Temperature':temp})
    md.add(Metadata.Type.Optics, {'Wavelength':Attr(660, 'nm')})
    # enums can be added using the enum value or the string representation of the enum member (case-insensitive and ignoring underscores and spaces)
    md.add(Metadata.Type.Brillouin, {'Signal_type': brim.metadata.SignalType.spontaneous, 
                                          'Phonons_measured': 'longitudinal',})
    # Add some metadata to the local data group   
    temp = Attr(37.0, 'C')
    md.add(Metadata.Type.Experiment, {'Temperature':temp}, local=True)

    # create the analysis results
    ar = d0.create_analysis_results_group({'shift':shift_GHz, 'shift_units': 'GHz',
                                             'width': width_GHz, 'width_units': 'GHz'},
                                             {'shift':shift_GHz, 'shift_units': 'GHz',
                                             'width': width_GHz, 'width_units': 'GHz'},
                                             name = 'test1_analysis')
    f.close()
```
and we can read it back:
``` Python
    from brimfile import File, Data, Metadata, AnalysisResults

    filename = 'path/to/your/file.brim.zarr' 

    f = File(filename)

    # check if the file is read only
    f.is_read_only()

    #list all the data groups in the file
    data_groups = f.list_data_groups(retrieve_custom_name=True)

    # get the first data group in the file
    d = f.get_data()
    # get the name of the data group
    d.get_name()

    # get the number of parameters which the spectra depend on
    n_pars = d.get_num_parameters()

    # get the metadata 
    md = d.get_metadata()
    all_metadata = md.all_to_dict()
    # the list of metadata is defined here https://github.com/brillouin-imaging/Brillouin-standard-file/blob/main/docs/brim_file_metadata.md
    time = md['Experiment.Datetime']
    time.value
    time.units
    temp = md['Experiment.Temperature']
    md_dict = md.to_dict(Metadata.Type.Experiment)


    #get the list of analysis results in the data group
    ar_list = d.list_AnalysisResults(retrieve_custom_name=True)
    # get the first analysis results in the data group
    ar = d.get_analysis_results()
    # get the name of the analysis results
    ar.get_name()
    # list the existing peak types and quantities in the analysis results
    pt = ar.list_existing_peak_types()
    qt = ar.list_existing_quantities()
    # get the image of the shift quantity for the average of the Stokes and anti-Stokes peaks
    img, px_size = ar.get_image(AnalysisResults.Quantity.Shift, AnalysisResults.PeakType.average)
    # get the units of the shift quantity
    u = ar.get_units(AnalysisResults.Quantity.Shift)

    # get a quantity at a specific pixel (coord) in the image
    coord = (1,3,4)
    qt_at_px = ar.get_quantity_at_pixel(coord, AnalysisResults.Quantity.Shift, AnalysisResults.PeakType.average)
    assert img[coord]==qt_at_px
    
    # get the spectrum in the image at a specific pixel (coord)
    PSD, frequency, PSD_units, frequency_units = d.get_spectrum_in_image(coord)    

    f.close()
```

## Export the data to a different format

### OME-TIFF

You can export a specific quantity in the analyzed data to OME-TIFF files using the `brimfile.analysis_results.AnalysisResults.save_image_to_OMETiff` method on an instance `ar` of the `AnalysisResults` class.
``` Python
ar_cls = AnalysisResults
ar.save_image_to_OMETiff(ar_cls.Quantity.Shift, ar_cls.PeakType.average, filename='path/to/your/exported_tiff' )
```
"""

try:
    from ._version import __version__
except ImportError:
    __version__ = "unknown"

import os
# Default: normal imports enabled.
# Override with env var: BRIMFILE_IMPORT_VALIDATION_ONLY=1
_IMPORT_VALIDATION_ONLY = os.getenv("BRIMFILE_IMPORT_VALIDATION_ONLY", "").lower() in {
    "1", "true", "yes"
}

if not _IMPORT_VALIDATION_ONLY:
    from .file import File
    from .data import Data
    from .analysis_results import AnalysisResults
    from .calibration import Calibration
    from .metadata import Metadata
    from .file_abstraction import StoreType