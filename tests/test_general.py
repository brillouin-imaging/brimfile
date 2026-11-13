import os
import shutil
from datetime import datetime

import numpy as np

import brimfile as brim


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


def test_general(tmp_path):

    filename = os.path.abspath(os.path.join(tmp_path, 'file.brim.zarr' ))

    f = brim.File.create(filename, store_type=brim.StoreType.AUTO)

    PSD, freq_GHz, (dz,dy,dx), shift_GHz, width_GHz = generate_data()

    d0 = f.create_data_group(PSD, freq_GHz, (dz,dy,dx), name='test1')

    # Create the metadata
    Attr = brim.Metadata.Item
    datetime_now = datetime.now().isoformat()
    temp = Attr(22.0, 'C')
    md = d0.get_metadata()

    md.add(brim.Metadata.Type.Experiment, {'Datetime':datetime_now, 'Temperature':temp})
    md.add(brim.Metadata.Type.Optics, {'Wavelength':Attr(660, 'nm')})
    # Add some metadata to the local data group   
    temp = Attr(37.0, 'C')
    md.add(brim.Metadata.Type.Experiment, {'Temperature':temp}, local=True)

    # create the analysis results
    ar = d0.create_analysis_results_group({'shift':shift_GHz, 'shift_units': 'GHz',
                                             'width': width_GHz, 'width_units': 'Hz'},
                                             {'shift':shift_GHz, 'shift_units': 'GHz',
                                             'width': width_GHz, 'width_units': 'Hz'},
                                             name = 'test1_analysis',
                                             fit_model=brim.Data.AnalysisResults.FitModel.Lorentzian)
    f.close()


    #%% reading the test file 

    f = brim.File(filename)

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
    md_dict = md.to_dict(brim.Metadata.Type.Experiment)


    #get the list of analysis results in the data group
    ar_list = d.list_AnalysisResults(retrieve_custom_name=True)
    # get the first analysis results in the data group
    ar = d.get_analysis_results()
    # get the name of the analysis results
    ar.get_name()
    # get the fit model
    ar.fit_model
    # list the existing peak types and quantities in the analysis results
    pt = ar.list_existing_peak_types()
    qt = ar.list_existing_quantities()
    assert brim.Data.AnalysisResults.Quantity.Viscous_contrast in qt
    # get the image of the shift quantity for the average of the Stokes and anti-Stokes peaks
    img, px_size = ar.get_image(brim.Data.AnalysisResults.Quantity.Shift, brim.Data.AnalysisResults.PeakType.average)
    # get the units of the shift quantity
    u = ar.get_units(brim.Data.AnalysisResults.Quantity.Shift)

    # get a quantity at a specific pixel (coord) in the image
    coord = (1,3,4)
    qt_at_px = ar.get_quantity_at_pixel(coord, brim.Data.AnalysisResults.Quantity.Shift, brim.Data.AnalysisResults.PeakType.average)
    assert img[coord]==qt_at_px

    # get the spectrum in the image at a specific pixel (coord)
    PSD, frequency, PSD_units, frequency_units = d.get_spectrum_in_image(coord)    

    f.close()
    
    #%% deleting the test file 
    if os.path.isfile(filename):
        os.remove(filename)
    elif os.path.isdir(filename):
        shutil.rmtree(filename)