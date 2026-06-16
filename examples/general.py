import numpy as np

import sys
import os
import shutil

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
import brimfile as brim

from datetime import datetime

filename = os.path.abspath(os.path.join(os.path.dirname(__file__), 'file.brim.zarr' ))

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


if __name__ == "__main__":
    #%% writing the test file 

    f = brim.File.create(filename, store_type=brim.StoreType.AUTO)

    # get the subtype of the file (should be None at this point since we haven't written any data yet)
    st = f.subtype

    PSD, freq_GHz, (dz,dy,dx), shift_GHz, width_GHz = generate_data()

    d0 = f.create_data_group(PSD, freq_GHz, (dz,dy,dx), name='test1')
    d1 = f.create_data_group(PSD, freq_GHz, (dz,dy,dx), name='test2')

    # add calibration groups to the data groups
    N = np.prod(PSD.shape[:-1])
    index = np.arange(N).reshape(PSD.shape[:-1])
    cal_d = {'spectra': np.empty((N, 50)), 'shift': 7.0, 'shift_units': 'GHz'}
    d0.create_calibration_group(index = index,
        calibration_data=[cal_d],
        attributes={'description': 'This is a test calibration group', 'Temperature': brim.Metadata.Item(22.0, 'C')})
    #test the same_as argument
    d1.create_calibration_group(same_as=0)
    d1.get_calibration() 
    
    # retrrieve calibration data
    c0 = d0.get_calibration()
    c0.get_spectrum_at_coor((1,2,3))

    # add raw data and calibration data in the format of the SinglePoint_VIPA subtype
    raw_data = np.tile(PSD[..., np.newaxis, np.newaxis, :], (1, 1, 1, 2, 17, 1))
    from brimfile.subtypes import single_point_VIPA
    single_point_VIPA.add_rawdata(d0, raw_data)
    single_point_VIPA.add_rawdata_calibration(c0, {0: np.empty((N, 23, 59))})
    single_point_VIPA.add_calibration_spectral_line(c0, np.empty((4)), linewidth = 4)

    # check the subtype of the file (should be SinglePoint_VIPA_v0.1 since we added raw data in the format of this subtype)
    st = f.subtype

    # print all the available metadata fields and their description
    brim.metadata.print_schema(True)

    # Create the metadata
    Attr = brim.Metadata.Item
    datetime_now = datetime.now().isoformat()
    temp = Attr(22.0, 'C')
    md = d0.get_metadata()

    md.add(brim.Metadata.Type.Experiment, {'Datetime':datetime_now, 'Temperature':temp})
    md.add(brim.Metadata.Type.Optics, {'Wavelength':Attr(660, 'nm')})
    # enums can be added using the enum value or the string representation of the enum member (case-insensitive and ignoring underscores and spaces)
    md.add(brim.Metadata.Type.Brillouin, {'Signal_type': brim.metadata.SignalType.spontaneous, 
                                          'Phonons_measured': 'longitudinal',})
    # Add some metadata to the local data group   
    temp = Attr(37.0, 'C')
    md.add(brim.Metadata.Type.Experiment, {'Temperature':temp}, local=True)

    # create the analysis results
    ar = d0.create_analysis_results_group({'shift':shift_GHz, 'shift_units': 'GHz',
                                             'width': width_GHz, 'width_units': 'Hz'},
                                             {'shift':shift_GHz, 'shift_units': 'GHz',
                                             'width': width_GHz, 'width_units': 'Hz'},
                                             name = 'test1_analysis',
                                             fit_model=brim.AnalysisResults.FitModel.Lorentzian)
    
    # add the spectral line to the analysis results in the format of the SinglePoint_VIPA subtype
    single_point_VIPA.add_analysis_results_spectral_line(ar, np.empty((4)))
    single_point_VIPA.get_raw_spectrum_in_image(d0, (1,2,3), analysis_results=ar)

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
    # the list of metadata is defined here https://github.com/prevedel-lab/Brillouin-standard-file/blob/main/docs/brim_file_metadata.md
    time = md['Experiment.Datetime']
    time.value
    time.units
    temp = md['Experiment.Temperature']
    md_dict = md.to_dict(brim.Metadata.Type.Experiment)
    # retrieve all the metadata, including validation and missing required fields
    all_dict = md.all_to_dict(validate=True, include_missing=True)


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
    # get the image of the shift quantity for the average of the Stokes and anti-Stokes peaks
    img, px_size = ar.get_image(brim.AnalysisResults.Quantity.Shift, brim.AnalysisResults.PeakType.average)
    # get the units of the shift quantity
    u = ar.get_units(brim.AnalysisResults.Quantity.Shift)

    # get a quantity at a specific pixel (coord) in the image
    coord = (1,3,4)
    qt_at_px = ar.get_quantity_at_pixel(coord, brim.AnalysisResults.Quantity.Shift, brim.AnalysisResults.PeakType.average)
    assert img[coord]==qt_at_px

    # get the spectrum in the image at a specific pixel (coord)
    PSD, frequency, PSD_units, frequency_units = d.get_spectrum_in_image(coord)    

    f.close()
    
    #%% deleting the test file 
    if os.path.isfile(filename):
        os.remove(filename)
    elif os.path.isdir(filename):
        shutil.rmtree(filename)