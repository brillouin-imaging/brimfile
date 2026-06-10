__docformat__ = "google"

reserved_attr_names = ('Units', 'Name')


class brim_obj_names:
    """
    This class contains the names of the objects in the brim file.
    """
    Brillouin_base_path = 'Brillouin_data/'

    class data:
        """
        This class contains the names of the data groups in the brim file.
        """
        base_group = 'Data'
        PSD = 'PSD'
        frequency = 'Frequency'
        parameters = 'Parameters'
        analysis_results = 'Analysis'
        spatial_map = 'Scanning/Spatial_map'
        cartesian_visualisation = 'Scanning/Cartesian_visualisation'
        raw_data = 'Raw_data'
        calibration = 'Calibration'
