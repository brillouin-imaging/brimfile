from .file_abstraction import FileAbstraction

# do not include 'Same_as' in the list of standard attributes, as it needs to be handled separately in the code
_STANDARD_ATTRIBUTES = ['Datetime', 'Description', 'Temperature', 'FSR']

class Calibration:
    def __init__(self, file: FileAbstraction, full_path: str, *, 
                 data_group_path: str):
        """
        Initialize the Calibration object.

        Args:
            file (File): The parent File object.
            full_path (str): path of the group storing the analysis results
            data_group_path (str): path of the data group associated with the analysis results
        """
        self._file = file
        self._path = full_path
        self._data_group_path = data_group_path
    