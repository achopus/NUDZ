import os
import pandas as pd
from pandas import DataFrame
from numpy import ndarray

class Measurement:
    """Measurement dataclass used for loading, storing and formating measurements.
    """
    def __init__(self, path: str) -> None:
        self.id = None
        self.drug = None
        self.order = None
        self.time = None
        self.measurement_data = dict()

        # NOTE: Might be used later
        # self.dosage = None 
        # self.modalities = dict()
        # self.paths = list()

        self.load(path)

    def load(self, path: str) -> None:
        """Load basic information about the file and any measured data.

        Args:
            path (`str`): Relative path to the measurement (available formats: ['.dat'])
        """
        # File name handling - generate data headers
        assert os.path.exists(path)
        self.id, self.drug, self.order, _, self.time = path.split(sep='_')

        # Group all saline-based drugs together
        if 'sal' in self.drug:
            self.drug = 'placebo'

        self.id = os.path.split(self.id)[-1]
        self.order = int(self.order)
        self.time = self.time.split(sep='.')[0]

        # Data loading
        data = pd.read_csv(path, delimiter=' ')
        data = data.applymap(lambda x: float(x.replace(",", "."))) # Convert from "X,XX" into X.XX
        data_np = data.to_numpy()

        # Convert data into a dict
        sequence_length, channel_count = data.shape
        self.measurement_data['sequence_length'] = sequence_length
        self.measurement_data['channel_count'] = channel_count
        self.measurement_data['channel_names'] = data.columns
        for i, channel_name in enumerate(data.columns):
            self.measurement_data[channel_name] = data_np[:, i]

    def __eq__(self, search_terms) -> bool:
        """ Decide whether measurement matches search criteria.

        Args:
            search_terms (`dict`): Dictionary with with required attributes

        Returns:
            bool: `True` if all attributes match, `False` otherwise.
        """

        if type(search_terms) != Measurement:
            kv_pairs = search_terms
        else:
            kv_pairs = {'drug': search_terms.drug, 'id': search_terms.id, 'time': search_terms.time, 'order': search_terms.order}

        for key, value in kv_pairs.items():
            if self.__getattribute__(key) != value: return False
        return True
    
    def __getitem__(self, key:str) -> ndarray:
        return self.measurement_data[key]
    
    def __str__(self) -> str:
        str_return = ""
        str_return += f"ID = {self.id}\t\t| Drug: {self.drug}\t| Order: {self.order}\t\t| Time: {self.time}"
        
        # NOTE: Might use in the future
        # str_return += f"{self.measuring_intrument}\n"
        # str_return += f"{self.modalities}\n"

        return str_return
      

if __name__ == "__main__":
    path = "ID1_cocaine_7_ar1.i16_T0.dat"