import os
from pathlib import PurePath
import sys
from time import time as t
import pickle

from measurement import Measurement

class Folder:
    def __init__(self, folder_path: str = 'data', save_path: str = 'saved_data.pickle') -> None:
        self.folder_path: str = folder_path
        self.measurements: list[Measurement] = list()
        self.load_()
        self.save_(save_path=save_path)

    def load_(self, verbose: bool = True) -> None:
        """Load all measurements inside of the folder.

        Args:
            verbose (bool, optional): Print status. Defaults to True.
        """
        files = [x for x in os.listdir(self.folder_path) if PurePath(x).suffix in ['.dat']]
        n_files = len(files)
        if verbose: print(f"Total number of files to be loaded: {n_files}")
        t0 = t()
        for i, file in enumerate(files):
            if verbose:
                p = 100 * i / n_files
                t1 = t()
                exp_rem_time = round((n_files - (i + 1)) * (t1 - t0) / (i + 1), 2)
                print(f"\rLoading data:\t{round(p, 2)}%\t({i + 1} / {n_files}) | Elapsed time: {round(t1 - t0, 2)} s | Expected remaining time: {exp_rem_time} s            ", end = '')
            file_name = os.path.join(self.folder_path, file)
            m = Measurement(file_name)
            self.measurements.append(m)
        
        if verbose: print()

    def save_(self, save_path: str, verbose: bool = True) -> None:
        if verbose:
            t0 = t()
            print(f"Saving into pickle file: {save_path} ...\t", end="")
        
        with open(save_path, "wb") as f:
            pickle.dump(self, f)
        
        if verbose:
            print(f"Object saved. (Elapsed time: {round(t() - t0, 2)} s)")


    def __len__(self) -> int:
        return len(self.measurements)
    
    def __getitem__(self, id: int) -> Measurement:
        return self.measurements[id]

if __name__ == "__main__":
    folder_load = Folder()
    t0 = t()
    with open("test.pickle", "rb") as f:
        folder_save = pickle.load(f)
    print()
    print(f"Loading took : {round(t() - t0, 2)} s")
    print()
    print(folder_save[0])