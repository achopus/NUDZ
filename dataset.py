import numpy as np
from numpy import ndarray
import pandas as pd
from pandas import DataFrame

from folder import Folder
from measurement import Measurement
from pathlib import PurePath
import os
from time import time as t
from constants import keys, channels, fs
import pickle
import matplotlib.pyplot as plt

from scipy.signal import welch, butter, spectrogram
from scipy import signal
from collections import OrderedDict


class Dataset:
    def __init__(self, path: str, is_loaded: bool) -> None:
        if is_loaded:
            with open(path, 'rb') as f:
                self.folder: Folder = pickle.load(f)
        else:
            self.folder = Folder(path)


        # Group all saline-based substance under a single class
        for i, measurement in enumerate(self.folder):
            if 'sal' in measurement.drug and 'canab' not in measurement.drug:
                self.folder[i].drug = 'placebo'

        
    def select(self, **kwargs) -> list[Measurement]:
        """Select measurements which match given search terms

        Args:
            `**kwargs`: Search tems in a form of a `key:value` pairs
        
        Returns:
            `list[Measurement]`: List of all measurements matching search terms
        """
        for key in kwargs.keys(): assert key in keys
        selected = [m for m in self.folder.measurements if m == kwargs]
        return selected
    
    def sumarize_basic_stats(self, visualize:bool = True) -> None:
        """Get and visual summary of the dataset. ({subject, drug, order, time}-count)"""
        ids = set()
        drugs = set()
        orders = set()
        times = set()

        for measurement in self.folder.measurements:
            ids.add(measurement.id)
            drugs.add(measurement.drug)
            orders.add(measurement.order)
            times.add(measurement.time)
        
        id_counts = np.zeros(len(ids), dtype=int)
        for i, ix in enumerate(ids): id_counts[i] = len(self.select(id=ix))

        drug_counts = np.zeros(len(drugs), dtype=int)
        for i, ix in enumerate(drugs): drug_counts[i] = len(self.select(drug=ix))

        order_counts = np.zeros(len(orders), dtype=int)
        for i, ix in enumerate(orders): order_counts[i] = len(self.select(order=ix))

        time_counts = np.zeros(len(times), dtype=int)
        for i, ix in enumerate(times): time_counts[i] = len(self.select(time=ix))

        self.ids:list  = list(ids)
        self.drugs:list = list(drugs)
        self.orders:list = list(orders)
        self.times:list = list(times)

        self.id_counts:ndarray = id_counts
        self.drug_counts:ndarray = drug_counts
        self.order_counts:ndarray = order_counts
        self.time_counts:ndarray = time_counts

        if not visualize: return

        plt.figure("Basic dataset statistics")
        
        plt.subplot(2, 2, 1)
        plt.bar(list(ids), id_counts)
        plt.title("Subject Counts")
        
        plt.subplot(2, 2, 2)
        plt.bar(list(drugs), drug_counts)
        plt.title("Drug Counts")
        
        plt.subplot(2, 2, 3)
        plt.bar(list(orders), order_counts)
        plt.title("Object Counts")
        
        plt.subplot(2, 2, 4)
        plt.bar(list(times), time_counts)
        plt.title("Time Counts")

        # Fullscreen
        #mng = plt.get_current_fig_manager()
        #mng.window.state('zoomed')
        
        plt.show()

    def get_average_spectrums(self, drug:str, time:str, visualize:bool = True) -> ndarray:
        selected = self.select(drug=drug, time=time)
        n_selected = len(selected)
        signal_length = len(selected[0].measurement_data[channels[0]])
        channel_spec = {ch: [] for ch in channels}
        
        #butter_sos = butter(N=5, Wn=[2 / fs, 40 / fs], btype='bandpass', output='sos')
        f_low = 1.5
        f_high = 20
        butter_sos = butter(N=10, Wn=[2*f_low/fs, 2*f_high/fs], btype='bandpass', output='sos')
        for s in selected:
            for ch in channels:
                sig = s.measurement_data[ch]
                sig_filtered = signal.sosfiltfilt(butter_sos, sig)
                F, spec = welch(sig_filtered, fs=fs, nperseg=5*round(fs), noverlap=round(4*fs), nfft=5*round(fs), scaling='density')
                spec /= np.sum(spec)
                channel_spec[ch].append(spec)
    
        channel_spec = OrderedDict(sorted(channel_spec.items()))
        
        if visualize:
            plt.figure(f"Averaged spectrums per-channel for {drug} at time {time}")
            
            for i, (key, value) in enumerate(channel_spec.items()):
                plt.subplot(6, 3, i+1)
                matrix = np.array(value)
                mean = np.mean(matrix, axis=0)

                plt.plot(F, mean, color='red')

                for i in range(matrix.shape[0]):
                    plt.plot(F, matrix[i, :], alpha=0.2, color='gray', label='_nolegend_')
        
                plt.xlim([0, 20])
                plt.xlabel('Frequency [Hz]')
                plt.legend([key])
            plt.show()

        avg_channels = [np.mean(np.array(channel), axis=0) for _, channel in channel_spec.items()]
        avg_channels = [s / np.sum(s) for s in avg_channels]
        avg_channels = np.array(avg_channels)

        if visualize:
            similarity_matrix = np.corrcoef(avg_channels)
            plt.matshow(similarity_matrix)
            plt.show()

        return avg_channels
    
    def get_spectrograms_train(self, id: int, cutoff_frequency: float = 49.0) -> tuple[str, ndarray]:
        item = self[id]
        items = self.select(id=item.id, drug=item.drug, order=item.order)
        label = item.drug
        
        cuttof_id = None
        spectrograms = None
        loaded = False
        channel_id = 0
        
        for it in items:
            for ch in channels:
                sig = it[ch]
                f, t, S = spectrogram(sig, fs, nperseg=512, noverlap=256)
                if not loaded:
                    cuttof_id = np.where(f >= cutoff_frequency)[0][0]
                    spectrograms = np.zeros((len(items) * len(channels), cuttof_id, len(t)))
                    loaded = True
                spectrograms[channel_id, ...] = S[:cuttof_id, :]
                channel_id += 1
        
        return label, spectrograms




    def __len__(self) -> int:
        return len(self.folder)
    
    def __getitem__(self, id: int) -> Measurement:
        return self.folder[id]

if __name__ == "__main__":
    dataset = Dataset('data', is_loaded=False)
    #dataset = Dataset('test.pickle', is_loaded=True)
    """
    dataset.sumarize_basic_stats(False)
    channels = dict()
    for drug in dataset.drugs:
        channels[drug] = dataset.get_average_spectrums(drug=drug, time='T0', visualize=False)
    #dataset.sumarize_basic_stats()
    channels_cbd = dataset.get_average_spectrums(drug='CBD', time='T2', visualize=False)
    channels_psilo = dataset.get_average_spectrums(drug='psilo', time='T2', visualize=False)
    """



