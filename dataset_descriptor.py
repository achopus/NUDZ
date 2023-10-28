import torch
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from copy import deepcopy

import numpy as np


class DatasetDescriptor(Dataset):
    def __init__(self, data: str, labels: str, transform_strength:float = None) -> None:
        super().__init__()
        self.data = np.load(data)
        self.labels = np.load(labels)

        self.classes = list(np.unique(self.labels))
        self.n_classes = len(self.classes)

        self.transform_strength = transform_strength
        self.N = len(self.labels)

    def __len__(self) -> int:
        return self.N
    
    def __getitem__(self, index) -> tuple[Tensor, Tensor]:
        labels_tensor = F.one_hot(torch.tensor(self.classes.index(self.labels[index])), self.n_classes)

        data = torch.from_numpy(self.data[index, :])

        if self.transform_strength:
            data += self.transform_strength * torch.rand_like(data)
        return data, labels_tensor
    
    def generate_train_validation_split(self, ratio: float = 0.25) -> tuple[Dataset, Dataset]:
        ids = {c: np.where(self.labels == c)[0] for c in self.classes}
        ids_valid = {c: np.random.choice(ids[c], size=int(len(ids[c]) * ratio), replace=True) for c in self.classes}
        
        ids_train = {}
        for c in self.classes: ids_train[c] = [x for x in ids[c] if x not in ids_valid[c]]

        idt = []
        idv = []
        for c in self.classes:
            for x in ids_train[c]: idt.append(x)
            for x in ids_valid[c]: idv.append(x)


        dataset_train = deepcopy(self)
        dataset_train.data = dataset_train.data[idt]
        dataset_train.labels = dataset_train.labels[idt]
        #dataset_train.labels = np.random.randint(0, len(self.classes), len(idt))
        dataset_train.N = len(idt)

        dataset_valid = deepcopy(self)
        dataset_valid.data = dataset_valid.data[idv]
        dataset_valid.labels = dataset_valid.labels[idv]
        dataset_valid.N = len(idv)
        dataset_valid.transform_strength = None

        return dataset_train, dataset_valid




def get_dataloader(dataset: DatasetDescriptor, batch_size: int, shuffle:bool = True) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

if __name__ == "__main__":
    dataset = DatasetDescriptor('data_descriptors.py.npy', 'labels.py.npy')
    dataset_train, dataset_valid = dataset.generate_train_validation_split()