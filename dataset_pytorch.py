from dataset import Dataset
from torch.utils.data import Dataset as DatasetTorch
from torch.utils.data import DataLoader
from measurement import Measurement
import torch
from torch import Tensor
import numpy as np

import random

class NSDDataset(DatasetTorch):
    def __init__(self, file_path: str) -> None:
        super().__init__()
        self.path = file_path
        self.data = Dataset(file_path, is_loaded=True)
        self.data.sumarize_basic_stats(visualize=False)
        self.N = len(self.data)
        self.possible_ids = [x for x in range(self.N)]
        self.epoch = 0

    def __len__(self) -> int:
        return self.N
    
    def __getitem__(self, id: list[int]) -> tuple[Tensor, Tensor]:
        output_data = [self.data.get_spectrograms_train(i) for i in id]
        output_data = [out for out in output_data if out[1].shape[0] == 72]
        labels_encoded = torch.zeros((len(output_data), len(self.data.drugs)))
        specs_matrix = torch.zeros((len(output_data), *list(output_data[0][1].shape)))

        for i, (label, spec) in enumerate(output_data):
            k = self.data.drugs.index(label)
            labels_encoded[i, k] = 1.0
            specs_matrix[i, ...] = torch.from_numpy(spec)
        
        specs_matrix = torch.nan_to_num(specs_matrix, nan=0)

        return labels_encoded, specs_matrix
    
    def get_batch(self, batch_size: int) -> tuple[Tensor, Tensor]:
        assert batch_size <= len(self)

        if batch_size >= len(self.possible_ids):
            self.possible_ids = [x for x in range(self.N)]
            self.epoch += 1

        ids = list(np.random.choice(self.possible_ids, size=batch_size, replace=False))

        for id in ids: self.possible_ids.remove(id)
        
        return self[ids]

    def get_triplet(self) -> tuple[Tensor, Tensor, Tensor]:
        id = random.randint(0, len(self))
        drug_anchor = self.data[id].drug
        positive_class = self.data.select(drug=drug_anchor)

        anchor: Measurement = random.sample(positive_class, k=1)[0]
        ids = [x.id for x in positive_class if x.id != anchor.id]

        positive = random.choice(self.data.select(drug=drug_anchor, id=random.choice(ids)))

        drugs = self.data.drugs.copy()
        drugs.remove(drug_anchor)

        negative_class = self.data.select(drug=random.choice(drugs))
        negative = random.sample(negative_class, k=1)[0]

        anchor = self.data.get_spectrograms_train(self.data.folder.measurements.index(anchor))
        positive = self.data.get_spectrograms_train(self.data.folder.measurements.index(positive))
        negative = self.data.get_spectrograms_train(self.data.folder.measurements.index(negative))

        # Call again, until shapes have correct shape
        if anchor[1].shape[0] != 72 or positive[1].shape[0] != 72 or negative[1].shape[0] != 72:
            return self.get_triplet()
        
        return torch.from_numpy(anchor[1]).float(), torch.from_numpy(positive[1]).float(), torch.from_numpy(negative[1]).float()



if __name__ == "__main__":
    dataset = NSDDataset('saved_data.pickle')
