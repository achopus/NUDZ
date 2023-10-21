from dataset import Dataset
from torch.utils.data import Dataset as DatasetTorch
from torch.utils.data import DataLoader
from measurement import Measurement
import torch
from torch import Tensor
import numpy as np



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

if __name__ == "__main__":
    dataset = NSDDataset('saved_data.pickle')
