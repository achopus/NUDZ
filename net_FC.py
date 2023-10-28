import torch
from torch import Tensor

import torch.nn as nn
from torch.nn import Module

import torch.nn.functional as F


class DualNet(Module):
    def __init__(self, n_features: list[int], n_classes: int = 10, out_dim: int = 2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            *nn.ModuleList([LayerBlock(in_f, out_f) for in_f, out_f in zip(n_features[:-1], n_features[1:])])
        )

        self.classifer_branch = nn.Linear(n_features[-1], n_classes, bias=True)
        self.clustering_branch = nn.Linear(n_features[-1], out_dim, bias= True)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        feature_map = self.net(x)

        class_predictions = self.classifer_branch(feature_map)
        cluster_mapping = self.clustering_branch(feature_map)

        return class_predictions, cluster_mapping


class NetFC(Module):
    def __init__(self, n_features: list[int], n_classes: int = None) -> None:
        super().__init__()
        self.net = nn.Sequential(
            *nn.ModuleList([LayerBlock(i, j) for i, j in zip(n_features[:-1], n_features[1:])]),
        )
        self.out = nn.Linear(n_features[-1], n_classes)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

class LayerBlock(Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.Dropout(p=0.25),
            nn.BatchNorm1d(out_features),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


if __name__ == "__main__":
    net = NetFC([1080, 1080, 640, 320, 420], 10)
    x = torch.zeros(10, 1080)

    y = net(x)
    print(F.softmax(y))