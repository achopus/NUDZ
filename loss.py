import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    def __init__(self, radius: float = 1.0) -> None:
        super().__init__()
        self.radius = radius
    
    def forward(self, feature_space: Tensor, labels: Tensor) -> Tensor:
        loss_value = torch.tensor([0]).to(device=feature_space.device, dtype=feature_space.dtype)
        B = feature_space.shape[0]
        
        for i in range(B):
            for j in range(i+1, B):
                dist = torch.linalg.norm(feature_space[i, ...] - feature_space[j, ...], ord=2)
                is_same = torch.all(torch.eq(labels[i, :], labels[j, :])).to(dist.dtype)
                loss_value += 0.5 * is_same * dist + 0.5 * (1.0 - is_same) * torch.clip(self.radius - dist, min=0.0)
        
        return loss_value / B


class TripletLoss(nn.Module):
    def __init__(self, radius: float = 1.0) -> None:
        super().__init__()
        self.radius = radius        

    def forward(self, anchor: Tensor, positive: Tensor, negative: Tensor) -> Tensor:
        loss_positive = torch.linalg.norm(anchor - positive)**2
        loss_negative = torch.linalg.norm(anchor - negative)**2

        loss = loss_positive + F.relu(self.radius - loss_negative)
        
        return loss

if __name__ == "__main__":
    feature_space = torch.rand((5, 256))
    labels = torch.zeros((5, 3))
    labels[0, 2] = 1.0
    labels[1, 1] = 1.0
    labels[2, 0] = 1.0
    labels[3, 0] = 1.0
    labels[4, 2] = 1.0

    loss = ContrastiveLoss()

    l = loss(feature_space, labels)

    print(l.item())
