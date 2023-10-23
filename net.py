import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

H_constant = 254

class Net(nn.Module):
    def __init__(self, channels: list[int], reductions: list[str], train_type: str, num_classes: int = None, output_dimension: int = None) -> None:
        super().__init__()
        assert len(channels) == len(reductions) + 1
        self.feature_extractor = nn.Sequential(
            nn.BatchNorm2d(num_features=channels[0]),
            *nn.ModuleList([ResBlock(ch1, ch2, red) for ch1, ch2, red in zip(channels[:-1], channels[1:], reductions)])
        )
        
        self.FC = FullyConnected(channels[-1] * (H_constant // 2), num_classes=num_classes, train_type=train_type, output_dimension=output_dimension)

    def forward(self, x: Tensor) -> Tensor:
        x = self.feature_extractor(x)
        x = F.adaptive_max_pool2d(x, output_size=(x.shape[2], 1))
        if x.shape[-2:] != (H_constant, 1):
            x = F.interpolate(x, (H_constant // 2, 1))
        B, C, H, W = x.shape
        x = x.view(B, C * H * W)
        return self.FC(x)

class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, reduction: str = None) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(15, 3), padding=(7, 1), bias=False),
            nn.BatchNorm2d(num_features=in_channels), 
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(15, 3), padding=(7, 1), bias=True),
            nn.ReLU(inplace=True)
        )

        self.res = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=True)

        if not reduction:
            reduction = nn.Identity()
        elif reduction == 'mean':
            reduction = nn.AvgPool2d(kernel_size=(2, 1))
        elif reduction == 'max':
            reduction = nn.MaxPool2d(kernel_size=(2, 1))

        self.out = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(15, 3), padding=(7, 1), bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, bias=True),
        )
    
    def forward(self, x: Tensor) -> Tensor:
        return self.out(self.block(x) + self.res(x))
    
class FullyConnected(nn.Module):
    def __init__(self, in_channels: int, num_classes: int = None, output_dimension :int = 2, train_type: str = 'classification') -> None:
        super().__init__()

        if train_type == 'classification' and not num_classes:
            raise RuntimeError('Cannot initilize network. Specify the number of classes for classification.')

        assert train_type in ['classification', 'clustering']
        if train_type == 'classification':
            out_layer = nn.Linear(in_channels // 2, num_classes, bias=True)
        else:
            out_layer = nn.Linear(in_channels // 2, output_dimension, bias=True)

        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2, bias=False),
            nn.BatchNorm1d(in_channels // 2),
            nn.ReLU(inplace=True),
            out_layer
        )
    
    def forward(self, x: Tensor) -> Tensor:
        return self.fc(x)
    
if __name__ == "__main__":
    from time import time as t
    channels = [72, 128, 128, 256, 2]
    reductions = ['mean', 'mean', 'mean', None]
    net = Net(channels, reductions, train_type='clustering', output_dimension=2).cuda()
    B = 5
    x = torch.rand((B, 72, 254, 290)).cuda()

    x_mapped = net(x)

    print(x_mapped)