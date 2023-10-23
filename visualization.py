import torch
from torch import Tensor
import numpy as np
import matplotlib.pyplot as plt

def plot_clusters(coords: Tensor, labels: list[str] = None) -> None:
    x, y = torch.split(coords, split_size_or_sections=1, dim=0)
    x = x.cpu().numpy().flatten()
    y = y.cpu().numpy().flatten()

    if not labels:
        labels = np.ones(len(x))

    labels_unique = np.unique(labels).tolist()
    labels = np.array([labels_unique.index(x) for x in labels])

    for i, l in enumerate(labels_unique):
        plt.scatter(x[labels == i], y[labels == i], label=l)

    plt.legend()
    plt.show()


if __name__ == "__main__":
    N = 180
    coords = torch.rand((2, N))
    labels = [str(i % 8) for i in range(N)]

    plot_clusters(coords, labels)