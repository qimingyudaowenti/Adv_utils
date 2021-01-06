from typing import Tuple

import numpy as np
import torch
from torchvision import datasets


def get_random_mnist_samples(dataset_path: str, num: int,
                             train: bool = False) -> Tuple[torch.Tensor, torch.Tensor, list]:
    dataset = datasets.MNIST(root=dataset_path,
                             train=train, download=False)
    idx = np.random.choice(len(dataset), num, replace=False)
    images = dataset.data[idx] / 255.0
    labels = dataset.targets[idx]
    cls = [str(i.item()) for i in labels]

    return images, labels, cls


def get_random_cifar10_samples(dataset_path: str, num: int,
                               train: bool = False) -> Tuple[torch.Tensor, torch.Tensor, list]:
    dataset = datasets.CIFAR10(root=dataset_path,
                               train=train, download=False)
    idx = np.random.choice(len(dataset), num, replace=False)
    images = dataset.data[idx] / 255.0  # np.ndarray
    labels = np.array(dataset.targets)[idx]

    images = torch.from_numpy(np.transpose(images,
                                           (0, 3, 1, 2))).to(torch.float)
    labels = torch.from_numpy(labels)

    cls_dict = dataset.class_to_idx
    f = lambda x: list(cls_dict.keys())[list(cls_dict.values()).index(x)]
    cls = [f(i) for i in labels]

    # NCHW
    return images, labels, cls
