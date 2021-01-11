from typing import Tuple

import numpy as np
import torch
from torchvision import datasets


def get_random_mnist_samples(dataset_path: str, num: int,
                             train: bool = False, balance: bool = False) -> Tuple[torch.Tensor, torch.Tensor, list]:
    dataset = datasets.MNIST(root=dataset_path,
                             train=train, download=False)
    images = dataset.data
    labels = dataset.targets

    if not balance:
        idx = np.random.choice(len(dataset), num, replace=False)
    else:
        assert num % 10 == 0
        num_per_class = num // 10
        idx_class = [np.where(labels == i)[0] for i in range(10)]
        idx = [np.random.choice(i, num_per_class, replace=False) for i in idx_class]
        idx = np.concatenate(idx)

    images_sampled = images[idx] / 255.0
    labels_sampled = labels[idx]

    cls = [str(i.item()) for i in labels]

    return images_sampled, labels_sampled, cls


def get_random_cifar10_samples(dataset_path: str, num: int,
                               train: bool = False, balance: bool = True) -> Tuple[torch.Tensor, torch.Tensor, list]:
    dataset = datasets.CIFAR10(root=dataset_path,
                               train=train, download=False)

    images = dataset.data
    labels = np.array(dataset.targets)

    if not balance:
        idx = np.random.choice(len(dataset), num, replace=False)
    else:
        assert num % 10 == 0
        num_per_class = num // 10
        idx_class = [np.where(labels == i)[0] for i in range(10)]
        idx = [np.random.choice(i, num_per_class, replace=False) for i in idx_class]
        idx = np.concatenate(idx)

    images = images[idx] / 255.0  # np.ndarray
    labels = labels[idx]

    images = torch.from_numpy(np.transpose(images,
                                           (0, 3, 1, 2))).to(torch.float)
    labels = torch.from_numpy(labels)

    cls_dict = dataset.class_to_idx
    f = lambda x: list(cls_dict.keys())[list(cls_dict.values()).index(x)]
    cls = [f(i) for i in labels]

    # NCHW
    return images, labels, cls


if __name__ == '__main__':
    from utils.config import dir_dataset

    _, a, _ = get_random_mnist_samples(dir_dataset, 20, balance=True)
    print(a)
