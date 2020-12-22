from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

from utils.config import *


def get_loader(ds_name: str, train: bool, batch_size: int, normed: bool = False):
    if ds_name.upper() == 'MNIST':
        aug_mnist = transforms.Compose([
            transforms.ToTensor(),
        ])

        if normed:
            aug_mnist = transforms.Compose([
                aug_mnist,
                transforms.Normalize(norm_mnist[0],
                                     norm_mnist[1])
            ])

        ds = datasets.MNIST(root=dir_dataset,
                            train=train, download=False,
                            transform=aug_mnist)
    elif ds_name.upper() == 'CIFAR10':
        aug_cifar10 = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(.25, .25, .25),
            transforms.RandomRotation(2),
            transforms.ToTensor(),
        ])

        if normed:
            aug_cifar10 = transforms.Compose([
                aug_cifar10,
                transforms.Normalize(norm_cifar10[0],
                                     norm_cifar10[1])
            ])

        ds = datasets.CIFAR10(root=dir_dataset,
                              train=train, download=False,
                              transform=aug_cifar10)
    else:
        raise ValueError('Dataset should be CIFAR10 or MNIST.')

    loader = DataLoader(ds, batch_size=batch_size,
                        shuffle=True, num_workers=4, pin_memory=True)

    return loader
