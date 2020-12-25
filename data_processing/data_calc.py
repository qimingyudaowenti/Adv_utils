import sys

import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm


def calc_norm(ds: Dataset):
    """ Get the mean and std of image Dataset.

    :param ds: CHW format. in range [0, 1]
    :return: normed mean and std
    """
    total = []
    for i in tqdm(range(len(ds)), file=sys.stdout):
        img = np.array(ds[i][0])
        total.append(img)
    total = np.vstack(total)

    if total.shape[-1] == 3:
        total = total.reshape(-1, 32, 32, 3)
    else:
        total = total.reshape(-1, 3, 32, 32)
        total = total.transpose((0, 2, 3, 1))

    ds_mean = total.mean(axis=(0, 1, 2))
    ds_std = total.std(axis=(0, 1, 2))

    if ds_mean.max() > 1.:
        ds_mean /= 255
        ds_std /= 255

    return ds_mean, ds_std


if __name__ == '__main__':
    from torchvision import datasets

    train_set1 = datasets.CIFAR10(root='~/torchvision_dataset',
                                  train=True, download=False)

    print(calc_norm(train_set1))

    train_set = datasets.CIFAR10(root='~/torchvision_dataset',
                                 train=True, download=False)

    print(train_set.data.mean(axis=(0, 1, 2)) / 255)
    print(train_set.data.std(axis=(0, 1, 2)) / 255)
