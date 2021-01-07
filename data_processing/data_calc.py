import sys

import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm


def calc_norm(ds: Dataset, max_v: int = 255, neg_mix: bool = False):
    """ Get the mean and std of image Dataset. Do not use data attribute for custom Dataset.

    :param neg_mix: if True, calc the norm of mix dataset
    :param max_v: 1 or 255
    :param ds: HWC or HW (if C is 1)
    :return: norm (mean and std)
    """
    total = []

    example_img = np.asarray(ds[0][0])
    num_imgs = len(ds)

    # if example_img.ndim == 3:
    #     img_type = 'color'
    #     h, w, c = example_img.shape
    # elif example_img.ndim == 2:
    #     img_type = 'gray'
    #     h, w = example_img.shape
    #     c = 1
    # else:
    #     raise ValueError

    for i in tqdm(range(num_imgs), file=sys.stdout):
        img = np.array(ds[i][0])
        total.append([img])
        if neg_mix:
            total.append([255 - img])

    total = np.vstack(total)

    if total.ndim == 4:
        if total.shape[-1] == 3 or total.shape[-1] == 1:
            pass
        elif total.shape[1] == 3 or total.shape[1] == 1:
            total = total.transpose((0, 2, 3, 1))
        else:
            raise ValueError

        ds_mean = total.mean(axis=(0, 1, 2))
        ds_std = total.std(axis=(0, 1, 2))
    elif total.ndim == 3:
        total = total * 1.0  # change to float
        ds_mean = total.mean()
        ds_std = total.std()
    else:
        raise ValueError

    ds_mean /= max_v
    ds_std /= max_v

    return ds_mean, ds_std


if __name__ == '__main__':
    from torchvision import datasets
    from utils.config import dir_dataset

    train_set_cifar10 = datasets.CIFAR10(root=dir_dataset,
                                         train=True, download=False)

    print('========== CIFAR10 ==========')
    print(calc_norm(train_set_cifar10))
    print(train_set_cifar10.data.mean(axis=(0, 1, 2)) / 255,
          train_set_cifar10.data.std(axis=(0, 1, 2)) / 255)
    print('Neg-Mix:', calc_norm(train_set_cifar10, neg_mix=True))
    # (array([0.5, 0.5, 0.5]), array([0.2471819 , 0.24413793, 0.26699652]))

    train_set_mnist = datasets.MNIST(root=dir_dataset,
                                     train=True, download=False)
    print('========== MNIST ==========')
    print(calc_norm(train_set_mnist))
    print((train_set_mnist.data * 1.0).mean() / 255,
          (train_set_mnist.data * 1.0).std() / 255)
    print('Neg-Mix:', calc_norm(train_set_mnist, neg_mix=True))
    # (0.5, 0.48098035571478076)
