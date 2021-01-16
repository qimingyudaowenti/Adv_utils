# For paper: On the Sensitivity of Adversarial Robustness to Input Data
# Distributions. ICLR 2019

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class MeanSmoothing(nn.Module):
    """
    only for mnist images
    """

    def __init__(self, kernel_size):
        super(MeanSmoothing, self).__init__()

        kernel = torch.full((1, 1, kernel_size, kernel_size),
                            1 / (kernel_size * kernel_size))
        self.register_buffer('weight', kernel)

    def forward(self, input):
        n, c, h, w = input.size()
        imgs_padded = torch.zeros((n, 1, h + 2, w + 2), dtype=torch.float)
        imgs_padded[:, :, 1:-1, 1:-1] = input
        return F.conv1d(imgs_padded, weight=self.weight)


def mnist_to_cifar10(mnist_imgs: torch.Tensor, kernel_size):
    # Smoothing
    imgs = mnist_imgs
    smooth_layer = MeanSmoothing(kernel_size)

    return smooth_layer(imgs)


def cifar10_to_mnist(cifar10_imgs: torch.Tensor, p):
    # Saturation
    # When p = 2 it does not change the image,
    # and when p = ∞ it becomes binarization.
    imgs = cifar10_imgs.numpy()
    temp = 2 * imgs - 1
    imgs_saturated = np.sign(temp) * \
        np.power(np.absolute(temp), 2 / p) / 2 + 0.5

    return torch.from_numpy(imgs_saturated)


if __name__ == '__main__':
    from utils.config import dir_dataset
    from utils.data_processing.get_samples import get_random_mnist_samples, get_random_cifar10_samples
    from utils.vis import show_images

    imgs, _, _ = get_random_mnist_samples(dir_dataset, 6)
    imgs = imgs.unsqueeze(1)
    imgs_c = mnist_to_cifar10(imgs, 3)
    print(imgs.size())
    print(imgs_c.size())
    show_images(imgs)
    show_images(imgs_c)

    imgs, _, _ = get_random_cifar10_samples(dir_dataset, 6)
    imgs_m = cifar10_to_mnist(imgs, 16)
    print(imgs.size())
    print(imgs_m.size())
    show_images(imgs)
    show_images(imgs_m)
