import matplotlib.pyplot as plt
import numpy as np

from utils.data_processing import get_random_cifar10_samples, get_random_mnist_samples
from utils.vis import show_images

dataset_path = '~/torchvision_dataset'
np.random.seed(2)


def vis_neg(dataset_name: str):
    assert dataset_name.upper() in ['MNIST', 'CIFAR10']

    if dataset_name.upper() == 'MNIST':
        get_samples = get_random_mnist_samples
    else:
        get_samples = get_random_cifar10_samples

    samples_num = 10
    images, _, cls = get_samples(dataset_path, samples_num, train=False, balance=True)

    if dataset_name.upper() == 'CIFAR10':
        images = images.permute(0, 2, 3, 1)
    images = images.numpy()

    neg_images = 1 - images
    pos_neg = np.vstack((images, neg_images))
    show_images(pos_neg, 2, cls, first_line_title=True)


if __name__ == '__main__':
    vis_neg('MNIST')
    vis_neg('CIFAR10')
    plt.show()
