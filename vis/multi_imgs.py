import matplotlib.pyplot as plt
import numpy as np
import torch


def show_images(images, num_per_col: int = 1, titles: list = None, first_line_title: bool = False):
    """Display a list of images(0~1) in a single figure with matplotlib.

    :param first_line_title:
    :param images: List of np.arrays compatible with plt.imshow.
    :param num_per_col: Number of columns in figure (number of rows is
                        set to np.ceil(n_images/float(cols))).
    :param titles: List of titles corresponding to each image.
    :return: None
    """

    n_images = len(images)

    if titles is None:
        # titles = [str(i) for i in range(1, n_images + 1)]
        titles = ['' for _ in range(n_images)]
    elif len(titles) < n_images:
        if first_line_title:
            assert len(titles) == n_images // num_per_col

        titles += ['' for _ in range(n_images - len(titles))]

    if isinstance(images, torch.Tensor):
        # NCHW
        if images.ndim == 4 and images.size(1) in [1, 3]:
            images = images.cpu().permute(0, 2, 3, 1)
        images = images.numpy()

    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        line_num = int(np.ceil(n_images / float(num_per_col)))
        a = fig.add_subplot(num_per_col, line_num, n + 1)
        if image.ndim == 2 or image.shape[2] == 1:
            plt.gray()
        plt.axis('off')
        plt.imshow(image, vmin=0, vmax=1)

        if first_line_title:
            if n < line_num:
                a.set_title(title)
        else:
            a.set_title(title)

    plt.tight_layout()
    plt.show()


def show_grid(img, num_channel=3, norm: tuple = None):
    npimg = img.numpy()
    if norm is not None:
        # TODO:
        raise NotImplementedError
    if num_channel == 1:
        plt.gray()
        plt.imshow(npimg[0, ...])
    elif num_channel == 3:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
    else:
        raise ValueError('Wrong channel number.')
    plt.show()
