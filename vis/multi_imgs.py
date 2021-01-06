import matplotlib.pyplot as plt
import numpy as np
import torch


def show_images(images, num_per_col: int = 1, titles: list = None):
    """Display a list of images(0~1) in a single figure with matplotlib.

    :param images: List of np.arrays compatible with plt.imshow.
    :param num_per_col: Number of columns in figure (number of rows is
                        set to np.ceil(n_images/float(cols))).
    :param titles: List of titles corresponding to each image. Must have
                   the same length as titles.
    :return: None
    """
    assert ((titles is None) or (len(images) == len(titles)))

    if isinstance(images, torch.Tensor):
        # NCHW
        images = images.cpu().permute(0, 2, 3, 1)
        images = images.numpy()

    n_images = len(images)
    if titles is None:
        titles = [str(i) for i in range(1, n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(num_per_col,
                            int(np.ceil(n_images / float(num_per_col))),
                            n + 1)
        if image.ndim == 2 or image.shape[2] == 1:
            plt.gray()
        plt.axis('off')
        plt.imshow(image, vmin=0, vmax=1)
        a.set_title(title)

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
