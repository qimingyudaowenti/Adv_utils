import matplotlib.pyplot as plt
import numpy as np


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
