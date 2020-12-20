from math import floor

import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.ticker import FuncFormatter

"""
plot steps:
1. make color map, choose color for every class in plot.
2. project values to the middle of interval.
3. For colorbar, make format.
"""

colors = ['purple', 'green', 'blue', 'pink', 'brown',
          'lightblue', 'teal', 'orange', 'lightgreen', 'grey']


def idx_map(cls_num):
    def bound_norm(f: float):
        f = f % cls_num
        r = floor(f / (1 / cls_num))
        if r == cls_num:
            r = cls_num - 1
        return r

    return bound_norm


def smooth_project(x: torch.Tensor):
    # x values range: [0, 1, ..., 9]
    # project values to the middle of sub-interval in (0, 10)
    values = x.unique(sorted=True).tolist()
    interval_num = len(values)
    interval_len = 10 / interval_num
    for i, v in enumerate(values):
        x[x == v] = i * interval_len + interval_len / 2
    return x


def plot_surface(X, Y, Z, C: torch.Tensor, classes: list):
    # values in C should be integer between 0 - 9
    labels_int = C.unique(sorted=True).tolist()
    cls_num = len(labels_int)
    classes_pred = [classes[i] for i in labels_int]

    cmap = ListedColormap([colors[i] for i in labels_int])
    # smooth values in C for plot

    C = smooth_project(C) * 0.1  # C should be in range 0-1
    bound_norm = idx_map(cls_num)
    fmt = FuncFormatter(lambda x, pos: classes_pred[bound_norm(x)])
    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    # can set shade to False or True
    surf = ax.plot_surface(X, Y, Z,
                           cmap=cmap,
                           facecolors=cmap(C),
                           shade=True, linewidth=0, antialiased=True)
    t = np.linspace(0, 1, 2 * cls_num + 1)[1::2]
    fig.colorbar(mappable=surf, format=fmt, ticks=t, shrink=0.8)


def plot_adv_loss_lanscape(model, img, img_adv, label, label_adv, norm, bound, classes):
    # img shape: [C, H, W]

    delta1 = img_adv - img
    delta2 = torch.rand_like(delta1) * 2 * bound - bound

    density = 31  # should be odd number
    x_axis = np.linspace(-bound, bound, density)
    y_axis = x_axis

    imgs_interp = torch.empty((density, density, *(delta1.size())))
    for ix, vx in enumerate(x_axis):
        for iy, vy in enumerate(y_axis):
            imgs_interp[ix, iy, ...] = img + vx / bound * delta1 + vy / bound * delta2

    size = imgs_interp.size()
    inputs = imgs_interp.reshape(size[0] * size[1], *size[2:])

    mean, std = norm
    try:
        inputs = (inputs - mean) / std

    except RuntimeError:
        mean = mean.reshape(3, 1, 1)
        std = std.reshape(3, 1, 1)
        inputs = (inputs - mean) / std

    outputs_interp = model(inputs.to('cuda'))
    _, predicted = torch.max(outputs_interp.data, 1)

    criterion = nn.CrossEntropyLoss(reduction='none')
    loss_interp = criterion(outputs_interp.cpu(),
                            torch.full((outputs_interp.size(0),), label))
    loss_interp = loss_interp.reshape(size[0], size[1])
    Z = loss_interp.detach().numpy()

    C = predicted.cpu().reshape(size[0], size[1])

    X, Y = np.meshgrid(x_axis, y_axis)

    plot_surface(X, Y, Z, C, classes)
    plt.title(f'nat: {classes[label.item()]} | adv: {classes[label_adv.item()]}')
    plt.show()


if __name__ == '__main__':
    from utils.config import cfg_attack_cifar10, norm_mnist
    from utils.attack import *
    from utils.data_processing import get_random_mnist_samples
    from models import MnistCls
    from utils.net_helper import *
    from utils.config import classes_mnist

    path_dataset_dir = '~/torchvision_dataset'
    path_weights = 'weights/2020-12-18-18-47-27_5_64_0.01_0.001_.pth'
    model = MnistCls()
    norm = norm_mnist
    model.load_state_dict(torch.load(path_weights))

    device = get_device()
    model.to(device)
    model.eval()

    set_seed(0)
    samples_num = 5

    # ------ attack ------
    attacker = AttackerPGD(model, cfg_attack_mnist, norm)
    attacker.to(device)
    normalizer = InputNormalize(*norm).to(device)

    imgs, labels, _ = get_random_mnist_samples(path_dataset_dir, samples_num)

    # for single channel images
    if imgs.ndim == 3:
        imgs = imgs.unsqueeze(1)

    x = imgs.to(device)
    x_normed = normalizer(x.to(device))
    y = labels.to(device)

    # ------ adversarial acc ------
    im_adv, im_adv_normed = attacker(x, y)
    outputs_adv = model(im_adv_normed)
    _, predicted = torch.max(outputs_adv.data, 1)
    correct = (predicted == y).sum().item()
    print(f'Adversarial accuracy of {samples_num} '
          f'images is: {correct / samples_num:.2%}')

    # ------ draw landscape of single image ------
    for i in range(samples_num):
        im_adv = im_adv.cpu()
        delta = im_adv - imgs
        img = imgs[i]
        label = labels[i]
        label_adv = predicted.cpu()[i]
        plot_adv_loss_lanscape(model, img, im_adv[0], label, label_adv, norm, bound=0.3, classes=classes_mnist)
