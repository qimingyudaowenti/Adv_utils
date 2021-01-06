import numpy as np
import torch


def norm_grad(grads, adjust_param=3):
    """ used to vis grad of 3-channel image

    :param grads:
    :param adjust_param:
    :return:
    """
    std = adjust_param * np.std(grads)
    img = np.clip(grads, a_max=std, a_min=-std)
    img = (1 + img / std) * 0.5

    return img


def pos_neg_split(grads, adjust_param=5):
    """ used to vis grad of 1-channel image (gray image)

    :param grads: nd.array, HW
    :param adjust_param:
    :return: split img
    """
    assert (grads.ndim == 2 or grads.shape[2] == 1)

    new_img = np.ones((*grads.shape[:2], 3))

    pos_scale = 1. / np.max(grads) * adjust_param
    neg_scale = -1. / np.min(grads) * adjust_param

    pos_idx = np.where(grads > 0)
    new_img[..., 1][pos_idx] = 1 - grads[pos_idx] * pos_scale
    new_img[..., 2][pos_idx] = new_img[..., 1][pos_idx]

    neg_idx = np.where(grads < 0)
    new_img[..., 1][neg_idx] = 1 + grads[neg_idx] * neg_scale
    new_img[..., 0][neg_idx] = new_img[..., 1][neg_idx]

    new_img = np.clip(new_img, 0, 1)

    return new_img


def vis_imgs_grad(grads: torch.Tensor, channel_num: int):
    if channel_num == 1:
        grads = grads.squeeze().cpu().numpy()
        imgs = [pos_neg_split(g) for g in grads]
    elif channel_num == 3:
        grads = grads.squeeze().cpu().permute(0, 2, 3, 1).numpy()
        imgs = [norm_grad(g) for g in grads]
    else:
        raise ValueError('Grad for vis should be 1 or 3 channel.')

    return imgs


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import torch
    import torch.nn as nn

    from models.mnist import MnistCls
    from models.cifar10 import PreActResNet18
    from utils.net_helper import get_input_grad
    from utils.vis.multi_imgs import show_images
    from utils.data_processing import get_random_mnist_samples,get_random_cifar10_samples

    np.random.seed(0)

    dataset_dir = '~/torchvision_dataset'

    def mnist():
        images, labels, cls = get_random_mnist_samples(dataset_dir, 5)
        model = MnistCls()
        model.load_state_dict(torch.load('weights/mnist/mix_train/2020-12-22-18-53-15_10_128_0.01_0.001_.pth'))
        model.to('cuda')
        model.eval()

        criterion = nn.CrossEntropyLoss()

        x = images.unsqueeze(1).to(dtype=torch.float)

        grads = get_input_grad(model, x, labels, criterion)

        vis_grad = vis_imgs_grad(grads, channel_num=1)
        show_images(images, num_per_col=1, titles=cls)
        show_images(vis_grad, num_per_col=1, titles=cls)


    def cifar10():
        images, labels, cls = get_random_cifar10_samples(dataset_dir, 5)
        model = PreActResNet18()
        model.load_state_dict(torch.load('weights/cifar10/mix_train/weights/cifar10/mix_train/2021-01-05-17-33-03_200_256_0.1_0.001_.pth'))
        model.to('cuda')
        model.eval()

        criterion = nn.CrossEntropyLoss()

        x = images.to(dtype=torch.float)

        grads = get_input_grad(model, x, labels, criterion)

        vis_grad = vis_imgs_grad(grads, channel_num=3)
        show_images(images.permute(0, 2, 3, 1), num_per_col=1, titles=cls)
        show_images(vis_grad, num_per_col=1, titles=cls)

    cifar10()

