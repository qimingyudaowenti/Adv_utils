def pos_neg_split(grad_img, adjust_param=5):
    """ used to vis grad of one-channel image (gray image)

    :param grad_img: nd.array, HW
    :return: split img
    """
    assert (grad_img.ndim == 2 or grad_img.shape[2] == 1)

    new_img = np.ones((*grad_img.shape[:2], 3))

    pos_scale = 1. / np.max(grad_img) * adjust_param
    neg_scale = -1. / np.min(grad_img) * adjust_param

    pos_idx = np.where(grad_img > 0)
    new_img[..., 1][pos_idx] = 1 - grad_img[pos_idx] * pos_scale
    new_img[..., 2][pos_idx] = new_img[..., 1][pos_idx]

    neg_idx = np.where(grad_img < 0)
    new_img[..., 1][neg_idx] = 1 + grad_img[neg_idx] * neg_scale
    new_img[..., 0][neg_idx] = new_img[..., 1][neg_idx]

    new_img = np.clip(new_img, 0, 1)

    return new_img


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.nn as nn

    from models.mnist import MnistCls
    from utils.net_helper import get_grad
    from utils.vis.multi_imgs import show_images
    from utils.data_processing import get_random_mnist_samples

    np.random.seed(0)

    dataset_dir = '~/torchvision_dataset'
    images, labels, cls = get_random_mnist_samples(dataset_dir, 5)
    model = MnistCls()
    model.load_state_dict(torch.load('weights/2020-12-18-18-47-27_5_64_0.01_0.001_.pth'))
    model.to('cuda')
    model.eval()

    criterion = nn.CrossEntropyLoss()

    x = images.unsqueeze(1).to(dtype=torch.float)

    grads = get_grad(model, x, labels, criterion)

    grads = grads.squeeze().cpu().numpy()
    vis_grad = [pos_neg_split(g) for g in grads]
    show_images(images, num_per_col=1, titles=cls)
    show_images(vis_grad, num_per_col=1, titles=cls)
    plt.show()
