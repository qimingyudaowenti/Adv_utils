# from: https://github.com/pedrodiamel/nettutorial/blob/master/pytorch/pytorch_visualization.ipynb

import matplotlib.pyplot as plt
import numpy as np
from torchvision import utils


def vistensor(tensor, ch=0, allkernels=False, nrow=8, padding=1):
    '''
    vistensor: visuzlization tensor
        @ch: visualization channel
        @allkernels: visualization every channel of kernels.
    '''

    n, c, h, w = tensor.shape
    if allkernels:
        tensor = tensor.view(n * c, -1, h, w)
    elif c != 3:
        tensor = tensor[:, ch, :, :].unsqueeze(dim=1)

    rows = np.min((tensor.shape[0] // nrow + 1, 64))
    grid = utils.make_grid(tensor, nrow=nrow, normalize=True, padding=padding)
    plt.figure(figsize=(nrow, rows))
    plt.imshow(grid.numpy().transpose((1, 2, 0)))


if __name__ == '__main__':
    import torch
    from models import MnistCls, PreActResNet18

    ds_name = 'CIFAR10'

    if ds_name == 'MNIST':
        model = MnistCls()
        path_mix_w = 'weights/mnist/mix_train/2020-12-22-18-53-15_10_128_0.01_0.001_.pth'
        path_nat_w = 'weights/mnist/2020-12-22-18-57-59_10_64_0.01_0.001_.pth'
        path_adv_w = 'weights/mnist/2020-08-19-10-45-58_20_64_0.01_0.001_adv.pth'
        w_path = path_adv_w
    elif ds_name == 'CIFAR10':
        model = PreActResNet18()
        path_mix_w = 'weights/cifar10/mix_train/weights/cifar10/mix_train/2021-01-05-17-33-03_200_256_0.1_0.001_.pth'
        path_nat_w = 'weights/cifar10/PreActResNet18_2020-12-01-16-15-49_150_128_0.1_0.001_nat.pth'
        path_adv_w = 'weights/cifar10/PreActResNet18_2020-12-02-20-30-28_200_128_0.1_0.001_adv.pth'
        w_path = path_adv_w
    else:
        raise ValueError('Wrong dataset.')

    model.load_state_dict(torch.load(w_path))
    # print(model)
    kernels = model.conv1.weight.data.clone()

    vistensor(kernels)
    plt.axis('off')
    plt.show()
