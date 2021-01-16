import torch

from utils.net_helper import get_input_grad


def l2_clipper(o, x, eps):
    x = torch.clamp(x, 0, 1)
    delta = x - o
    n = torch.norm(delta.view(o.size(0), -1), dim=1)

    for idx in range(len(x)):
        if n[idx] > eps:
            x[idx] = o[idx] + delta[idx] / n[idx] * eps

    return x


def make_adv_l2(model, x, y, criterion):
    # normalize?
    advs = torch.clone(x).detach()
    advs.requires_grad = True

    s = 1.5
    eps = 40.0
    iteration = 40

    for i in range(iteration):
        g_ = get_input_grad(model, advs, y, criterion)
        g_ = g_ / torch.norm(g_.view(g_.size(0), -1),
                             dim=1)[..., None, None, None]
        advs.data = l2_clipper(x, advs.data + g_ * s, eps)
        advs.grad.data.zero_()

    return advs


def linf_clipper(o, x, eps):
    x = torch.clamp(x, 0, 1)
    delta = x - o
    delta = torch.clamp(delta, -eps, eps)

    return torch.clamp(o + delta, 0, 1)


def make_adv_linf(model, x, y, criterion):
    advs = torch.clone(x).detach()
    advs.requires_grad = True

    s = 0.1
    eps = 0.8
    iteration = 40

    for i in range(iteration):
        g_ = get_input_grad(model, advs, y, criterion)
        delta = s * torch.sign(g_)
        advs.data = linf_clipper(x, advs.data + delta, eps)
        advs.grad.data.zero_()

    return advs


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import torch.nn as nn

    from models import MnistCls, PreActResNet18
    from utils.vis import make_adv_l2, show_images
    from utils.data_processing import get_random_mnist_samples, get_random_cifar10_samples, InputNormalize
    from utils.net_helper import set_seed
    from utils.config import *

    set_seed(0)

    dataset_dir = '~/torchvision_dataset'


    def mnist():
        images, labels, cls = get_random_mnist_samples(dataset_dir, 5)
        norm = norm_mnist_mix

        model = MnistCls()
        model.load_state_dict(torch.load('weights/mnist/mix_train/norm.pth'))
        model.to('cuda')
        model.eval()
        normalizer = InputNormalize(*norm).to('cuda')

        criterion = nn.CrossEntropyLoss()

        x = images.unsqueeze(1).to(device='cuda', dtype=torch.float)
        x = normalizer(x)
        x.requires_grad = True
        y = labels.to('cuda')

        advs = make_adv_l2(model, x, y, criterion)

        preds = model(advs).max(1)[1]
        adv_cls = [str(i) for i in preds.cpu().tolist()]

        adv_images = advs.detach().cpu().squeeze(1).numpy()
        print(images.size(), len(cls))
        show_images(images, num_per_col=1, titles=cls)
        show_images(adv_images, num_per_col=1, titles=adv_cls)
        plt.show()


    def cifar10():
        cls_dict = {'airplane': 0, 'automobile': 1, 'bird': 2,
                    'cat': 3, 'deer': 4, 'dog': 5, 'frog': 6,
                    'horse': 7, 'ship': 8, 'truck': 9}
        images, labels, cls = get_random_cifar10_samples(dataset_dir, 5)
        model = PreActResNet18()
        w_path = 'weights/cifar10/mix_train/2021-01-07-21-32-55_200_256_0.1_0.001_.pth'
        model.load_state_dict(torch.load(w_path))
        norm = norm_cifar10_mix
        normalizer = InputNormalize(*norm).to('cuda')

        model.to('cuda')
        model.eval()

        criterion = nn.CrossEntropyLoss()

        x = images.to(dtype=torch.float, device='cuda')
        x = normalizer(x)
        x.requires_grad = True
        y = labels.to('cuda')

        advs = make_adv_l2(model, x, y, criterion)

        preds = model(advs).max(1)[1]

        adv_cls = [i for i in preds.cpu().tolist()]
        f = lambda x: list(cls_dict.keys())[list(cls_dict.values()).index(x)]
        adv_cls = [f(i) for i in adv_cls]

        images = images.cpu().permute(0, 2, 3, 1).numpy()
        adv_images = advs.detach().cpu().permute(0, 2, 3, 1).numpy()

        show_images(images, num_per_col=1, titles=cls)
        show_images(adv_images, num_per_col=1, titles=adv_cls)
        plt.show()

    cifar10()
