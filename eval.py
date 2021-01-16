import sys

import torch
from tqdm import tqdm

from utils.attack import *
from utils.config import *
from utils.data_processing import InputNormalize
from utils.data_processing.data_prefetcher import DataPrefetcher
from utils.net_helper import get_device


def test_accuracy(model, loader_eval, norm, proportion: float = 1.0):
    device = get_device()

    model.to(device)
    model.eval()

    # norm out of loader so that the loader can also be used for robustness test.
    normalizer = InputNormalize(*norm).to(device)

    correct = 0
    total = 0
    total_batches = len(loader_eval)
    stop_batch_num = int(total_batches * proportion)

    iterator_tqdm = tqdm(loader_eval, file=sys.stdout, position=0)

    with torch.no_grad():
        for i, test_batch in enumerate(iterator_tqdm):

            # skip some batches for saving time
            if i > stop_batch_num:
                break

            inputs = normalizer(test_batch[0].to(device))
            labels = test_batch[1].to(device)

            outputs = model(inputs)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            iterator_tqdm.set_description_str(f'Test on {total} examples. '
                                              f'Natural acc-{correct / total:.2%}')
    iterator_tqdm.close()

    return total, correct / total


def test_accuracy_pre(model, loader_eval, norm):
    device = get_device()

    model.to(device)
    model.eval()

    correct = 0
    total = 0

    iterator_tqdm = tqdm(range(len(loader_eval)), file=sys.stdout, position=0)

    prefetcher = DataPrefetcher(loader_eval, *norm)
    inputs, labels = prefetcher.next()
    i = 0

    while inputs is not None:
        i += 1

        with torch.no_grad():
            outputs = model(inputs)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        iterator_tqdm.set_description_str(f'Test on {total} examples. '
                                          f'Natural acc-{correct / total:.2%}')
        iterator_tqdm.update(1)
        inputs, labels = prefetcher.next()
    iterator_tqdm.close()

    return total, correct / total


def test_robustness(model, loader,
                    attack_config: ConfigAttack, proportion: float = 1.0):
    device = get_device()

    model.to(device)
    model.eval()

    correct = 0
    total = 0
    total_batches = len(loader)
    stop_batch_num = int(total_batches * proportion)

    attacker = AttackerPGD(model, attack_config)
    attacker.to(device)

    iterator_tqdm = tqdm(loader, file=sys.stdout, position=0)

    for i, test_batch in enumerate(iterator_tqdm):

        # skip some batches for saving time
        if i > stop_batch_num:
            break

        imgs = test_batch[0].to(device)
        labels = test_batch[1].to(device)

        im_adv, im_adv_normed = attacker(imgs, labels)

        # Get predicted labels for adversarial examples
        pred = model(im_adv_normed)
        pred_labels_adv = torch.argmax(pred, dim=1)

        correct += (pred_labels_adv == labels).sum().item()
        total += labels.size(0)

        iterator_tqdm.set_description_str(f'{total} examples. '
                                          f'Adversarial acc-{correct / total:.2%}')
    iterator_tqdm.close()

    return total, correct / total


def test_samples_accuracy(model, inputs: torch.Tensor, labels: torch.Tensor,
                          norm: tuple = None):
    from math import ceil
    assert len(inputs) == len(labels)
    device = 'cuda:0'

    model.to(device)
    model.eval()

    if norm is not None:
        # norm out of loader so that the loader can also be used for robustness test.
        normalizer = InputNormalize(*norm).to(device)

    correct = 0
    total = 0
    num_per_batch = 512
    num_iter_infer = ceil(len(inputs) / num_per_batch)

    with torch.no_grad():
        for i in range(num_iter_infer):
            s_idx = i * num_iter_infer
            e_idx = (i + 1) * num_per_batch

            imgs = inputs[s_idx:e_idx].to(device)
            if norm is not None:
                imgs = normalizer(imgs)
            y = labels[s_idx:e_idx].to(device)

            outputs = model(imgs)

            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

    return total, correct / total


def test_samples_robustness(model, inputs: torch.Tensor, labels: torch.Tensor,
                            cfg_attack: ConfigAttack):
    from math import ceil
    assert len(inputs) == len(labels)
    device = 'cuda:0'

    model.to(device)
    model.eval()

    attacker = AttackerPGD(model, cfg_attack)
    attacker.to(device)

    correct = 0
    total = 0
    num_per_batch = 512
    num_iter_infer = ceil(len(inputs) / num_per_batch)

    for i in range(num_iter_infer):
        s_idx = i * num_iter_infer
        e_idx = (i + 1) * num_per_batch

        imgs = inputs[s_idx:e_idx].to(device)
        labels = labels[s_idx:e_idx].to(device)

        im_adv, im_adv_normed = attacker(imgs, labels)
        outputs = model(im_adv_normed)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return total, correct / total


if __name__ == '__main__':
    from torchvision import datasets
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader
    from models import PreActResNet18
    from config import norm_cifar10, cfg_attack_cifar10

    # eval loader (testing set)
    ds_test = datasets.CIFAR10(root=dir_dataset,
                               train=False, download=False,
                               transform=transforms.ToTensor())
    loader_test = DataLoader(ds_test,
                             batch_size=256,
                             shuffle=False, num_workers=4)

    model = PreActResNet18()
    model.load_state_dict(torch.load('weights/cifar10/nat/2021-01-09-10-06-40_200_256_0.1_0.001_.pth'))
    model.to('cuda:0')

    print(test_accuracy(model, loader_test, norm_cifar10))

    print(test_robustness(model, loader_test, cfg_attack_cifar10))
