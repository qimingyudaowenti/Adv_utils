import sys

import torch
from tqdm import tqdm

from utils.attack import *
from utils.config import *


def test_accuracy(model, loader_eval, norm, proportion: float = 1.0):
    device = 'cuda:0'

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


def test_robustness(model, loader_eval, norm,
                    attack_config: ConfigAttack, proportion: float = 1.0):
    device = 'cuda:0'

    model.to(device)
    model.eval()

    correct = 0
    total = 0
    total_batches = len(loader_eval)
    stop_batch_num = int(total_batches * proportion)

    attacker = AttackerPGD(model, attack_config, norm)
    attacker.to(device)

    iterator_tqdm = tqdm(loader_eval, file=sys.stdout, position=0)

    for i, train_batched in enumerate(iterator_tqdm):

        # skip some batches for saving time
        if i > stop_batch_num:
            break

        imgs = train_batched[0].to(device)
        labels = train_batched[1].to(device)

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


if __name__ == '__main__':
    from torchvision import datasets
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader
    from models import PreActResNet18
    from config import norm_cifar10, cfg_attack_cifar10
    import numpy as np

    # eval loader (testing set)
    ds_test = datasets.CIFAR10(root=dir_dataset,
                               train=False, download=False,
                               transform=transforms.ToTensor())
    loader_test = DataLoader(ds_test,
                             batch_size=256,
                             shuffle=False, num_workers=4)

    model = PreActResNet18()
    model.to('cuda:0')

    print(test_accuracy(model, loader_test, norm_cifar10, proportion=0.5))

    print(test_robustness(model, loader_test, norm_cifar10, cfg_attack_cifar10, proportion=0.5))
