import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10

from models import MnistCls, PreActResNet18
from utils.config import *
from utils.net_helper import set_seed
from utils.train import train_adv

set_seed(0)


def mnist():
    model = MnistCls()

    norm = norm_mnist
    cfg_train = ConfigTrain(
        batch_size=128,
        epoch=10,
        lr_schedule='cosine',
        max_lr=0.01,
        min_lr=0.001,
        weight_decay=5e-4,
        momentum=0.9,
        name_dataset='MNIST',
        norm=norm,
        bs_info=200,
    )

    cfg_attack = \
        ConfigAttack(
            constraint='inf',
            epsilon=0.3,
            steps=40,
            step_size=0.1,
            norm=norm,
            random_start=True)

    ds_train = MNIST(root=dir_dataset,
                     train=True, download=False,
                     transform=transforms.Compose([
                         transforms.ToTensor(),
                     ]))
    loader_train = DataLoader(ds_train, batch_size=cfg_train.batch_size,
                              shuffle=True, num_workers=10, pin_memory=True)
    ds_test = MNIST(root=dir_dataset,
                    train=False, download=False,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                    ]))
    loader_test = DataLoader(ds_test, batch_size=512,
                             shuffle=True, num_workers=6, pin_memory=True)

    dir_weight = 'weights/mnist/adv'
    record = train_adv(
        model,
        cfg_train,
        cfg_attack,
        loader_train,
        loader_test,
        dir_w=dir_weight,
        record=True)
    print(record)


def cifar10():
    model = PreActResNet18()
    norm = norm_cifar10

    cfg_train = ConfigTrain(
        batch_size=256,
        epoch=200,
        lr_schedule='cosine',
        max_lr=0.1,
        min_lr=0.001,
        weight_decay=1e-4,
        momentum=0.9,
        name_dataset='CIFAR10',
        norm=norm,
        bs_info=200,
    )

    cfg_attack = \
        ConfigAttack(
            constraint='inf',
            epsilon=8 / 255,
            steps=10,
            step_size=0.007843,  # 2.5*eps/steps in robustness-lib
            norm=norm,
            random_start=True)

    ds_train = CIFAR10(root=dir_dataset,
                       train=True, download=False,
                       transform=transforms.Compose([
                           transforms.RandomCrop(32, padding=4),
                           transforms.RandomHorizontalFlip(),
                           transforms.ColorJitter(.25, .25, .25),
                           transforms.RandomRotation(2),
                           transforms.ToTensor(),
                       ]))

    loader_train = DataLoader(ds_train, batch_size=cfg_train.batch_size,
                              shuffle=True, num_workers=10, pin_memory=True)
    ds_test = CIFAR10(root=dir_dataset,
                      train=False, download=False,
                      transform=transforms.Compose([
                          transforms.ToTensor(),
                      ]))

    loader_test = DataLoader(ds_test, batch_size=512,
                             shuffle=True, num_workers=6, pin_memory=True)

    dir_weight = 'weights/cifar10/adv'
    val_acc_rob = train_adv(
        model,
        cfg_train,
        cfg_attack,
        loader_train,
        loader_test,
        val_freq=10,
        dir_w=dir_weight,
        record=True)
    np.save(dir_weight + '/' + 'val_acc_rob.npy', np.array(val_acc_rob))


if __name__ == '__main__':
    cifar10()
