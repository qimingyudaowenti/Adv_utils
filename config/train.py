from dataclasses import dataclass
import torch


@dataclass()
class ConfigTrain:
    batch_size: int = 64
    epoch: int = 1

    lr_schedule: str = ''
    max_lr: float = 0.01
    min_lr: float = 0.0

    weight_decay: float = 5e-5
    momentum: float = 0.9

    num_workers: int = 6

    dataset_name: str = ''
    dataset_norm: tuple = None
    model_name: str = None
    bs_info: int = 100  # print info every x batches


cfg_train_mnist = ConfigTrain(
    batch_size=64,
    epoch=5,
    max_lr=0.01,
    min_lr=0.001,
    weight_decay=1e-5,
    momentum=0.9,
    dataset_name='MNIST',
    dataset_norm=(torch.tensor([0.0]),
                  torch.tensor([1.0]))
)

cfg_train_cifar10 = ConfigTrain(
    batch_size=128,
    epoch=200,
    max_lr=0.1,
    min_lr=0.001,
    weight_decay=1e-4,
    momentum=0.9,
    dataset_name='CIFAR10',
    dataset_norm=(torch.tensor([0.4914, 0.4822, 0.4465]),
                  torch.tensor([0.2470, 0.2435, 0.2616]))
)
