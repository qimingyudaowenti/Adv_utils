import sys

import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.nn import Module
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from tqdm import tqdm

from utils.attack.pgd import AttackerPGD
from utils.config import dir_dataset, ConfigTrain, ConfigAttack
from utils.data_processing import InputNormalize
from utils.data_processing.data_prefetcher import DataPrefetcher
from utils.eval import test_accuracy, test_accuracy_pre, test_robustness
from utils.net_helper import get_device, save_weights, get_lr_scheduler


def train_nat(model: Module,
              cfg_train: ConfigTrain, train_loader: DataLoader,
              val_loader: DataLoader = None, val_freq: int = 1,
              dir_w: str = 'weights', use_amp: bool = False,
              record: bool = False):
    device = get_device()
    model.to(device)
    model.train()
    normalizer = InputNormalize(*cfg_train.norm).to(device)

    if use_amp:
        scaler = GradScaler()
    else:
        scaler = None

    if record:
        record_val = []
    else:
        record_val = None

    bs_print = cfg_train.bs_info

    # ------ loss and optimizer ------
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(),
                          lr=cfg_train.max_lr,
                          weight_decay=cfg_train.weight_decay,
                          momentum=cfg_train.momentum)
    lr_scheduler = get_lr_scheduler(cfg_train.lr_schedule,
                                    cfg_train.max_lr,
                                    cfg_train.epoch)

    tqdm_bar = tqdm(total=len(train_loader) * cfg_train.epoch,
                    ncols=100, file=sys.stdout)

    for i_epoch in range(cfg_train.epoch):
        loss_print = 0
        for i_batch, batch in enumerate(train_loader):
            lr = lr_scheduler(i_epoch + (i_batch + 1) / len(train_loader))
            optimizer.param_groups[0].update(lr=lr)

            inputs = normalizer(batch[0].to(device))
            labels = batch[1].to(device)

            with autocast(enabled=use_amp):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            optimizer.zero_grad()

            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            loss_print += loss.item()

            if i_batch % bs_print == bs_print - 1:
                tqdm_bar.write(f'[{i_epoch + 1:2}, {i_batch + 1:5}] '
                               f'loss: {loss_print / bs_print:<10.4}'
                               f'lr: {lr:.5f}')
                loss_print = 0

            tqdm_bar.update(1)
            tqdm_bar.set_description(f'epoch-{i_epoch + 1}|'
                                     f'batch-{i_batch + 1}|'
                                     f'b-loss:{loss.item():<.4f}|'
                                     f'lr: {lr:.5f}')

        if val_loader is not None:
            # do valuation
            if i_epoch % val_freq == val_freq - 1:
                model.eval()
                _, acc = test_accuracy(model, val_loader, cfg_train.norm)
                if record:
                    record_val.append(acc)
                model.train()

    tqdm_bar.close()

    if dir_w is not None:
        save_weights(dir_w, model, cfg_train)

    return record_val


def train_custom_amp_prefetech(model: Module,
                               cfg_train: ConfigTrain, train_loader: DataLoader,
                               val_loader: DataLoader = None, val_freq: int = 1,
                               dir_w: str = 'weights'):
    device = get_device()
    model.to(device)

    bs_print = cfg_train.bs_info

    # ------ loss and optimizer ------
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(),
                          lr=cfg_train.max_lr,
                          weight_decay=cfg_train.weight_decay,
                          momentum=cfg_train.momentum)
    lr_scheduler = get_lr_scheduler(cfg_train.lr_schedule,
                                    cfg_train.max_lr,
                                    cfg_train.epoch)

    scaler = GradScaler()
    use_amp = True

    tqdm_bar = tqdm(total=len(train_loader) * cfg_train.epoch,
                    ncols=100, file=sys.stdout)

    for i_epoch in range(cfg_train.epoch):
        prefetcher = DataPrefetcher(train_loader, *cfg_train.norm)
        inputs, labels = prefetcher.next()

        loss_print = 0
        i_batch = -1
        while inputs is not None:
            i_batch += 1
            lr = lr_scheduler(i_epoch + (i_batch + 1) / len(train_loader))
            optimizer.param_groups[0].update(lr=lr)

            with autocast(enabled=use_amp):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_print += loss.item()

            if i_batch % bs_print == bs_print - 1:
                tqdm_bar.write(f'[{i_epoch + 1:2}, {i_batch + 1:5}] '
                               f'loss: {loss_print / bs_print:<10.4}'
                               f'lr: {lr:.5f}')
                loss_print = 0

            tqdm_bar.update(1)
            tqdm_bar.set_description(f'epoch-{i_epoch + 1}|'
                                     f'batch-{i_batch + 1}|'
                                     f'b-loss:{loss.item():<.4f}|'
                                     f'lr: {lr:.5f}')

            inputs, labels = prefetcher.next()

        if val_loader is not None:
            # do valuation
            if i_epoch % val_freq == val_freq - 1:
                model.eval()
                test_accuracy_pre(model, val_loader, cfg_train.norm)
                model.train()

    tqdm_bar.close()

    if dir_w is not None:
        save_weights(dir_w, model, cfg_train)


def train_adv(model: Module,
              cfg_train: ConfigTrain, cfg_attack: ConfigAttack,
              train_loader: DataLoader, val_loader: DataLoader = None,
              val_freq: int = 1, dir_w: str = 'weights',
              use_amp: bool = False, record: bool = False):
    assert cfg_train.norm == cfg_attack.norm

    device = get_device()
    model.to(device)
    model.train()
    attacker = AttackerPGD(model, cfg_attack).to(device)

    if use_amp:  # TODO: amp in adv example?
        scaler = GradScaler()
    else:
        scaler = None

    if record:
        record_val = [[], []]
    else:
        record_val = None

    bs_print = cfg_train.bs_info

    # ------ loss and optimizer ------
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(),
                          lr=cfg_train.max_lr,
                          weight_decay=cfg_train.weight_decay,
                          momentum=cfg_train.momentum)
    lr_scheduler = get_lr_scheduler(cfg_train.lr_schedule,
                                    cfg_train.max_lr,
                                    cfg_train.epoch)

    tqdm_bar = tqdm(total=len(train_loader) * cfg_train.epoch,
                    ncols=100, file=sys.stdout)

    for i_epoch in range(cfg_train.epoch):
        loss_print = 0
        for i_batch, batch in enumerate(train_loader):
            lr = lr_scheduler(i_epoch + (i_batch + 1) / len(train_loader))
            optimizer.param_groups[0].update(lr=lr)

            inputs = batch[0].to(device)
            labels = batch[1].to(device)
            inputs_adv, inputs_adv_normed = attacker(inputs, labels)

            with autocast(enabled=use_amp):
                outputs = model(inputs_adv_normed)
                loss = criterion(outputs, labels)

            optimizer.zero_grad()

            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            loss_print += loss.item()

            if i_batch % bs_print == bs_print - 1:
                tqdm_bar.write(f'[{i_epoch + 1:2}, {i_batch + 1:5}] '
                               f'loss: {loss_print / bs_print:<10.4}'
                               f'lr: {lr:.5f}')
                loss_print = 0

            tqdm_bar.update(1)
            tqdm_bar.set_description(f'epoch-{i_epoch + 1}|'
                                     f'batch-{i_batch + 1}|'
                                     f'b-loss:{loss.item():<.4f}|'
                                     f'lr: {lr:.5f}')

        if val_loader is not None:
            # do valuation
            if i_epoch % val_freq == val_freq - 1:
                model.eval()
                _, acc = test_accuracy(model, val_loader, cfg_train.norm)
                _, rob = test_robustness(model, val_loader, cfg_attack)
                if record:
                    record_val[0].append(acc)
                    record_val[1].append(rob)
                model.train()

    tqdm_bar.close()

    if dir_w is not None:
        save_weights(dir_w, model, cfg_train)

    return record_val


def do_cifar10_train():
    from models.cifar10_resnet import ResNet18

    dataset_name = 'CIFAR10'

    cfg_loader_train = ConfigTrain(
        batch_size=128,
        epoch=300,
        start_lr=0.1,
        end_lr=0.001,
        weight_decay=5e-4,
        momentum=0.9
    )

    model = ResNet18()

    # models.load_state_dict(torch.load('/home/geyao/robust_analyse/weights/'
    #                                  'CIFAR10/ResNet18/natural/2020-08-27-12-57-31_2020-08-27-12-57-31_100_128_0.01_0.01_.pth'))

    cifar10_aug = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(.25, .25, .25),
        transforms.RandomRotation(2),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616))
    ])

    train_set = datasets.CIFAR10(
        root=dir_dataset, train=True, download=False, transform=cifar10_aug)
    train_loader = DataLoader(
        train_set, batch_size=cfg_loader_train.batch_size, shuffle=True, num_workers=4)

    train_nat(model, cfg_loader_train, train_loader, 'weights/CIFAR10/ResNet18/natural')

    cfg_loader_test = ConfigTrain(
        train=False,
        batch_size=128,
        shuffle=False,
        num_workers=4,
        do_norm=True
    )
    test_accuracy(model, device=get_device(),
                  dataset_name=dataset_name, loader_param=cfg_loader_test)


def do_mnist_train():
    from models.mnist import MnistCls

    dataset_name = 'MNIST'

    cfg_loader_train = ConfigTrain(
        batch_size=64,
        epoch=10,
        max_lr=0.01,
        min_lr=0.001,
        weight_decay=5e-4,
        momentum=0.9
    )

    model = MnistCls()

    mnist_aug = transforms.Compose([
        transforms.ToTensor()
    ])

    train_set = datasets.MNIST(
        root=dir_dataset, train=True, download=False, transform=mnist_aug)
    train_loader = DataLoader(
        train_set, batch_size=cfg_loader_train.batch_size, shuffle=True, num_workers=4)

    train_nat(model, cfg_loader_train, train_loader, 'weights/MNIST/natural')

    cfg_loader_test = LoaderConfig(
        train=False,
        batch_size=128,
        shuffle=False,
        num_workers=4,
        do_norm=True
    )
    test_accuracy(model, dataset_name=dataset_name, loader_param=cfg_loader_test)
