import datetime
from pathlib import Path

import numpy as np
import torch
from torch.nn import Module

from utils.config import *


def set_seed(seed: int = 0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_grad(model, x, y, criterion):
    x = x.to('cuda')
    x.requires_grad = True
    y = y.to('cuda')

    outputs = model(x)
    loss = criterion(outputs, y)
    loss.backward()

    return x.grad


def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device('cpu')

    print(f'Find device: {device}.')

    return device


def add_time_prefix(s):
    now_time = datetime.datetime.now()

    return f'{now_time.strftime("%Y-%m-%d-%H-%M-%S")}_' + s


def save_weights(save_dir: str, model: Module,
                 cfg_train: ConfigTrain = None, pth_name: str = None):
    if pth_name is None:
        pth_name = '.pth'
    else:
        pth_name += '.pth'

    if cfg_train is not None:
        pth_name = f'{cfg_train.epoch}_{cfg_train.batch_size}_' \
                   f'{cfg_train.max_lr}_{cfg_train.min_lr}_' + pth_name

    pth_name = add_time_prefix(pth_name)
    if cfg_train.model_name is not None:
        pth_name = cfg_train.model_name + '_' + pth_name

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    weights_path = save_dir / pth_name
    torch.save(model.state_dict(), weights_path)
    print('Save weights in {}\n'.format(weights_path))


def get_lr_scheduler(schedule_type: str, lr_max: float, epochs: int,
                     lr_one_drop: float = 0.01, lr_drop_epoch: int = 100):
    # TODO: add min_lr
    if schedule_type == 'superconverge':
        def lr_schedule(t):
            return np.interp([t], [0, epochs * 2 // 5, epochs],
                             [0, lr_max, 0])[0]
    elif schedule_type == 'piecewise':
        def lr_schedule(t):
            if t / epochs < 0.5:
                return lr_max
            elif t / epochs < 0.75:
                return lr_max / 10.
            else:
                return lr_max / 100.
    elif schedule_type == 'linear':
        def lr_schedule(t):
            return np.interp([t], [0, epochs // 3, epochs * 2 // 3, epochs],
                             [lr_max, lr_max, lr_max / 10, lr_max / 100])[0]
    elif schedule_type == 'onedrop':
        def lr_schedule(t):
            if t < lr_drop_epoch:
                return lr_max
            else:
                return lr_one_drop
    elif schedule_type == 'multipledecay':
        def lr_schedule(t):
            return lr_max - (t // (epochs // 10)) * (lr_max / 10)
    elif schedule_type == 'cosine':
        def lr_schedule(t):
            return lr_max * 0.5 * (1 + np.cos(t / epochs * np.pi))
    else:
        raise ValueError(f"Unknown learning rate schedule: {schedule_type}")

    return lr_schedule
