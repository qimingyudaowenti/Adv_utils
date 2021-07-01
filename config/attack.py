from dataclasses import dataclass
from utils.config.data_info import norm_mnist, norm_cifar10

@dataclass
class ConfigAttack:
    constraint: str
    epsilon: float
    num_steps: int
    step_size: float

    norm: tuple = None

    random_start: bool = False
    targeted: bool = False
    do_norm: bool = True
    use_best: bool = True


cfg_attack_mnist = \
    ConfigAttack(
        constraint='inf',
        epsilon=0.3,
        num_steps=40,
        step_size=0.1,
        norm=norm_mnist,
        random_start=True)

cfg_attack_cifar10 = \
    ConfigAttack(
        constraint='inf',
        epsilon=8 / 255,
        num_steps=10,
        step_size=0.007843,  # 2.5*eps/steps in robustness-lib
        norm=norm_cifar10,
        random_start=True)
