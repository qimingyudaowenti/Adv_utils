from dataclasses import dataclass


@dataclass
class ConfigAttack:
    constraint: str
    epsilon: float
    steps: int
    step_size: float

    random_start: bool = False
    targeted: bool = False
    do_norm: bool = True
    use_best: bool = True


cfg_attack_mnist = \
    ConfigAttack(
        constraint='inf',
        epsilon=0.3,
        steps=40,
        step_size=0.1,
        random_start=True)

cfg_attack_cifar10 = \
    ConfigAttack(
        constraint='inf',
        epsilon=8 / 255,
        steps=10,
        step_size=0.007843,  # 2.5*eps/steps in robustness-lib
        random_start=True)
