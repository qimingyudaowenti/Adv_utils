"""
Implementation of attack methods. Running this file as a program will
apply the attack to the models specified by the config file and store
the examples in an .npy file.

ref: https://github.com/MadryLab/robustness/blob/375d9efd6f9ee4bcab83b46b46560669d029e9c4/robustness/attack_steps.py
"""

import torch

from utils.config.attack import ConfigAttack


def clamp_with_tensors(x: torch.Tensor,
                       min_tensor: torch.Tensor, max_tensor: torch.Tensor):
    # eps > 0

    x = torch.where(x > max_tensor, max_tensor, x)
    x = torch.where(x < min_tensor, min_tensor, x)

    return x

def choose_attack_step(config: ConfigAttack):
    if config.constraint == '2':
        return L2Step(config)
    elif config.constraint == 'inf':
        return LinfStep(config)


class AttackerStep:
    def __init__(self, config: ConfigAttack):
        """Attack parameter initialization. The attack performs steps of
           step_size, while always staying within epsilon from the initial
           point."""

        self.epsilon = config.epsilon
        self.steps = config.steps
        self.step_size = config.step_size
        self.rand = config.random_start

        self.criterion = torch.nn.CrossEntropyLoss(reduction='none')

    def project(self, x, orig_input):
        raise NotImplementedError

    def random_perturb(self, x):
        raise NotImplementedError

    def step(self, x, grad):
        raise NotImplementedError


class LinfStep(AttackerStep):
    def __init__(self, config: ConfigAttack):
        super(LinfStep, self).__init__(config)

    def project(self, x, orig_input):
        diff = x - orig_input
        diff = torch.clamp(diff, -self.epsilon, self.epsilon)
        return torch.clamp(diff + orig_input, 0, 1)

    def random_perturb(self, x):
        new_x = x + 2 * (torch.rand_like(x) - 0.5) * self.epsilon
        return torch.clamp(new_x, 0, 1)

    def step(self, x, grad):
        step = torch.sign(grad) * self.step_size
        return x + step


class WeakLinfStep(LinfStep):
    def __init__(self, config: ConfigAttack):
        super(WeakLinfStep, self).__init__(config)
        self.alphas = None  # add in the attack forward process

    def project(self, x, orig_input):
        diff = x - orig_input
        diff = clamp_with_tensors(diff, -self.epsilon * self.alphas,
                                  self.epsilon * self.alphas)

        return clamp_with_tensors(diff + orig_input,
                                  torch.zeros_like(self.alphas),
                                  self.alphas)

    def random_perturb(self, x):
        new_x = x + 2 * (torch.rand_like(x) - 0.5 * self.alphas) \
                * self.epsilon * self.alphas

        return clamp_with_tensors(new_x,
                                  torch.zeros_like(self.alphas),
                                  self.alphas)

    def step(self, x, grad):
        step = torch.sign(grad) * self.step_size * self.alphas
        return x + step


class L2Step(AttackerStep):
    def __init__(self, config: ConfigAttack):
        super(L2Step, self).__init__(config)

    def project(self, x, orig_input):
        diff = x - orig_input
        diff = diff.renorm(p=2, dim=0, maxnorm=self.epsilon)
        return torch.clamp(orig_input + diff, 0, 1)

    def random_perturb(self, x):
        l = len(x.shape) - 1
        rp = torch.randn_like(x)
        rp_norm = rp.view(rp.shape[0], -1).norm(dim=1).view(-1, *([1] * l))
        return torch.clamp(x + self.epsilon * rp / (rp_norm + 1e-10), 0, 1)

    def step(self, x, grad):
        l = len(x.shape) - 1
        g_norm = torch.norm(grad.view(grad.shape[0], -1), dim=1).view(-1, *([1] * l))
        scaled_g = grad / (g_norm + 1e-10)
        return x + scaled_g * self.step_size
