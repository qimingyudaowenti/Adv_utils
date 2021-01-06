import torch

from utils.attack.pgd import AttackerPGD
from utils.config import ConfigAttack


def get_adv_examples(model,
                     imgs: torch.Tensor, labels: torch.Tensor,
                     cfg_attack: ConfigAttack):
    attacker = AttackerPGD(model, cfg_attack)

    imgs = imgs.to('cuda')
    labels = labels.to('cuda')
    attacker.to('cuda')

    im_adv, im_adv_normed = attacker(imgs, labels)

    return im_adv, im_adv_normed
