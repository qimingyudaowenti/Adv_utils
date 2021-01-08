import sys

import torch
from tqdm import tqdm

from utils.attack.pgd import AttackerPGD
from utils.config import ConfigAttack
from utils.data_processing import InputNormalize
from utils.net_helper import get_device


def get_adv_examples(model,
                     imgs: torch.Tensor, labels: torch.Tensor,
                     cfg_attack: ConfigAttack):
    attacker = AttackerPGD(model, cfg_attack)

    imgs = imgs.to('cuda')
    labels = labels.to('cuda')
    attacker.to('cuda')

    im_adv, im_adv_normed = attacker(imgs, labels)

    return im_adv, im_adv_normed


# the 'norm' in cfg_attack should be 'source_norm'
def test_adv_transfer(source_model, source_norm, target_model, target_norm,
                      data_loader, cfg_attack: ConfigAttack):
    device = get_device()

    source_model.to(device)
    source_model.eval()
    target_model.to(device)
    source_model.eval()

    attacker = AttackerPGD(source_model, cfg_attack).to(device)
    s_normalizer = InputNormalize(*source_norm).to(device)
    t_normalizer = InputNormalize(*target_norm).to(device)

    correct = 0
    total = 0

    tqdm_bar = tqdm(total=len(data_loader), ncols=100, file=sys.stdout)
    for i_batch, batch in enumerate(data_loader):
        inputs = s_normalizer(batch[0].to(device))
        labels = batch[1].to(device)

        im_adv, _ = attacker(inputs, labels)

        with torch.no_grad():
            outputs = target_model(t_normalizer(im_adv))

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        tqdm_bar.update(1)
        tqdm_bar.set_description_str(f'Test on {total} examples. '
                                     f'Adv transfer acc-{correct / total:.2%}')

    tqdm_bar.close()
