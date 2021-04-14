import torch

from utils.config.attack import ConfigAttack
from utils.attack.step import choose_attack_step
from utils.data_processing.norm import InputNormalize
from utils.attack.step import WeakLinfStep


class AttackerPGD(torch.nn.Module):
    def __init__(self, model, config: ConfigAttack):
        super(AttackerPGD, self).__init__()
        self.model = model
        self.attack_config = config
        self.step = choose_attack_step(config)

        self.criterion = torch.nn.CrossEntropyLoss(reduction='none')

        self.normalizer = InputNormalize(*config.norm)

        self.random_start = config.random_start
        self.targeted = config.targeted
        self.do_norm = config.do_norm
        self.use_best = config.use_best

    def calc_loss(self, x, y):
        """ Calculates the loss of an input with respect to target labels
        Uses custom loss (if provided) otherwise the criterion
        """

        if self.do_norm:
            x = self.normalizer(x)
        output = self.model(x)

        return self.criterion(output, y), output

        # Main function for making adversarial examples

    def get_adv_examples(self, x, label_y, orig_input):
        # Multiplier for gradient ascent [untargeted] or descent [targeted]
        m = -1 if self.targeted else 1

        if self.random_start:
            x = self.step.random_perturb(x)

        # Keep track of the "best" (worst-case) loss and its
        # corresponding input
        best_loss = None
        best_x = None

        # A function that updates the best loss and best input
        def replace_best(loss, bloss, x, bx):
            if bloss is None:
                bx = x.clone().detach()
                bloss = loss.clone().detach()
            else:
                replace = m * bloss < m * loss
                bx[replace] = x[replace].clone().detach()
                bloss[replace] = loss[replace]

            return bloss, bx

        # PGD iterates
        for _ in range(self.attack_config.steps):
            x = x.clone().detach().requires_grad_(True)
            losses, out = self.calc_loss(x, label_y)
            assert losses.shape[0] == x.shape[0], \
                'Shape of losses must match input!'

            loss = torch.mean(losses)

            grad, = torch.autograd.grad(m * loss, [x])

            with torch.no_grad():
                args = [losses, best_loss, x, best_x]
                best_loss, best_x = replace_best(*args) if self.use_best else (losses, x)

                x = self.step.step(x, grad)
                x = self.step.project(x, orig_input)

        # Save computation (don't compute last loss) if not use_best
        if not self.use_best:
            ret = x.clone().detach()
            return ret

        losses, _ = self.calc_loss(x, label_y)
        args = [losses, best_loss, x, best_x]
        best_loss, best_x = replace_best(*args)
        return best_x

    def forward(self, input_x, label_y):

        orig_input = input_x.detach()

        adv_ret = self.get_adv_examples(input_x, label_y, orig_input)
        adv_ret_normalized = self.normalizer(adv_ret)

        return adv_ret, adv_ret_normalized


class AttackerPGDBound(AttackerPGD):
    def __init__(self, model, config: ConfigAttack):
        super(AttackerPGDBound, self).__init__(model, config)

    def forward(self, input_x, label_y, imgs_bound=None):

        if imgs_bound is None:
            adv_ret, adv_ret_normalized = super(AttackerPGDBound, self).forward(input_x, label_y)
        else:
            orig_input = imgs_bound.detach()
            adv_ret = self.get_adv_examples(input_x, label_y, orig_input)
            adv_ret_normalized = self.normalizer(adv_ret)

        return adv_ret, adv_ret_normalized
