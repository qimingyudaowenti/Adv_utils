import torch

from utils.config.attack import ConfigAttack
from utils.attack.step import choose_attack_step
from utils.attack.norm import InputNormalize
from utils.attack.step import WeakLinfStep


class AttackerPGD(torch.nn.Module):
    def __init__(self, model, config: ConfigAttack, data_norm: tuple):
        super(AttackerPGD, self).__init__()
        self.model = model
        self.attack_config = config
        self.step = choose_attack_step(config)

        self.criterion = torch.nn.CrossEntropyLoss(reduction='none')

        self.normalizer = InputNormalize(*data_norm)

        self.random_start = config.random_start
        self.targeted = config.targeted
        self.do_norm = config.do_norm
        self.use_best = config.use_best

    def forward(self, input_x, label_y):

        def calc_loss(x, y):
            """ Calculates the loss of an input with respect to target labels
            Uses custom loss (if provided) otherwise the criterion
            """

            if self.do_norm:
                x = self.normalizer(x)
            output = self.model(x)

            return self.criterion(output, y), output

        # Main function for making adversarial examples
        def get_adv_examples(x):
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
                losses, out = calc_loss(x, label_y)
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

            losses, _ = calc_loss(x, label_y)
            args = [losses, best_loss, x, best_x]
            best_loss, best_x = replace_best(*args)
            return best_x

        orig_input = input_x.detach()

        # Multiplier for gradient ascent [untargeted] or descent [targeted]
        m = -1 if self.targeted else 1
        adv_ret = get_adv_examples(input_x)
        adv_ret_normalized = self.normalizer(adv_ret)

        return adv_ret, adv_ret_normalized


class AttackerPGDWeak(AttackerPGD):
    def __init__(self, model, config: ConfigAttack, data_norm: tuple):
        super(AttackerPGDWeak, self).__init__(model, config, data_norm)
        self.step = WeakLinfStep(config)

    def forward(self, input_x, label_y, alphas):
        self.step.alphas = alphas
        return super(AttackerPGDWeak, self).forward(input_x, label_y)
