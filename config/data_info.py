import torch

dir_dataset = '~/torchvision_dataset'
dir_weight = 'weights'

norm_mnist = (torch.tensor([0.0]),
              torch.tensor([1.0]))

norm_cifar10 = (torch.tensor([0.4914, 0.4822, 0.4465]),
                torch.tensor([0.2470, 0.2435, 0.2616]))

classes_mnist = ('0', '1', '2', '3', '4',
                 '5', '6', '7', '8', '9')
classes_cifar10 = ('plane', 'car', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck')
