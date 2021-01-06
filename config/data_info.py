import torch

dir_dataset = '/home/gy/torchvision_dataset'
dir_weight = 'weights'

norm_mnist = (torch.tensor([0.0]),
              torch.tensor([1.0]))

norm_cifar10 = (torch.tensor([0.4914, 0.4822, 0.4465]),
                torch.tensor([0.2470, 0.2435, 0.2616]))

norm_cifar10_mix = (torch.tensor([0.5, 0.5, 0.5]),
                    torch.tensor([0.24718182, 0.24413806, 0.2669961]))

norm_imagenet = (torch.tensor([0.485, 0.456, 0.406]),
                 torch.tensor([0.229, 0.224, 0.225]))

classes_mnist = ('0', '1', '2', '3', '4',
                 '5', '6', '7', '8', '9')
classes_cifar10 = ('plane', 'car', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck')
