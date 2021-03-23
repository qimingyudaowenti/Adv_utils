import torch

dir_dataset = '/home/gy/torchvision_dataset'
dir_weight = 'weights'

# norm: (mean, std)
norm_none_mnist = (torch.tensor([0.0]),
                   torch.tensor([1.0]))

norm_none_cifar10 = (torch.tensor([0.0, 0.0, 0.0]),
                     torch.tensor([1.0, 1.0, 1.0]))

norm_mnist = (torch.tensor([0.1306604762738429]),
              torch.tensor([0.30810780385646264]))

norm_mnist_mix = (torch.tensor([0.5]),
                  torch.tensor([0.48098034]))

norm_cifar10 = (torch.tensor([0.49139968, 0.48215841, 0.44653091]),
                torch.tensor([0.24703223, 0.24348513, 0.26158784]))

norm_cifar10_mix = (torch.tensor([0.5, 0.5, 0.5]),
                    torch.tensor([0.2471819, 0.24413793, 0.26699652]))

norm_imagenet = (torch.tensor([0.485, 0.456, 0.406]),
                 torch.tensor([0.229, 0.224, 0.225]))

classes_mnist = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

classes_cifar10 = (
    'plane',
    'car',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck')
