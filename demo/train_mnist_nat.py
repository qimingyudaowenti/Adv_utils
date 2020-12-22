from models.mnist import MnistCls
from utils.config import *
from utils.data_processing import get_loader
from utils.train import train_custom
from utils.eval import test_accuracy

dataset_name = 'MNIST'
model = MnistCls()

cfg_train = ConfigTrain(
    batch_size=64,
    epoch=10,
    max_lr=0.01,
    min_lr=0.001,
    weight_decay=5e-4,
    momentum=0.9,
    dataset_name='MNIST'
)
loader_train = get_loader('MNIST', train=True, batch_size=128, normed=True)

dir_weight = 'weights/mnist'
train_custom(model, cfg_train, loader_train, dir_w=dir_weight)

loader_eval= get_loader('MNIST', train=False, batch_size=256, normed=False)
test_accuracy(model, loader_eval, norm_mnist)
