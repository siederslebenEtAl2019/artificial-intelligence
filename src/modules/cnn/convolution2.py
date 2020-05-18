
import numpy as np
import torch
import torchvision
import torch.nn as nn

import unittest


from torchvision import transforms
from torch.utils.data import DataLoader


# Hyperparameters
num_epochs = 1
batch_size = 100
learning_rate = 0.001

DATA_PATH = "/samples"
MODEL_STORE_PATH = "/models/"


seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

#The compose function allows for multiple transforms
#transforms.ToTensor() converts our PILImage to a tensor of shape (C x H x W) in the range [0,1]
#transforms.Normalize(mean,std) normalizes a tensor to a (mean, std) for (R, G, B)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_set = torchvision.datasets.CIFAR10(root='./cifardata', train=True, download=True, transform=transform)
test_set = torchvision.datasets.CIFAR10(root='./cifardata', train=False, download=True, transform=transform)



class ConvNet(nn.Module):
    pass


class Testmodel(unittest.TestCase):
    pass