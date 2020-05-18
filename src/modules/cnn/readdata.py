

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

# transforms to apply to the data
trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])


class Testread(unittest.TestCase):

    def testRun(self):

        # MNIST dataset
        train_dataset = torchvision.datasets.MNIST(root=DATA_PATH, train=False, transform=trans, download=False)
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(num_epochs):
            for i, (images, labels) in enumerate(train_loader):
                if i % 100 == 0:
                    print('\n', i, '\n', images.shape, '\n', labels.shape)

        print(i)




