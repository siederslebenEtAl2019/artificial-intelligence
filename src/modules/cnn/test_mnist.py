#
# https://adventuresinmachinelearning.com/convolutional-neural-networks-tutorial-in-pytorch/
# fast original, mit cuda
# 29.4.2020
#
import time
import torch
import torchvision
import unittest

import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

DATA_PATH = "test/samples"
MODEL_STORE_PATH = "test/models"

# transforms to apply to the data
trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(7 * 7 * 64, 1000)
        self.fc2 = nn.Linear(1000, 10)

    def forward(self, x):  # (100, 1, 28, 28)
        out = self.layer1(x)  # (100, 32, 14, 14) pooling
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


def makeTwoLayers(n, k1, k2):
    return nn.Sequential(
        nn.Linear(n, k1),
        nn.Sigmoid(),
        nn.Linear(k1, k2)
    )


def getHash(*args):
    return str(sum([hash(arg) * 10 ** k for k, arg in enumerate(args)]))


def train_mnist(data_size, num_epochs, batch_size, learning_rate, num_opt, device):
    """
    :param data_size: max number of entries to to_ints
    :param num_epochs: number of epochs
    :param batch_size: batch_sizes
    :param learning_rate: learning rate
    :param num_opt: number of optimizations per batch
    :param device: cpu or cuda
    :return: the model
    """

    # MNIST dataset
    train_dataset = torchvision.datasets.MNIST(root=DATA_PATH, train=True, transform=trans, download=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    model = ConvNet()
    if device == torch.device("cuda:0"):
        model.cuda(device)

    # Loss and optimizer
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    outputs = None
    loss = None

    # Train the model
    num_batches = len(train_loader)  # == len(data_set) / batch_size
    protocol = [[data_size, num_epochs, batch_size, learning_rate, num_opt, device]]

    for epoch in range(num_epochs):
        counter = 0  # counter number of processed entries
        for i, (images, labels) in enumerate(train_loader):  # i in range(num_batches)
            counter += num_batches
            if counter >= data_size:
                break

            images = images.to(device)
            labels = labels.to(device)

            # optimize this batch
            for _ in range(num_opt):
                # Run the forward pass
                outputs = model(images)
                loss = loss(outputs, labels)

                # Backprop and perform Adam optimisation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Track the accuracy
            total = labels.size(0)  # == batch_size
            _, predicted = torch.max(outputs.data, 1)
            accuracy = (predicted == labels).sum().item() / total
            protocol.append([epoch, i, loss.item(), accuracy])

    return model, protocol


def test_mnist(batch_size, device, modelpath=None, model=None):
    test_dataset = torchvision.datasets.MNIST(root=DATA_PATH, train=False, transform=trans)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    if model is None:
        model = ConvNet()
        if device == torch.device("cuda:0"):
            model.cuda(device)
        model.load_state_dict(torch.load(modelpath))

    model.eval()
    count = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            count += labels.size(0)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = (correct / total) * 100
        # print('Test Accuracy on {} test images: {}%'.format(count, accuracy))
        return accuracy


class TestMNIST(unittest.TestCase):
    def test1(self):
        data_size = 60000
        num_epochs = 1
        batch_size = 100
        learning_rate = 0.001
        num_opt = 1
        device = torch.device("cuda:0")
        # device = torch.device("cpu")

        thishash = getHash(data_size, num_epochs, batch_size, learning_rate, num_opt, device)
        modelpath = MODEL_STORE_PATH + 'conv' + thishash + '.ckpt'

        model, protocol = train_mnist(data_size, num_epochs, batch_size, learning_rate, num_opt, device)
        torch.save(model.state_dict(), modelpath)

        batch_size = 10000
        test_mnist(batch_size, device, modelpath=modelpath)

    def test2(self):
        data_size = 5000
        num_epochs = 1
        batch_size = 100
        learning_rate = 0.001
        num_opt = 1
        device = torch.device("cuda:0")
        # device = torch.device("cpu")

        thishash = getHash(data_size, num_epochs, batch_size, learning_rate, num_opt, device)
        modelpath = MODEL_STORE_PATH + 'conv' + thishash + '.ckpt'

        model, protocol = train_mnist(data_size, num_epochs, batch_size, learning_rate, num_opt, device)
        torch.save(model.state_dict(), modelpath)

        batch_size = 10000
        test_mnist(batch_size, device, modelpath=modelpath)

    def test3(self):
        data_size = 1000
        num_epochs = 1
        batch_size = 100
        learning_rate = 0.1
        num_opt = 1000
        device = torch.device("cuda:0")
        # device = torch.device("cpu")

        thishash = getHash(data_size, num_epochs, batch_size, learning_rate, num_opt, device)
        modelpath = MODEL_STORE_PATH + 'conv' + thishash + '.ckpt'

        model, protocol = train_mnist(data_size, num_epochs, batch_size, learning_rate, num_opt, device)
        torch.save(model.state_dict(), modelpath)

        batch_size = 1000
        test_mnist(batch_size, device, modelpath=modelpath)

    def testBulk(self):
        device = torch.device("cuda:0")
        # device = torch.device("cpu")
        result = []

        # for data_size in [10, 1000, 10000, 30000]:
        for data_size in [2000]:
            for num_epochs in range(2):
                # for batch_size in [10, 100, 1000]:
                for batch_size in [1000]:
                    for learning_rate in [0.001]:
                        for num_opt in range(4):
                            tic = time.perf_counter()
                            model, _ = train_mnist(data_size, num_epochs, batch_size, learning_rate, num_opt, device)
                            toc = time.perf_counter()
                            acc = test_mnist(1000, device, model=model)
                            result.append([data_size, num_epochs, batch_size,
                                           learning_rate, num_opt, acc, toc - tic])

        print()
        print(f"{'data_size':>12} {'num_epochs':>12} {'batch_size':>12} {'lrng_rate':>12} "
              f"{'num_opts':>12} {'accuracy':>12} {'duration':>12}")

        for row in result:
            print(f'{row[0]:12} {row[1]:12} {row[2]:12} {row[3]:12.3f} {row[4]:12} {row[5]:12.2f} {row[6]:12.2f}')
