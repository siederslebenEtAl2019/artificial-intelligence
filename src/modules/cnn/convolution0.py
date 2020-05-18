#
# https://adventuresinmachinelearning.com/convolutional-neural-networks-tutorial-in-pytorch/
# fast original, mit cuda
# 29.4.2020
#

import torch
import torchvision
import torch.nn as nn

import unittest

from torchvision import transforms
from torch.utils.data import DataLoader


# Hyperparameters
num_epochs = 5
batch_size = 100
learning_rate = 0.001

# device = torch.device("cpu")
device = torch.device("cuda")

DATA_PATH = "/samples"
MODEL_STORE_PATH = "/models/"

# transforms to apply to the data
trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(7 * 7 * 64, 1000)
        self.fc2 = nn.Linear(1000, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


class Testmodel(unittest.TestCase):
    def testTrain(self):

        # MNIST dataset
        train_dataset = torchvision.datasets.MNIST(root=DATA_PATH, train=True, transform=trans, download=True)
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

        model = ConvNet()
        model.cuda(device)

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        print()

        # Train the model
        total_steps = len(train_loader)   # == len(data_set) / batch_size
        loss_list = []
        acc_list = []
        for epoch in range(num_epochs):
            for i, (images, labels) in enumerate(train_loader):    # i in range(total_steps)
                # Run the forward pass
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                loss_list.append(loss.item())

                # Backprop and perform Adam optimisation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Track the accuracy
                total = labels.size(0)   # == batch_size
                _, predicted = torch.max(outputs.data, 1)
                correct = (predicted == labels).sum().item()
                acc_list.append(correct / total)

                if (i + 1) % 100 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                          .format(epoch + 1, num_epochs, i + 1, total_steps, loss.item(),
                                  (correct / total) * 100))

                # Save the model and plot
                torch.save(model.state_dict(), MODEL_STORE_PATH + 'conv_net_model.ckpt')

    def testModel(self):
        test_dataset = torchvision.datasets.MNIST(root=DATA_PATH, train=False, transform=trans)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

        model = ConvNet()
        model.cuda(device)
        model.load_state_dict(torch.load(MODEL_STORE_PATH + 'conv_net_model.ckpt'))

        model.eval()

        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:

                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print('Test Accuracy on the 10000 test images: {} %'.format((correct / total) * 100))