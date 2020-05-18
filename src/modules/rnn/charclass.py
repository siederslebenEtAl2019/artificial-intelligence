# from
# https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html

from __future__ import unicode_literals, print_function, division

import glob
import math
import os
import random
import string
import time
import torch
import unicodedata
import unittest
from io import open

import torch.nn as nn

device = torch.device("cpu")
device = torch.device("cuda")


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def findFiles(path):
    return glob.glob(path)


def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)  # p : 128 x 185
        self.i2o = nn.Linear(input_size + hidden_size, output_size)  # p : 18 x 185
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size, device=device)


class Charclassification(object):
    def __init__(self):
        self.all_letters = string.ascii_letters + " .,;'"
        self.n_letters = len(self.all_letters)
        self.all_categories = []
        self.category_lines_train = {}
        self.category_lines_test = {}
        self.readCategories()
        self.n_categories = len(self.all_categories)
        self.n_hidden = 128
        self.learning_rate = 0.005  # If you set this too high, it might explode. If too low, it might not learn
        self.criterion = nn.NLLLoss()
        self.rnn = RNN(self.n_letters, self.n_hidden, self.n_categories)
        if device == torch.device("cuda"):
            self.rnn.cuda(device)

    def saveNN(self, path):
        torch.save(self.rnn.state_dict(), path + 'cc_rnn.ckpt')

    def loadNN(self, path):
        self.rnn.load_state_dict(torch.load(path + 'cc_rnn.ckpt'))
        self.rnn.eval()

    # Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
    def unicodeToAscii(self, s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
            and c in self.all_letters
        )

    # Read a file and split into lines
    def readLines(self, filename):
        lines = open(filename, encoding='utf-8').read().strip().split('\n')
        return [self.unicodeToAscii(line) for line in lines]

    def readCategories(self):
        print()
        # Build the category_lines dictionary
        # build two lists of names per language: one for training, one for testing
        for filename in findFiles('data/names/*.txt'):
            category = os.path.splitext(os.path.basename(filename))[0]
            self.all_categories.append(category)
            lines = self.readLines(filename)
            self.category_lines_train[category] = lines[:math.floor(len(lines) * 0.8)]
            self.category_lines_test[category] = lines[math.floor(len(lines) * 0.8):]

    def categoryFromOutput(self, output):
        """
        :param output: a probability density on 18 categories
        :return: the category with the highest probability
        """
        top_n, top_i = output.topk(1)
        category_i = top_i[0].item()
        return self.all_categories[category_i], category_i

    def letterToIndex(self, letter):
        """
        :param letter: an ascii letter
        :return: its index in all_letters
        """
        return self.all_letters.find(letter)

    def lineToTensor(self, line):
        """
        :param line: a line = a name
        :return: a <line_length x 1 x n_letters> tensor
        """
        tensor = torch.zeros(len(line), 1, self.n_letters, device=device)
        for li, letter in enumerate(line):
            tensor[li][0][self.letterToIndex(letter)] = 1
        return tensor

    def randomTrainingExample(self):
        """
        :return: a category (language) and a line (name) as string and tensor
        """
        category = randomChoice(self.all_categories)
        line = randomChoice(self.category_lines_train[category])
        category_tensor = torch.tensor([self.all_categories.index(category)], device=device, dtype=torch.long)
        line_tensor = self.lineToTensor(line)
        return category, line, category_tensor, line_tensor

    def randomTestExample(self):
        """
        :return: a category (language) and a line (name) as string and tensor
        """
        category = randomChoice(self.all_categories)
        line = randomChoice(self.category_lines_test[category])
        category_tensor = torch.tensor([self.all_categories.index(category)], device=device, dtype=torch.long)
        line_tensor = self.lineToTensor(line)
        return category, line, category_tensor, line_tensor

    def train(self, category_tensor, line_tensor, repetitions):
        """
        :param category_tensor: true category of line, e.g. [3]
        :param line_tensor: a name, e.g. ['Jones'] as 5 x 1 x 57 tensor
        :return: output: a 1 x 18 tensor = probability density on 18 categories
        :return: loss: the NLL difference between given category and output

        rnn is applied n times (n = length of line_tensor, 5 for 'Jones'),
        tracked by torch.
        Side effect: one optimization step applied to rnn.
        """
        for cnt in range(repetitions):
            self.rnn.zero_grad()
            hidden = self.rnn.initHidden()

            for i in range(line_tensor.size()[0]):
                output, hidden = self.rnn(line_tensor[i], hidden)

            loss = self.criterion(output, category_tensor)
            loss.backward()

            # Add parameters' gradients to their values, multiplied by learning rate
            for p in self.rnn.parameters():
                p.data.add_(-self.learning_rate, p.grad.data)

        return output, loss.item()

    def trainloop(self, n_iters, repetitions):
        # generate n_iter training examples, randomly chosen from the data set
        # apply train to each of them.

        all_losses = []

        for iter in range(1, n_iters + 1):
            _, _, category_tensor, line_tensor = self.randomTrainingExample()
            output, loss = self.train(category_tensor, line_tensor, repetitions)
            all_losses.append(loss)

        return all_losses

    def evaluate(self, line_tensor):
        """
        :param line_tensor: line_tensor: a name, e.g. ['Jones'] as 5 x 1 x 57 tensor
        :return: output: a 1 x 18 tensor = probability density on 18 categories
        """
        hidden = self.rnn.initHidden()
        output = None

        for i in range(line_tensor.size()[0]):
            output, hidden = self.rnn(line_tensor[i], hidden)

        return output

    def testloop(self, n_confusion):
        # Go through a bunch of examples and record which are correctly guessed

        # Keep track of correct guesses in a confusion matrix
        confusion = torch.zeros(self.n_categories, self.n_categories, device=device)

        for i in range(n_confusion):
            category, line, category_tensor, line_tensor = self.randomTestExample()
            output = self.evaluate(line_tensor)
            guess, guess_i = self.categoryFromOutput(output)
            category_i = self.all_categories.index(category)
            confusion[category_i][guess_i] += 1

        print()
        allguesses = 0
        allhits = 0
        for i, cat in enumerate(self.all_categories):
            thesehits = confusion[i, i].item()
            theseguesses = confusion[i].sum().item()
            print(f'{cat:10} {theseguesses:8} {thesehits:8} {thesehits / theseguesses:8.3f}')
            allguesses += theseguesses
            allhits += thesehits

        print(f"{'all':10} {allguesses:8} {allhits:8} {allhits / allguesses:8.3f}")

        # # Normalize by dividing every row by its sum
        # for i in range(self.n_categories):
        #     confusion[i] = confusion[i] / confusion[i].sum()
        #
        # # Set up plot
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # cax = ax.matshow(confusion.numpy())
        # fig.colorbar(cax)
        #
        # # Set up axes
        # ax.set_xticklabels([''] + self.all_categories, rotation=90)
        # ax.set_yticklabels([''] + self.all_categories)
        #
        # # Force label at every tick
        # ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        # ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
        #
        # # sphinx_gallery_thumbnail_number = 2
        # plt.show()


class TestClassification((unittest.TestCase)):
    def testTrain(self):
        cc = Charclassification()
        category, line, category_tensor, line_tensor = cc.randomTrainingExample()
        output, loss = cc.train(category_tensor, line_tensor, 1)
        print('\n', output, '\n', loss)

    def testRandom(self):
        print()
        cc = Charclassification()
        print(list(enumerate(cc.all_categories)))
        for _ in range(1):
            category, line, category_tensor, line_tensor = cc.randomTrainingExample()
            print('category =', category, '/ line =', line)
            print(category_tensor)
            for i in range(line_tensor.size()[0]):
                print(line_tensor[i, 0])

    def testTrainloop(self):
        cc = Charclassification()
        cc.trainloop(3000, 1)
        cc.saveNN('models/')

    def testTestloop(self):
        cc = Charclassification()
        cc.loadNN('models/')
        cc.testloop(1000)
