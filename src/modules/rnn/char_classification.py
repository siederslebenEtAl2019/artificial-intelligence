# from
# https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html

from __future__ import unicode_literals, print_function, division

import glob
import math
import os
import random
import string
import time
import unittest
from io import open

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch
import torch.nn as nn
import unicodedata

# device = torch.device("cuda")
device = torch.device("cpu")

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)
category_lines = {}
all_categories = []


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def findFiles(path):
    return glob.glob(path)


# Read a file and split into lines
def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]


# Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )


# Find letter index from all_letters, e.g. "a" = 0
def letterToIndex(letter):
    return all_letters.find(letter)


# Just for demonstration, turn a letter into a <1 x n_letters> Tensor
def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor


# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor


def categoryFromOutput(output):
    """
    :param output: a probability density on 18 categories
    :return: the category with the highest probability
    """
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i


def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]


def randomTrainingExample():
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = lineToTensor(line)
    return category, line, category_tensor, line_tensor


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)


n_categories = len(all_categories)
n_hidden = 128

learning_rate = 0.005  # If you set this too high, it might explode. If too low, it might not learn
criterion = nn.NLLLoss()


def setUp():
    # Build the category_lines dictionary, a list of names per language
    for filename in findFiles('data/names/*.txt'):
        category = os.path.splitext(os.path.basename(filename))[0]
        all_categories.append(category)
        lines = readLines(filename)
        category_lines[category] = lines

    return RNN(n_letters, n_hidden, n_categories)


def train(category_tensor, line_tensor, rnn):
    """
    :param category_tensor: the given category, e.g. [3]
    :param line_tensor: a name, e.g. ['Jones'] as 5 x 1 x 57 tensor
    :return: output: a 1 x 18 tensor = probability density on 18 categories
    :return: loss: the NLL difference between given category and output

    rnn is applied n times (n = length of line_tensor, 5 for 'Jones'),
    tracked by torch.
    Side effect: one optimization step applied to rnn.
    """

    rnn.zero_grad()
    hidden = rnn.initHidden()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)
    loss.backward()

    # Add parameters' gradients to their values, multiplied by learning rate
    for p in rnn.parameters():
        p.data.add_(-learning_rate, p.grad.data)

    return output, loss.item()


def loop(rnn):
    # generate n_iter training examples, randomly chosen from the data set
    # apply train to each of them.

    # n_iters = 100000
    n_iters = 100
    print_every = 5
    plot_every = 1

    # Keep track of losses for plotting
    current_loss = 0
    all_losses = []

    start = time.time()

    for iter in range(1, n_iters + 1):
        category, line, category_tensor, line_tensor = randomTrainingExample()
        output, loss = train(category_tensor, line_tensor, rnn)
        current_loss += loss

        # Print iter number, loss, name and guess
        if iter % print_every == 0:
            guess, guess_i = categoryFromOutput(output)
            correct = '✓' if guess == category else '✗ (%s)' % category
            print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100,
                                                    timeSince(start), loss, line, guess, correct))

        # Add current loss avg to list of losses
        if iter % plot_every == 0:
            all_losses.append(current_loss / plot_every)
            current_loss = 0

    plt.figure()
    plt.plot(all_losses)


def evaluate(line_tensor, rnn):
    """
    :param line_tensor: line_tensor: a name, e.g. ['Jones'] as 5 x 1 x 57 tensor
    :return: output: a 1 x 18 tensor = probability density on 18 categories
    """
    hidden = rnn.initHidden()
    output = None

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    return output


def assess(rnn):
    # Go through a bunch of examples and record which are correctly guessed

    # Keep track of correct guesses in a confusion matrix
    confusion = torch.zeros(n_categories, n_categories)
    n_confusion = 10000

    for i in range(n_confusion):
        category, line, category_tensor, line_tensor = randomTrainingExample()
        output = evaluate(line_tensor, rnn)
        guess, guess_i = categoryFromOutput(output)
        category_i = all_categories.index(category)
        confusion[category_i][guess_i] += 1

    # Normalize by dividing every row by its sum
    for i in range(n_categories):
        confusion[i] = confusion[i] / confusion[i].sum()

    # Set up plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion.numpy())
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + all_categories, rotation=90)
    ax.set_yticklabels([''] + all_categories)

    # Force label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    # sphinx_gallery_thumbnail_number = 2
    plt.show()


class TestClassification((unittest.TestCase)):
    def testTrain(self):
        rnn = setUp()
        category, line, category_tensor, line_tensor = randomTrainingExample()
        output, loss = train(category_tensor, line_tensor, rnn)

        print('\n', output, '\n', loss)

    def testRandom(self):
        print()
        setUp()
        print(list(enumerate(all_categories)))
        for _ in range(1):
            category, line, category_tensor, line_tensor = randomTrainingExample()
            print('category =', category, '/ line =', line)
            print(category_tensor)
            for i in range(line_tensor.size()[0]):
                print(line_tensor[i, 0])

    def testFull(self):
        rnn = setUp()
        loop(rnn)
        # testloop()
