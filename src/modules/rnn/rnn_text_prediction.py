# from
# https://github.com/gabrielloye/RNN-walkthrough/blob/master/main.ipynb

import torch
from torch import nn

from src.exploring.field import Field

device = torch.device("cuda:0")
# device = torch.device("cpu")


class TextPredictionModule(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(TextPredictionModule, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        """
        :param x: x represents a list of sentences of equal length onehot-formated
        x.size = batch_size x sequence_length x nbr_of_characters
        :return:
        """
        batch_size = x.size(0)
        hidden = self.init_hidden(batch_size)
        out, hidden = self.rnn(x, hidden)
        out = out.reshape(-1, self.hidden_dim)
        out = self.fc(out)
        return out, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(self.n_layers, batch_size, self.hidden_dim, device=device)


class TextPredictionManager(object):

    def __init__(self, chartext):
        self.chartext = chartext
        self.maxlen = max([len(t) for t in chartext])
        self.field = Field(chartext, device)
        input_size = self.field.dict_len
        self.module = TextPredictionModule(input_size=input_size, output_size=input_size,
                                           hidden_dim=12, n_layers=1)
        self.module = self.module.to(device)
        self.criterion = nn.CrossEntropyLoss()

    def trainingLoop(self, epochs, learning_rate):
        optimizer = torch.optim.Adam(self.module.parameters(), lr=learning_rate)

        input = self.field.clip(self.chartext, -1)
        input = self.field.encode(input)

        target = self.field.clip(self.chartext, 1)
        target = self.field.to_ints(target)
        target = target.reshape(-1)

        for _ in range(epochs):
            optimizer.zero_grad()
            output, hidden = self.module(input)
            loss = self.criterion(output, target)
            loss.backward()
            optimizer.step()

    def predict(self, from_text):
        self.module.eval()
        guess = from_text
        for _ in range(self.maxlen - len(from_text)):
            input = self.field.encode([guess])
            output, _ = self.module(input)
            c = self.field.decode(output)
            guess += c
        return guess

