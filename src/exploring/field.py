import unittest
import torch
import torch.nn.functional as F
from torchtext import data


class Field(object):
    """
            Exploring torchtext.data.Field.
            Field is function object which transforms strings into numeric vectors.
            sample is a list of 3 sentences of length 14 or 15
            chars contains all occurring characters (17 in this case)
            f is a function object which stores the padding character and the ordering (batch_first=True)
            f.build_vocab builds a dictionary that maps each char of chars to an integer
            f.to_ints pads all sentences to the sam length and transforms each character to its
            numerical equivalent. It returns a tensor of size len(sample) x maxlen = 3 x 15
            """
    def __init__(self, chartext, device):
        blank = ' '
        self.device = device
        self.field = data.Field(pad_token=blank, unk_token=blank, batch_first=True)
        self.field.build_vocab(blank.join(chartext))
        self.reverse_dict = {i: c for c, i in self.field.vocab.stoi.items()}
        self.dict_len = len(self.field.vocab.stoi)

    def to_ints(self, chartext):
        return self.field.process(chartext, device=self.device)

    def to_onehot(self, inttext):
        result = torch.zeros(inttext.size()[0], inttext.size()[1], self.dict_len, device=self.device)
        for i in range(inttext.size()[0]):
            for j in range(inttext.size()[1]):
                result[i, j, inttext[i][j]] = 1
        return result

    def encode(self, chartext):
        aux = self.to_ints(chartext)
        return self.to_onehot(aux)

    def from_onehot(self, onehottext):
        result = torch.zeros(onehottext.size()[0], onehottext.size()[1],
                             dtype=torch.int32, device=self.device)
        for i in range(onehottext.size()[0]):
            for j in range(onehottext.size()[1]):
                result[i, j] = torch.argmax(onehottext[i, j]).item()
        return result

    def from_ints(self, inttext):
        """
        :param inttext: a list of list of integers
        :return: a list of list of characters; each integer is transformed into a character
        according to self.reverse_dict
        """
        result = inttext.tolist()
        for i in range(len(result)):
            for j in range(len(result[i])):
                result[i][j] = self.reverse_dict[result[i][j]]
        return result

    @staticmethod
    def clip(input, places):
        """
        :param inttext: a list of lists of anything
        :param places: an integer
        :return: the list of lists; each list clipped left (if places > 0)
        or right (if places < 0)
        """
        slc = slice(places, None) if places >= 0 else slice(None, places)
        return [lst[slc] for lst in input]

    def decode(self, output):
        output = F.log_softmax(output[-1], dim=0)
        idx = torch.argmax(output).item()
        return self.reverse_dict[idx]


class TestField(unittest.TestCase):
    def test1(self):

        sample = ['hey how are you', 'good i am fine', 'have a nice day']
        # sample = [' ']

        device = torch.device('cpu')
        # device = torch.device('cuda:0')

        f = Field(sample, device)
        a = f.to_ints(sample)
        b = f.to_onehot(a)
        r = f.from_onehot(b)
        s = f.from_ints(r)

        u = f.clip(sample, -1)
        v = f.clip(sample, 1)
        w = f.clip(sample, 0)

        print('\n', u, '\n', v, '\n', w)





































        # chars = blank.join(sample)
        # f = data.Field(pad_token=blank, unk_token=blank, batch_first=True)
        # a = f.pad(sample)
        # f.build_vocab(chars)
        # b = f.process(sample)
        # c = torch.narrow(b, 1, 1, b.size()[1] - 1)
        # d = torch.narrow(b, 1, 0, b.size()[1] - 1)
        #
        # print('\n', f.vocab.stoi.items())
        # # print('\n', a, '\n', b, '\n', b.size(), '\n', c, '\n', d)
        # print('\n', a, '\n', b, '\n', b.size())
        #
        # r = torch.zeros(b.size()[0], b.size()[1], len(f.vocab.stoi))
        # for i in range(b.size()[0]):
        #     for j in range(b.size()[1]):
        #         r[i, j, b[i][j]] = 1
        #
        # print('\n', r, '\n', r.size())
        #
        # s = torch.zeros(r.size()[0], r.size()[1], dtype=torch.int32)
        # for i in range(r.size()[0]):
        #     for j in range(r.size()[1]):
        #         s[i, j] = torch.argmax(r[i, j]).item()
        #
        # print('\n', s, '\n', s.size())
        #
        # m = {k: v for v, k in f.vocab.stoi.items()}
        #
        # t = s.tolist()
        # for i in range(len(t)):
        #     for j in range(len(t[i])):
        #         t[i][j] = m[t[i][j]]
        #
        # print('\n', t)





