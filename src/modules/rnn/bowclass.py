# from
# https://pytorch.org/tutorials/beginner/nlp/deep_learning_tutorial.html

import torch
import unittest

import torch.nn as nn
import torch.optim as optim

device = torch.device("cpu")
# device = torch.device("cuda:0")

train_data = [("me gusta comer en la cafeteria".split(), "SPANISH"),
              ("Give it to me".split(), "ENGLISH"),
              ("No creo que sea una buena idea".split(), "SPANISH"),
              ("No it is not a good idea to get lost at sea".split(), "ENGLISH")]

test_data = [("Yo creo que si".split(), "SPANISH"),
             ("it is lost on me".split(), "ENGLISH")]

category_to_ix = {"SPANISH": 0, "ENGLISH": 1}


# train_data = [("aa bb cc ddd eee kjh dfge kli".split(), "LETTERS"),
#               ("11 234 765 897 987".split(), "NUMBERS"),
#               ("aab jhg piu mnbv ppl klk cc ddd eee uzhg werd kli".split(), "LETTERS"),
#               ("876 564 345 876 9987 34 21 098 9876 3452".split(), "NUMBERS")]
#
# test_data = [("edr bb piu cc".split(), "LETTERS"),
#              ("234 987 34 564 11".split(), "NUMBERS")]

# train_data = [("aa bb".split(), "LETTERS"),
#               ("aab jhg".split(), "LETTERS"),
#               ("11 234".split(), "NUMBERS"),
#               ("876 564".split(), "NUMBERS")]
#
# test_data = [("edr bb aa".split(), "LETTERS"),
#              ("564 11".split(), "NUMBERS")]
#
# label_to_ix = {"LETTERS": 0, "NUMBERS": 1}


def make_word_to_ix():
    # word_to_ix maps each word in the vocab to a unique integer, which will be its
    # index into the Bag of words vector
    word_to_ix = {}
    for sent, _ in train_data + test_data:
        for word in sent:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
    return word_to_ix


def make_target(label, label_to_ix):
    return torch.LongTensor([label_to_ix[label]])


class BoWClassifier(nn.Module):

    def __init__(self, n_categories, vocab_size):
        """
        :param n_categories: number of categories (two)
        :param vocab_size: length of input vector (one weight per word)
        """
        # calls the init function of nn.Module.  Dont get confused by syntax,
        # just always do it in an nn.Module
        super(BoWClassifier, self).__init__()

        # Define the parameters that you will need.  In this case, we need A and b,
        # the parameters of the affine mapping.
        # Torch defines nn.Linear(), which provides the affine map.
        # Make sure you understand why the input dimension is vocab_size
        # and the output is num_labels!
        self.linear = nn.Linear(vocab_size, n_categories)

        # NOTE! The non-linearity log softmax does not have parameters! So we don't need
        # to worry about that here

    def forward(self, bow_vec):
        """
        :param bow_vec: a vector of size 1 x vocab_size
        :return: a vector 0f size 1 x n_catgories
        """
        # Pass the input through the linear layer
        return self.linear(bow_vec)


class BowClassification(object):
    def __init__(self, num_labels):
        self.word_to_ix = make_word_to_ix()
        self.vocab_size = len(self.word_to_ix)
        self.model = BoWClassifier(num_labels, self.vocab_size)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.1)

    def make_bow_vector(self, sentence):
        """
        :param sentence: iterable of words
        :return: histogram indicating the number of occurrences for each word,
        shape = (1, len(self.word_to_ix))
        """
        vec = torch.zeros(len(self.word_to_ix))
        for word in sentence:
            vec[self.word_to_ix[word]] += 1
        return vec.view(1, -1)

    def testloop(self):
        # Run on test data before we train, just to see a before-and-after
        with torch.no_grad():
            for instance, label in test_data:
                bow_vec = self.make_bow_vector(instance)
                log_probs = self.model(bow_vec)
                print(instance, label, log_probs)

        # Print the matrix column corresponding to "creo"
        print(next(self.model.parameters())[:, 0])

    def trainloop(self, n_iter):
        # Usually you want to pass over the training data several times.
        # 100 is much bigger than on a real data set, but real datasets have more than
        # two instances.  Usually, somewhere between 5 and 30 epochs is reasonable.
        for epoch in range(n_iter):
            for instance, category in train_data:
                # Step 1. Remember that PyTorch accumulates gradients.
                # We need to clear them out before each instance
                self.model.zero_grad()

                # Step 2. Make our BOW vector and also we must wrap the target in a
                # Tensor as an integer. For example, if the target is SPANISH, then
                # we wrap the integer 0. The loss function then knows that the 0th
                # element of the log probabilities is the log probability
                # corresponding to SPANISH
                bow_vec = self.make_bow_vector(instance)
                target = make_target(category, category_to_ix)

                # Step 3. Run our forward pass.
                output = self.model(bow_vec)  # shape = [1, 2]

                # Step 4. Compute the loss, gradients, and update the parameters by
                # calling optimizer.step()
                loss = self.loss_function(output, target)
                loss.backward()
                self.optimizer.step()

        with torch.no_grad():
            for instance, category in test_data:
                bow_vec = self.make_bow_vector(instance)
                output = self.model(bow_vec)
                print(output)

        # Index corresponding to Spanish goes up, English goes down!
        print(next(self.model.parameters())[:, 0])


class TestClassification((unittest.TestCase)):
    def testPlay(self):
        print()
        bc = BowClassification(2)
        bc.testloop()

    def testTrainloop(self):
        print('\n', 'before training')
        bc = BowClassification(2)
        bc.testloop()
        print(list(bc.model.parameters()))
        bc.trainloop(10)
        print('\n', 'after training')
        bc.testloop()
        print(list(bc.model.parameters()))
