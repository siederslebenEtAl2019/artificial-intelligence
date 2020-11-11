

import torch
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertModel


def vectorize(text, tokenizer, model):
    marked_text = "[CLS] " + text + " [SEP]"
    tokenized_text = tokenizer.tokenize(marked_text)  # 22 tokens
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_ids = [1] * len(tokenized_text)  # just ones
    segments_tensors = torch.tensor([segments_ids])

    with torch.no_grad():
        output = model(tokens_tensor, segments_tensors)

    hidden_states = output[2]
    token_vecs = hidden_states[-2][0]  # shape = (22, 768)
    sentence_embedding = torch.mean(token_vecs, dim=0)

    return sentence_embedding


if __name__ == '__main__':

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    text = "After stealing money from the bank vault, the bank robber was seen " \
           "fishing on the Mississippi river bank."
    marked_text = "[CLS] " + text + " [SEP]"
    tokenized_text = tokenizer.tokenize(marked_text)  # 22 tokens
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    tokens_tensor = torch.tensor([indexed_tokens])

    # Display the words with their indeces.
    for tup in zip(tokenized_text, indexed_tokens):
        print('{:<12} {:>6,}'.format(tup[0], tup[1]))

    segments_ids = [1] * len(tokenized_text) # just ones
    segments_tensors = torch.tensor([segments_ids])

    model = BertModel.from_pretrained('bert-base-uncased',
                                      output_hidden_states=True
                                      )
    model.eval()
    with torch.no_grad():
        output = model(tokens_tensor, segments_tensors)

    hidden_states = output[2]

    # For the 5th token in our sentence, select its feature values from layer 5.
    token_i = 5
    layer_i = 5
    batch_i = 0
    vec = hidden_states[layer_i][batch_i][token_i]  # shape = (13, 1, 22, 768)

    # Plot the values as a histogram to show their distribution.
    plt.figure(figsize=(10,10))
    plt.hist(vec, bins=200)
    plt.show()

    token_vecs = hidden_states[-2][0]  # shape = (22, 768)
    sentence_embedding = torch.mean(token_vecs, dim=0)
    pass

