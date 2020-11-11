# http://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/

# The function vectorize transforms a text (length < 512 words) into a
# a vector of 768 reals. This is done in several steps:
# 1) tokenize using the BERT dictionary
# 2) transform the tokens in indices
# 3) add a vector of segment ids (all ones, that's BERT
# 4) put the whole thing into a BERT model
# 5) the output's shape is (22, 1, 13, 768), corresponding to
#    (nbr of tokens, nbr of batches, nbr of layers, height of layers)
# the resu





import torch
from transformers import BertTokenizer, BertModel
from scipy.spatial.distance import cosine


def vectorize(text, tokenizer, model):
    marked_text = "[CLS] " + text + " [SEP]"
    tokenized_text = tokenizer.tokenize(marked_text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_ids = [1] * len(tokenized_text)  # just ones
    segments_tensors = torch.tensor([segments_ids])

    # print()
    # for i, token_str in enumerate(tokenized_text):
    #     print(i, token_str)

    with torch.no_grad():
        output = model(tokens_tensor, segments_tensors)

    hidden_states = output[2]
    token_vecs = hidden_states[-2][0]       # shape = (#tokens, 768)
    # result[i] = average token_vecs[i, j], j = 0 .. 767
    result = torch.mean(token_vecs, dim=0)  # shape = (768)
    return result


if __name__ == '__main__':

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased',
                                      output_hidden_states=True
                                      )
    model.eval()

    text1 = "after stealing money from the bank vault, the bank robber was seen " \
           "fishing on the Mississippi river bank."

    text2 = "The bank robber was seen fishing on the Mississippi river bank" \
            "after stealing money from the bank vault"

    text3 = "Here is the sentence I want embeddings for."
    text4 = "This is the sentence I want no embeddings for."

    vector1 = vectorize(text1, tokenizer, model)
    vector2 = vectorize(text2, tokenizer, model)
    dist = cosine(vector1, vector2)

    print(dist)