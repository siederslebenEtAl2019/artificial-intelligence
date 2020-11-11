
from transformers import BertTokenizer, BertModel, BertForSequenceClassification
import torch


if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # model = BertModel.from_pretrained('bert-base-uncased')

    sequence_a = "HuggingFace is based in NYC"
    sequence_b = "Where is HuggingFace based?"

    encoded_dict = tokenizer(sequence_a, sequence_b)
    decoded = tokenizer.decode(encoded_dict["input_ids"])
    #
    # print(encoded_dict)
    # print(decoded)

    # inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
    # inputs = tokenizer("xx yy zz", return_tensors="pt")
    # print(inputs)
    # outputs = model(**inputs)
    # for t in outputs:
    #     print(t.shape)
    #
    #
    # last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

    inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
    labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
    outputs = model(**inputs, labels=labels)
    loss, logits = outputs[:2]

    print(inputs)
    print(labels)
    print(loss)
    print(logits)
