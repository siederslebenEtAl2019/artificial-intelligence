# from https://mccormickml.com/2019/07/22/BERT-fine-tuning/#introduction

# The Corpus of Linguistic Acceptability (CoLA)

import os
import random

import numpy as np
import pandas as pd
import torch
import wget

from torch.utils.data import TensorDataset, random_split, DataLoader, RandomSampler, SequentialSampler
from torch.nn.utils import clip_grad_norm_
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup

from bert.mccormick.stats import Stats
from exploring.confusion import makeConfusion


def result(logits, labels):
    prediction = torch.argmax(logits, 1)
    histogram = torch.histc(2 * prediction + labels + 0., bins=4, min=0, max=3)
    return histogram


def getDevice():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device


def download():

    # The URL for the dataset zip file.
    url = 'https://nyu-mll.github.io/CoLA/cola_public_1.1.zip'

    # Download the file (if we haven't already)
    if not os.path.exists('./cola_public_1.1.zip'):
        print('Downloading dataset...')
        wget.download(url, './cola_public_1.1.zip')


def getDataframe(device):
    """
    @param device: cpu or cuda
    @return: a dataframe for sentences and another one for labels, the latter on device
    """
    df = pd.read_csv("./cola_public/raw/in_domain_train.tsv", delimiter='\t', header=None,
                     names=['sentence_source', 'label', 'label_notes', 'sentence'])

    return df.sentence.values, torch.tensor(df.label.values, device=device)


def encode(sentences, tokenizer, max_length):
    """
    @param sentences: a dataframe of sentences
    @param tokenizer: a tokenizer
    @param max_length: max length of a sentence
    @return: input_ids = encoded sentences, attention_masks as pytorch tensors
    """

    input_ids = []
    attention_masks = []

    for s in sentences:
        # `encode_plus` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        #   (5) Pad or truncate the sentence to `max_length`
        #   (6) Create attention masks for [PAD] tokens.
        encoded_dict = tokenizer.encode_plus(
            s,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            truncation=True,
            max_length=max_length,  # Pad & truncate all sentences.
            pad_to_max_length=True,
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors='pt',  # Return pytorch tensors.
        )

        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    return input_ids, attention_masks


def getDataloader(train_dataset, validation_dataset, batch_size, device):
    """
    @param train_dataset:
    @param validation_dataset:
    @param batch_size: 16 or 32
    @param device:
    @return: train_dataloader, validation_dataloader for device
    """

    train_dataloader = DataLoader(
        train_dataset,  # The training samples.
        sampler=RandomSampler(train_dataset),  # Select batches randomly
        batch_size=batch_size  # trains with this batch size.
    )

    validation_dataloader = DataLoader(
        validation_dataset,  # The validation samples.
        sampler=SequentialSampler(validation_dataset),  # Pull out batches sequentially.
        batch_size=batch_size  # validate with this batch size.
    )

    return train_dataloader, validation_dataloader


def getDataset(input_ids, attention_masks, labels):

    dataset = TensorDataset(input_ids, attention_masks, labels)

    # Create a 90-10 train-validation split.
    train_size = int(0.9 * len(dataset))
    eval_size = len(dataset) - train_size

    # Divide the dataset by randomly selecting samples.
    train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])
    return train_dataset, eval_dataset


def getModel(device):
    # Load BertForSequenceClassification, the pretrained BERT model with a single
    # linear classification layer on top.
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",  # Use the 12-layer BERT model, with an uncased vocab.
        num_labels=2,  # The number of output labels--2 for binary classification.
        # You can increase this for multi-class tasks.
        output_attentions=False,  # Whether the model returns attentions weights.
        output_hidden_states=False,  # Whether the model returns all hidden-states.
    )

    model.cuda(device)
    return model


def getTokenizer():
    return BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)


def getOptimizer(model):
    # Note: AdamW is a class from the huggingface library (as opposed to pytorch)
    # I believe the 'W' stands for 'Weight Decay fix"
    optimizer = AdamW(model.parameters(),
                      lr=2e-5,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
                      eps=1e-8  # args.adam_epsilon  - default is 1e-8.
                      )
    return optimizer


def getScheduler(optimizer, epochs, batch_length):

    # Total number of training steps is [number of batches] x [number of epochs].
    total_steps = epochs * batch_length
    return get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,  # Default value in run_glue.py
                                                num_training_steps=total_steps)


def seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def train(model, dataloader, optimizer, scheduler, stats):
    stats.start()
    model.train()
    cnt = 0

    for batch in dataloader:
        cnt += 1
        if cnt > 2:
            break

        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        model.zero_grad()
        loss, logits = model(b_input_ids,
                             token_type_ids=None,
                             attention_mask=b_input_mask,
                             labels=b_labels)

        stats.append(loss, result(logits, b_labels))

        loss.backward()
        clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    stats.stop()


def eval(model, dataloader, stats):
    stats.start()
    model.eval()
    cnt = 0

    for batch in dataloader:
        cnt += 1
        if cnt > 2:
            break

        with torch.no_grad():
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            loss, logits = model(b_input_ids,
                                 token_type_ids=None,
                                 attention_mask=b_input_mask,
                                 labels=b_labels)

        stats.append(loss, result(logits, b_labels))

    stats.stop()


if __name__ == '__main__':

    seed(42)
    epochs = 1
    batch_size = 16  # changed from 32 to 16
    max_length = 64
    device = getDevice()

    download()

    sentences, labels = getDataframe(device)
    tokenizer = getTokenizer()
    input_ids, attention_masks = encode(sentences, tokenizer, max_length)
    train_dataset, eval_dataset = getDataset(input_ids, attention_masks, labels)
    train_dataloader, eval_dataloader = getDataloader(train_dataset, eval_dataset, batch_size, device)

    model = getModel(device)
    optimizer = getOptimizer(model)
    scheduler = getScheduler(optimizer, epochs, len(train_dataloader))

    train_stats = Stats('training')
    eval_stats = Stats('evaluation')

    for _ in range(epochs):
        train(model, train_dataloader, optimizer, scheduler, train_stats)
        eval(model, eval_dataloader, eval_stats)

    print(train_stats.summary())
#    print(eval_stats.summary())



