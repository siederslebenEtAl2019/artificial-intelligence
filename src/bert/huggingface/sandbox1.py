

import torch
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification

if __name__ == '__main__':

    classifier1 = pipeline('sentiment-analysis')
    result = classifier1(['We are very happy to show you the 🤗 Transformers library.',
                        "We hope you don't hate it."])
    # classifier = pipeline('sentiment-analysis', model="nlptown/bert-base-multilingual-uncased-sentiment")

    print(result)

    model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer2 = AutoTokenizer.from_pretrained(model_name)
    classifier2 = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer2)
