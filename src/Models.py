from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.Helpers import *
import torch


def deberta_nli(premise, hypothesis):

    model = AutoModelForSequenceClassification.from_pretrained(
        'cross-encoder/nli-deberta-base')
    tokenizer = AutoTokenizer.from_pretrained('cross-encoder/nli-deberta-base')

    features = tokenizer([premise], [hypothesis],
                         padding=True, truncation=True, return_tensors="pt")

    model.eval()
    with torch.no_grad():
        scores = model(**features).logits
        # Apply softmax along dimension 1
        probabilities = torch.softmax(scores, dim=1)
        label_mapping = ['contradiction', 'entailment', 'neutral']
        labels = [label_mapping[score_max]
                  for score_max in probabilities.argmax(dim=1)]

    probabilities_decimal = convert_probabilities_to_decimal(probabilities)

    return standardise_deberta(probabilities_decimal), labels


def bart_nli(premise, hypothesis):
    nli_model = AutoModelForSequenceClassification.from_pretrained(
        'facebook/bart-large-mnli')
    tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large-mnli')

    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")

    # run through model pre-trained on MNLI
    x = tokenizer.encode(premise, hypothesis, return_tensors='pt',
                         truncation=True,)
    logits = nli_model(x.to(device))[0]

    probabilities = logits.softmax(dim=1)
    label_mapping = ['contradiction', 'neutral', 'entailment']
    labels = [label_mapping[score_max]
              for score_max in probabilities.argmax(dim=1)]

    probabilities_decimal = convert_probabilities_to_decimal(probabilities)

    return probabilities_decimal, labels
