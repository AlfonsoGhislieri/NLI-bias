from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.Helpers import *
import torch

DEVICE = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")


def deberta_base_nli(premise, hypothesis):

    model_name = 'cross-encoder/nli-deberta-base'
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

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

    probabilities = convert_probabilities(probabilities, label_mapping)

    return standardise_deberta(probabilities), labels


def deberta_v3_nli(premise, hypothesis):

    model_name = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    inputs = tokenizer(premise, hypothesis,
                       truncation=True, return_tensors="pt")
    inputs.to(DEVICE)

    outputs = model(**inputs)

    probabilities = torch.softmax(outputs.logits, dim=1)
    label_mapping = ["entailment", "neutral", "contradiction"]
    labels = [label_mapping[score_max]
              for score_max in probabilities.argmax(dim=1)]

    probabilities = convert_probabilities(probabilities, label_mapping)

    return standardise_deberta_v3(probabilities), labels


def bart_nli(premise, hypothesis):

    model_name = 'facebook/bart-large-mnli'
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # run through model pre-trained on MNLI
    x = tokenizer.encode(premise, hypothesis, return_tensors='pt',
                         truncation=True,)
    logits = model(x.to(DEVICE))[0]

    probabilities = logits.softmax(dim=1)
    label_mapping = ['contradiction', 'neutral', 'entailment']
    labels = [label_mapping[score_max]
              for score_max in probabilities.argmax(dim=1)]

    probabilities = convert_probabilities(probabilities, label_mapping)

    return probabilities, labels
