from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.Helpers import *
import torch
import time
import pandas as pd


DEVICE = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")


def standardise_results(results):
    label_mapping = {'contradiction': results['contradiction'],
                     'neutral': results['neutral'],
                     'entailment': results['entailment']}
    return label_mapping


def convert_probabilities(probabilities, label_mapping):
    # Convert the tensor to a list and extract the first (and only) batch
    probabilities_list = probabilities.tolist()[0]
    return {name: round(float(pred) * 100, 1) for pred, name in zip(probabilities_list, label_mapping)}


def convert_probabilities_batched(probabilities, label_mapping):
    probabilities_list = probabilities.tolist()  # Convert the tensor to a list
    return [{name: round(pred * 100, 1) for pred, name in zip(preds, label_mapping)} for preds in probabilities_list]


def get_random_samples(csv_filename, num_samples):
    df = pd.read_csv(csv_filename)
    random_samples = df.sample(n=num_samples)
    return random_samples


def benchmark_test(premise, hypothesis, model, num_runs):
    # Setup device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    results = []
    for _ in range(num_runs):
        start_time = time.time()
        _, _ = model(premise, hypothesis)
        execution_time = time.time() - start_time
        results.append(execution_time)

    avg_execution_time = sum(results) / len(results)
    return device, avg_execution_time
