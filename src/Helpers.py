import pandas as pd
import torch
import time


def standardise_results(results):
    label_mapping = {'contradiction': results['contradiction'],
                     'neutral': results['neutral'],
                     'entailment': results['entailment']}
    return label_mapping


def convert_probabilities(probabilities, label_mapping):
    return {name: round(float(pred) * 100, 1)
            for pred, name in zip(probabilities[0], label_mapping)}


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
