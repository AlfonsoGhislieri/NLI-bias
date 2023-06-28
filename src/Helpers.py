import pandas as pd
import torch
import time


def convert_probabilities_to_decimal(probabilities, decimal_places=8):
    probabilities_list = probabilities.tolist()
    probabilities_decimal = [
        [format(p, f'.{decimal_places}f') for p in prob] for prob in probabilities_list]
    return probabilities_decimal


def standardise_deberta(deberta_results):
    label_mapping = {'contradiction': deberta_results['contradiction'],
                     'neutral': deberta_results['neutral'],
                     'entailment': deberta_results['entailment']}
    return label_mapping


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
