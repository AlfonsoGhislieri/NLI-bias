import numpy as np
import pandas as pd


def convert_probabilities_to_decimal(probabilities, decimal_places=8):
    probabilities_list = probabilities.tolist()
    probabilities_decimal = [
        [format(p, f'.{decimal_places}f') for p in prob] for prob in probabilities_list]
    return probabilities_decimal


def standardise_deberta(deberta_results):
    return [[result[0], result[2], result[1]] for result in deberta_results]


def get_random_samples(csv_filename, num_samples):
    df = pd.read_csv(csv_filename)
    random_samples = df.sample(n=num_samples)
    return random_samples
