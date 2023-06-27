def convert_probabilities_to_decimal(probabilities, decimal_places=8):
    probabilities_list = probabilities.tolist()
    probabilities_decimal = [
        [format(p, f'.{decimal_places}f') for p in prob] for prob in probabilities_list]
    return probabilities_decimal
