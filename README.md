# Using NLI to detect social biases in toxic data

## Installing depencies

To install all required dependencies run:

`pip install -r requirements.txt`

---

## Files

### Data folder

Includes all the data that was used for training, testing and fine-tuning the models

### Results folder

Includes all the data that was outputed from the models for analysis and results.

These can be used to run the code without having to re-run the models each time.

---

## Models

Contains the three different models:

- Bart-large
- Deberta-v3
- Deberta base

## Helpers

Contains helper functions used by models

---

## Initial testing of models and using rationales as hypotheses

### testing-models.ipynb

Includes the initial testing of the three models:

- Bart-large
- Deberta-v3
- Deberta base

This includes rationales being used as hypotheses and experimentation with different hypotheses, and investigating neutral cases.

---

## Generate results

### generate-results.ipynb

Generates csv outputs of the results for the bart-large and deberta-v3 models using custom curated hypotheses.

Also runs fine-tuning of bart-large model using religion data that was manually annotated

---

## Anaylsis

### misc.ipynb

Contains misc anaylsis like, frequency distribution of lengths of hypotheses in NLI training data.

### analysis.ipynb

Contains all f1 scores, AUC scores, graphs and accuracy breadowns for the test and training results
