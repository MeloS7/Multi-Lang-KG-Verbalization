# FineTuneMT

This directory contains all the required elements to replicate our FTMT appraoch.

## Directory content

- **`data`:** this directory contains train and dev .json files with graph-text pairs and test .json files with graphs.
- **`train.py`:** contains the code to train a given MT model on a given dataset.
- **`test.py`:** contains the code to test a given MT model on a given dataset.
- **`train_script.sh`:** contains the script used to train the specific models from our paper.
- **`test_script.sh`:** contains the script used to test the specific models from our paper.

## MT Model Selection

As described on our paper (Table 2, Section 3.2) we selected different MT model for each language baased on best translation eprformance:

| Language | Model |
|-|-|
| Breton | Helsinki |
| Chinese | M2M-100 |
| English | NLLB-200 |
| French | Helsinki |
| Irish | NLLB-200 |
| Maltese | Helsinki |
| Russian | M2M-100 |
| Spanish | Helsinki |
| Welsh | M2M-100 |
