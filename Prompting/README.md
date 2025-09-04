# Prompting

This directory contains all the required elements to replicate our Prompting appraoch.

## Directory content

- **`data`:** this directory contains train and dev .json files with graphs, texts, descriptions, and labels pairs and test .json files with graphs, descriptions, and labels.
- **`load_model.py`:** contains the code to load an LLM.
- **`build_prompts.py`:** contains the code to build the different types of prompts.
- **`generate.sh`:** contains the main script used to run a dataset through a model.
- **`test_script.sh`:** contains the script used to test the specific models from our paper.
