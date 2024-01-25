This repository contains the codes and artifacts for our paper, TraWiC: Trained Without My Consent.

`TraWiC` is a tool designed for dataset inclusion detection in the training dataset of large language models trained on code using membership inference attacks.

# How to Run

## 1 - Dependencies
The experiments were carried out using Python 3.10.
Install the dependencies with:
```bash
pip install -r requirements.txt
```

huggingface-cli is required for downloading the dataset. Please install it with:
```bash
pip install huggingface-hub[cli]
```
In order to have access to [TheStack](https://huggingface.co/datasets/bigcode/the-stack), you need to login with your HuggingFace account as outlined in the [documentation](https://huggingface.co/docs/huggingface_hub/main/guides/cli).

## 2 - Download The Dataset
In order to download the dataset used for this study, run the following command:
```bash
python src/dataset.py
```
This will download the dataset and save it in the `data` directory. The dataset is extremely large. Therefore, ensure that you have enough space in your disk.

## 3 - Run The Tests
After getting the dataset, check that everything works and the directories are as they are supposed to be by running the following command:
```bash
python src/run_tests.py
```

## 4 - Run The Experiments
There are two modes for running the experiments. `single script` mode and `block` mode. The former runs the experiments for detecting token similarity and is used for TraWiC itself. The latter runs the experiments for generating the data for clone detection using NiCad.

- For generating the data for TraWiC, run the following command:
    ```bash
    python src/main.py
    ```
    The outputs will be saved in the `run_results\TokensRun{run_num}` directory in the `results.jsonl` file.

- For generating the data for NiCad, run the following command:
    ```bash
    python src/main_block.py
    ```
    The outputs will be saved in the `run_results\BlocksRun{run_num}` directory in the `results_block_{run_num}.jsonl` file.


This should provide you with the data necessary to train the classifier and use NiCad for clone detection.

# How to Reproduce the Results

## 1 - Create The Dataset for Classification
In order to create the dataset for classification, run the following command:
```bash
python src/data/dataset_builder.py
```
This will create the dataset for classification and save it in the `rf_data` directory.

## 2 - Create The Dataset for NiCad
In order to create the dataset for NiCad, run the following command:
```bash
python src/utils/block_code_builder.py
```
This will create the dataset for NiCad and save it in the `blocks` directory.

## 3 - Train The Classifier
In order to train the classifier, run the following command:
```bash
python src/inspector_train.py --classifier {classifier_name} --syntactic_threshold {syntactic_threshold} --semantic_threshold {semantic_threshold} 
```
Where `{classifier_name}` is the name of the classifier you want to train. The available classifiers are:
- `rf` for Random Forest.
- `svm` for Support Vector Machine.
- `xgb` for XGBoost.

The `{syntactic_threshold}` and `{semantic_threshold}` are the thresholds for the syntactic and semantic similarity respectively. The default values are `100` and `80` respectively. 

## 4 - Run NiCad
Please ensure that you have NiCad installed by following the instructions in the [NiCad Clone Detector](https://www.txl.ca/txl-nicaddownload.html). After installing NiCad, run the following command:
```bash
python utils/nicad_checker.py
```
***PLEASE NOTE: You need to manually set the path for NiCad's exe file in the script.***

This will take a long time to run. The results will be saved in the `nicad_results` directory.

Afterwards, run the following command:
```bash
python src/nicad_test.py
```

# How to Cite
If you use this code in your research, please cite the following paper:
