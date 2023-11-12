import pickle
import os
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import (
    precision_score,
    accuracy_score,
    f1_score,
    recall_score,
    confusion_matrix,
)
from typing import List

# specify the semantic, syntactic, and similarity thresholds
semantic_thresholds = [20, 40, 60, 70, 80]
syntactic_threshold = 100
sensitivity_thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]

models_dir = os.path.join(os.getcwd(), "rf_models")
test_data_dir = os.path.join(os.getcwd(), "rf_data")


for semantic_threshold in semantic_thresholds:
    model_to_use = pickle.load(
        open(
            os.path.join(
                models_dir,
                f"rf_model__syn{syntactic_threshold}_sem{semantic_threshold}.sav",
            ),
            "rb",
        )
    )

    final_repo_result = {}
    final_repo_ground_truth = {}
    final_repo_total = {}

    for sensitivity_threshold in sensitivity_thresholds:
        dataset_to_use = pd.read_csv(
            os.path.join(
                test_data_dir,
                f"syn{syntactic_threshold}_sem{semantic_threshold}_sen{sensitivity_threshold}",
                "test.csv",
            )
        )
        dataset_to_use.drop(
            columns=[
                "class_nums_total",
                "function_nums_total",
                "variable_nums_total",
                "string_nums_total",
                "comment_nums_total",
                "docstring_nums_total",
            ],
            inplace=True,
        )

        x, y = dataset_to_use.iloc[:, 1:-1].values, dataset_to_use.iloc[:, -1].values

        true_positives = 0
        false_positives = 0
        true_negatives = 0
        false_negatives = 0

        for i in tqdm(range(len(x))):
            predicted_value = model_to_use.predict(x[i].reshape(1, -1))[0]
            actual_value = y[i]
            if predicted_value == actual_value:
                if actual_value == 1:
                    true_positives += 1
                else:
                    true_negatives += 1
            else:
                if predicted_value == 1:
                    false_positives += 1
                else:
                    false_negatives += 1

            repo_name = dataset_to_use.iloc[i, 0].split("/")[0]
            # if even one repo is predicted as 1, then the whole repo is predicted as 1
            final_repo_result[repo_name] = (
                final_repo_result.get(repo_name, 0) + predicted_value
            )
            final_repo_total[repo_name] = final_repo_total.get(repo_name, 0) + 1
            final_repo_ground_truth[repo_name] = actual_value

        accuracy = (true_positives + true_negatives) / (
            true_positives + true_negatives + false_positives + false_negatives
        )
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
        f1 = 2 * ((precision * recall) / (precision + recall))

        print(
            f"Model Sem_{semantic_threshold}_Syn_{syntactic_threshold}_sensitivity{sensitivity_threshold}:\nPrecision: {precision}\nAccuracy: {accuracy}\nF1: {f1}\nSensitivity: {recall}\nSpecificity: {true_negatives / (true_negatives + false_positives)}\n"
        )
