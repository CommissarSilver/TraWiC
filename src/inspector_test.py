import os
import multiprocessing
from tqdm import tqdm

import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier

# load the model

clf = pickle.load(open("/Users/ahura/Nexus/TWMC/rf_model.sav", "rb"))
ds_s = pd.read_csv("/Users/ahura/Nexus/TWMC/all_test.csv")
ds_s.drop(
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
x, y = ds_s.iloc[:, 1:-1].values, ds_s.iloc[:, -1].values
# calculate the accuracy, recall, precision, f1-score
final_repo_result = {}
final_repo_ground_truth = {}

final_repo_total = {}

true_positives = 0
false_positives = 0
true_negatives = 0
false_negatives = 0

with open("/Users/ahura/Nexus/TWMC/inspector_test_repo_level_detail.csv", "w") as f:
    f.write("repo_name,actual,predicted\n")
    accuracy = 0
    recall = 0
    precision = 0
    f1 = 0
    for i in tqdm(range(len(x))):
        predicted_value = clf.predict(x[i].reshape(1, -1))[0]
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

        f.write(f"{ds_s.iloc[i, 0]},{actual_value,{predicted_value}}\n")
        repo_name = ds_s.iloc[i, 0].split("/")[0]
        # if even one repo is predicted as 1, then the whole repo is predicted as 1
        final_repo_result[repo_name] = (
            final_repo_result.get(repo_name, 0) + predicted_value
        )
        final_repo_total[repo_name] = final_repo_total.get(repo_name, 0) + 1

        final_repo_ground_truth[repo_name] = actual_value
        accuracy += actual_value == predicted_value
        recall += actual_value == predicted_value and actual_value == 1
        precision += actual_value == predicted_value and predicted_value == 1

    print(f"accuracy: {accuracy/len(x)}")
    print(f"recall: {recall/len(x)}")
    print(f"precision: {precision/len(x)}")

    print("True Positives: ", true_positives)
    print("False Positives: ", false_positives)
    print("True Negatives: ", true_negatives)
    print("False Negatives: ", false_negatives)

with open("/Users/ahura/Nexus/TWMC/inspector_test.csv", "w") as f:
    f.write("repo_name,predicted,actual\n")
    for k, v in final_repo_result.items():
        f.write(f"{k},{1 if v>0 else 0},{final_repo_ground_truth[k]}\n")

repo_true_positives = 0
repo_false_positives = 0
repo_true_negatives = 0
repo_false_negatives = 0
accuracy = 0
threshold = 0.4
with open("/Users/ahura/Nexus/TWMC/inspector_test_repo_level.csv", "w") as f:
    f.write("repo_name,predicted,actual\n")
    for k, v in final_repo_result.items():
        f.write(f"{k},{1 if v/final_repo_total[k]>threshold else 0},1\n")
        if v / final_repo_total[k] > threshold:
            repo_true_positives += 1
        else:
            repo_false_positives += 1
        if v / final_repo_total[k] > threshold and final_repo_ground_truth[k] == 1:
            accuracy += 1
        if v / final_repo_total[k] > threshold and final_repo_ground_truth[k] == 0:
            accuracy += 1

print("accuracy: ", repo_true_positives / len(list(final_repo_total.keys())))
print("Repo level true positives: ", repo_true_positives)
print("Repo level false positives: ", repo_false_positives)
