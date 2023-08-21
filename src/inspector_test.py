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
with open("/Users/ahura/Nexus/TWMC/inspector_test_detail.csv", "w") as f:
    f.write("file_name,actual,predicted\n")
    accuracy = 0
    recall = 0
    precision = 0
    f1 = 0
    for i in tqdm(range(len(x))):
        f.write(f"{ds_s.iloc[i, 0]},{y[i]},{clf.predict(x[i].reshape(1, -1))[0]}\n")
        repo_name = ds_s.iloc[i, 0].split("/")[0]
        # if even one repo is predicted as 1, then the whole repo is predicted as 1
        final_repo_result[repo_name] = (
            final_repo_result.get(repo_name, 0) + clf.predict(x[i].reshape(1, -1))[0]
        )
        final_repo_ground_truth[repo_name] = y[i]
        accuracy += y[i] == clf.predict(x[i].reshape(1, -1))[0]
        recall += y[i] == clf.predict(x[i].reshape(1, -1))[0] and y[i] == 1
        precision += (
            y[i] == clf.predict(x[i].reshape(1, -1))[0]
            and clf.predict(x[i].reshape(1, -1))[0] == 1
        )
    print(f"accuracy: {accuracy/len(x)}")
    print(f"recall: {recall/len(x)}")
    print(f"precision: {precision/len(x)}")

with open("/Users/ahura/Nexus/TWMC/inspector_test.csv", "w") as f:
    f.write("repo_name,predicted,actual\n")
    for k, v in final_repo_result.items():
        f.write(f"{k},{1 if v>0 else 0},{final_repo_ground_truth[k]}\n")
