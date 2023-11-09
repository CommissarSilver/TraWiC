import multiprocessing
import os
import pickle

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm

syntactic_threshold = 100  # threshold for considering syntactic similarity
semantic_threshold = 60  # threshold for considering semantic similarity

# load the model
clf = pickle.load(
    open(
        f"/store/travail/vamaj/TWMC/rf_model__syn{syntactic_threshold}_sem{semantic_threshold}.sav",
        "rb",
    )
)

ds_s = pd.read_csv(
    os.path.join(
        os.getcwd(),
        "rf_data",
        f"syn{syntactic_threshold}_sem{semantic_threshold}",
        "train.csv",
    )
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

with open(
    f"/store/travail/vamaj/TWMC/inspector_test_per_script__syn{syntactic_threshold}_sem{semantic_threshold}.csv",
    "w",
) as f:
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

print("precision - file: ", true_positives / (true_positives + false_positives))
print("accuracy - file: ", (true_positives + true_negatives) / len(x))
print(
    "f-score - file: ",
    2 * true_positives / (2 * true_positives + false_positives + false_negatives),
)

print("sensitivity - file: ", true_positives / (true_positives + false_negatives))
print("specificity - file: ", true_negatives / (true_negatives + false_positives))

print("*" * 100)

with open(
    f"/store/travail/vamaj/TWMC/inspector_test_repo_min_thresh__syn{syntactic_threshold}_sem{semantic_threshold}.csv",
    "w",
) as f:
    f.write("repo_name,predicted,actual\n")
    for k, v in final_repo_result.items():
        f.write(f"{k},{1 if v>0 else 0},{final_repo_ground_truth[k]}\n")

repo_true_positives = 0
repo_false_positives = 0
repo_true_negatives = 0
repo_false_negatives = 0
accuracy = 0
threshold = 0.4  # if more than 40% of the files in a repo are predicted as 1, then the whole repo is predicted as 1
with open(
    f"/store/travail/vamaj/TWMC/inspector_test_repo_level_thresh_{threshold}__syn{syntactic_threshold}_sem{semantic_threshold}.csv",
    "w",
) as f:
    f.write("repo_name,predicted,actual\n")
    for k, v in final_repo_result.items():
        predicted = 1 if v / final_repo_total[k] > threshold else 0
        actual = final_repo_ground_truth[k]

        f.write(f"{k},{predicted},{actual}\n")

        # Calculate true positives, false positives, true negatives, and false negatives
        if predicted == 1 and actual == 1:
            repo_true_positives += 1
        elif predicted == 1 and actual == 0:
            repo_false_positives += 1
        elif predicted == 0 and actual == 0:
            repo_true_negatives += 1
        elif predicted == 0 and actual == 1:
            repo_false_negatives += 1


print(
    "precision - 0.4: ",
    repo_true_positives / (repo_true_positives + repo_false_positives),
)
print(
    "accuracy - 0.4: ",
    (repo_true_positives + repo_true_negatives) / len(list(final_repo_total.keys())),
)
print(
    "f-score - 0.4: ",
    2
    * repo_true_positives
    / (2 * repo_true_positives + repo_false_positives + repo_false_negatives),
)

print(
    "sensitivity - repo - 0.4: ",
    repo_true_positives / (repo_true_positives + repo_false_negatives),
)
print(
    "specificity - repo - 0.4: ",
    repo_true_negatives / (repo_true_negatives + repo_false_positives),
)

print("*" * 100)

repo_true_positives = 0
repo_false_positives = 0
repo_true_negatives = 0
repo_false_negatives = 0

threshold = 0.6  # if more than 40% of the files in a repo are predicted as 1, then the whole repo is predicted as 1
with open(
    f"/store/travail/vamaj/TWMC/inspector_test_repo_level_thresh_{threshold}__syn{syntactic_threshold}_sem{semantic_threshold}.csv",
    "w",
) as f:
    f.write("repo_name,predicted,actual\n")
    for k, v in final_repo_result.items():
        predicted = 1 if v / final_repo_total[k] > threshold else 0
        actual = final_repo_ground_truth[k]

        f.write(f"{k},{predicted},{actual}\n")

        # Calculate true positives, false positives, true negatives, and false negatives
        if predicted == 1 and actual == 1:
            repo_true_positives += 1
        elif predicted == 1 and actual == 0:
            repo_false_positives += 1
        elif predicted == 0 and actual == 0:
            repo_true_negatives += 1
        elif predicted == 0 and actual == 1:
            repo_false_negatives += 1
print(
    "precision - 0.6: ",
    repo_true_positives / (repo_true_positives + repo_false_positives),
)
print(
    "accuracy - 0.6: ",
    (repo_true_positives + repo_true_negatives) / len(list(final_repo_total.keys())),
)
print(
    "f-score - 0.6: ",
    2
    * repo_true_positives
    / (2 * repo_true_positives + repo_false_positives + repo_false_negatives),
)
print(
    "sensitivity - repo - 0.6: ",
    repo_true_positives / (repo_true_positives + repo_false_negatives),
)
print(
    "specificity - repo - 0.6: ",
    repo_true_negatives / (repo_true_negatives + repo_false_positives),
)
