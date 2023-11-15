import multiprocessing
import os
import pickle
from argparse import ArgumentParser

import pandas as pd
import xgboost as xgb
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm


class Colors:
    GREEN = "\033[92m"  # GREEN
    YELLOW = "\033[93m"  # YELLOW
    BLUE = "\033[94m"  # BLUE
    END = "\033[0m"  # reset to the default color


arg_parse = ArgumentParser()
arg_parse.add_argument(
    "--classifier",
    type=str,
    choices=["rf", "svm", "xgb"],
    default="xgb",
)
arg_parse.add_argument(
    "--syntactic_threshold",
    type=int,
    default=100,
)
arg_parse.add_argument(
    "--semantic_threshold",
    type=int,
    default=80,
)

args = arg_parse.parse_args()

combined_ds = pd.read_csv(
    os.path.join(
        os.getcwd(),
        "rf_data",
        f"syn{args.syntactic_threshold}_sem{args.semantic_threshold}",
        "test.csv",
    )
)

x, y = combined_ds.iloc[:, 1:-1].values, combined_ds.iloc[:, -1].values

if args.classifier == "rf":
    clf = RandomForestClassifier()

elif args.classifier == "svm":
    clf = svm.SVC()

elif args.classifier == "xgb":
    clf = xgb.XGBClassifier(objective="binary:logistic")

# load the model
clf = pickle.load(
    open(
        f"{args.classifier}_model__syn{args.syntactic_threshold}_sem{args.semantic_threshold}.sav",
        "rb",
    )
)


# calculate the accuracy, recall, precision, f1-score
final_repo_result = {}
final_repo_ground_truth = {}

final_repo_total = {}

true_positives = 0
false_positives = 0
true_negatives = 0
false_negatives = 0

# final results file
results_file = open(
    f"clf_{args.classifier}.csv",
    "a",
)
results_file.write(
    "semantic threshold,syntactic threshold,inclusion criterion,precision,accuracy,f1-score,sensitivity,specificity\n"
)

with open(
    f"inspector_test_file_level_clf_{args.classifier}__syn{args.syntactic_threshold}_sem{args.semantic_threshold}.csv",
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

        f.write(f"{combined_ds.iloc[i, 0]},{actual_value,{predicted_value}}\n")
        repo_name = combined_ds.iloc[i, 0].split("/")[0]
        # if even one repo is predicted as 1, then the whole repo is predicted as 1
        final_repo_result[repo_name] = (
            final_repo_result.get(repo_name, 0) + predicted_value
        )
        final_repo_total[repo_name] = final_repo_total.get(repo_name, 0) + 1
        final_repo_ground_truth[repo_name] = actual_value

    precision = true_positives / (true_positives + false_positives)
    accuracy = (true_positives + true_negatives) / len(x)
    f1 = 2 * true_positives / (2 * true_positives + false_positives + false_negatives)
    sensitivity = true_positives / (true_positives + false_negatives)
    specificity = true_negatives / (true_negatives + false_positives)

results_file.write(
    f"{args.semantic_threshold},{args.syntactic_threshold},single file,{precision},{accuracy},{f1},{sensitivity},{specificity}\n"
)


repo_true_positives = 0
repo_false_positives = 0
repo_true_negatives = 0
repo_false_negatives = 0

threshold = 0.4  # if more than 40% of the files in a repo are predicted as 1, then the whole repo is predicted as 1
with open(
    f"inspector_test_repo_level_0.4_clf_{args.classifier}__syn{args.syntactic_threshold}_sem{args.semantic_threshold}.csv",
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
    precision = repo_true_positives / (repo_true_positives + repo_false_positives)
    accuracy = (repo_true_positives + repo_true_negatives) / len(
        list(final_repo_total.keys())
    )
    f1 = (
        2
        * repo_true_positives
        / (2 * repo_true_positives + repo_false_positives + repo_false_negatives)
    )
    sensitivity = repo_true_positives / (repo_true_positives + repo_false_negatives)
    specificity = repo_true_negatives / (repo_true_negatives + repo_false_positives)
results_file.write(
    f"{args.semantic_threshold},{args.syntactic_threshold},repo 0.4,{precision},{accuracy},{f1},{sensitivity},{specificity}\n"
)


repo_true_positives = 0
repo_false_positives = 0
repo_true_negatives = 0
repo_false_negatives = 0

threshold = 0.6  # if more than 40% of the files in a repo are predicted as 1, then the whole repo is predicted as 1
with open(
    f"inspector_test_repo_level_0.6_clf_{args.classifier}__syn{args.syntactic_threshold}_sem{args.semantic_threshold}.csv",
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
    precision = repo_true_positives / (repo_true_positives + repo_false_positives)
    accuracy = (repo_true_positives + repo_true_negatives) / len(
        list(final_repo_total.keys())
    )
    f1 = (
        2
        * repo_true_positives
        / (2 * repo_true_positives + repo_false_positives + repo_false_negatives)
    )
    sensitivity = repo_true_positives / (repo_true_positives + repo_false_negatives)
    specificity = repo_true_negatives / (repo_true_negatives + repo_false_positives)
results_file.write(
    f"{args.semantic_threshold},{args.syntactic_threshold},repo 0.6,{precision},{accuracy},{f1},{sensitivity},{specificity}\n"
)
