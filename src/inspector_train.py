import os
import pickle
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn import svm, tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GridSearchCV, train_test_split


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
arg_parse.add_argument(
    "--visualisation",
    type=bool,
    default=False,
)
args = arg_parse.parse_args()

combined_ds = pd.read_csv(
    os.path.join(
        os.getcwd(),
        "rf_data",
        f"syn{args.syntactic_threshold}_sem{args.semantic_threshold}",
        "train.csv",
    )
)

# Split the dataset into training and testing datasets
train_ds, test_ds = train_test_split(
    combined_ds,
    test_size=0.2,
    random_state=42,
    stratify=combined_ds["trained_on"],
)

# drop the index column
train_ds.drop(columns=["Unnamed: 0"], inplace=True)
test_ds.drop(columns=["Unnamed: 0"], inplace=True)

# split the training and testing datasets into x and y
x, y = train_ds.iloc[:, :-1].values, train_ds.iloc[:, -1].values
print(f"Features shape: {x.shape}")
print(f"Target shape: {y.shape}")
print(f"Features Snippet: {x[:1]}")
print(f"Target Snippet: {y[:1]}")

# classifier
if args.classifier == "rf":
    clf = RandomForestClassifier()
    # grid search parameters for random forest
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_features": ["sqrt", "log2"],
        "max_depth": [10, 20, 30],
        "criterion": ["gini", "entropy"],
    }


elif args.classifier == "svm":
    clf = svm.SVC()
    # grid search parameters for svm
    param_grid = {
        "C": [0.1, 1, 10, 100],
        "gamma": [1, 0.1, 0.01, 0.001],
        "kernel": ["rbf", "linear"],
    }

elif args.classifier == "xgb":
    clf = xgb.XGBClassifier(objective="binary:logistic")
    # grid search parameters for xgboost
    param_grid = {
        "learning_rate": [0.01, 0.1, 0.5],
        "max_depth": [3, 5, 7],
        "n_estimators": [50, 100, 200],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
    }

grid_search = GridSearchCV(
    estimator=clf,
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,
    verbose=2,
    scoring="f1",
)
grid_search.fit(x, y)
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print(f"Best Parameters: {best_params}")
print(f"Best Score: {best_score}")
clf = grid_search.best_estimator_

print(
    "Number of 1s and 0s in the train dataset:", train_ds["trained_on"].value_counts()
)
print("Number of 1s and 0s in the test dataset:", test_ds["trained_on"].value_counts())

# create a confusion matrix and print it
tn, fp, fn, tp = confusion_matrix(
    test_ds.iloc[:, -1].values,
    clf.predict(test_ds.iloc[:, 1:].values),
).ravel()

print(
    f"{Colors.GREEN}True Negatives:{Colors.END} {Colors.YELLOW}{tn/(tn+fp+fn+tp)}{Colors.END}",
    f"{Colors.BLUE}False Positives:{Colors.END} {Colors.YELLOW}{fp/(tn+fp+fn+tp)}{Colors.END}",
    f"{Colors.BLUE}False Negatives:{Colors.END} {Colors.YELLOW}{fn/(tn+fp+fn+tp)}{Colors.END}",
    f"{Colors.GREEN}True Positives:{Colors.END} {Colors.YELLOW}{tp/(tn+fp+fn+tp)}{Colors.END}",
)
# print the accuracy
accuracy = accuracy_score(
    test_ds.iloc[:, -1].values,
    clf.predict(test_ds.iloc[:, :-1].values),
)
# calcualte the precision and recall
precision, recall, fscore, _ = precision_recall_fscore_support(
    test_ds.iloc[:, -1].values,
    clf.predict(test_ds.iloc[:, :-1].values),
    average="weighted",
)

print(f"{Colors.GREEN}Precision:{Colors.END} {Colors.YELLOW}{precision}{Colors.END}")
print(f"{Colors.GREEN}Accuracy:{Colors.END} {Colors.YELLOW}{accuracy}{Colors.END}")
print(f"{Colors.GREEN}F-score:{Colors.END} {Colors.YELLOW}{fscore}{Colors.END}")
print(f"{Colors.GREEN}Sensitivity:{Colors.END} {Colors.YELLOW}{recall}{Colors.END}")
print(
    f"{Colors.GREEN}Specificity:{Colors.END} {Colors.YELLOW}{tn / (tn + fp)}{Colors.END}"
)

pickle.dump(
    clf,
    open(
        f"{args.classifier}_model__syn{args.syntactic_threshold}_sem{args.semantic_threshold}.sav",
        "wb",
    ),
)

if args.visualisation:
    sns.set_theme(style="dark")

    fig, ax = plt.subplots(figsize=(12, 12))

    # Adjust font sizes
    ax.tick_params(labelsize=27)
    ax.set_xlabel("Importance", fontsize=20, fontdict={"weight": "bold"})
    # ax.set_ylabel("Features", fontsize=10)

    # Horizontal bar chart with feature importances
    ax.barh(train_ds.columns[:-1], clf.feature_importances_)

    # Rotate y-axis labels to fit
    plt.yticks(rotation=0)
    plt.tight_layout()
    # Save the figure with a descriptive filename
    plt.savefig(
        f"feature_importance__syn{syntactic_threshold}_sem{semantic_threshold}.png",
        dpi=300,
    )

    #### Correlation Matrix ####
    fig, ax = plt.subplots(figsize=(12, 12))
    # Create the heatmap, ensuring square cells and other configurations
    sns.heatmap(train_ds.corr(), annot=True, fmt=".2f", ax=ax, square=True)
    # Adjusting the Y-axis limit
    ax.set_ylim(len(train_ds.columns), 0)
    ax.tick_params(labelsize=12)
    plt.tight_layout(pad=2)
    plt.savefig(
        f"correlation_matrix__syn{syntactic_threshold}_sem{semantic_threshold}.png",
        dpi=300,
    )
