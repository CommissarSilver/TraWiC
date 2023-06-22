from models import InspectorModel, InspectorModelRF
import torch
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# read the list of csv files, and combine them all into a single dataframe
datasets = [
    "/Users/ahura/Nexus/TWMC/Run 02_processed_dataset.csv",
    "/Users/ahura/Nexus/TWMC/Run 03_processed_dataset.csv",
    "/Users/ahura/Nexus/TWMC/Run 04_processed_dataset.csv",
    "/Users/ahura/Nexus/TWMC/Run 05_processed_dataset.csv",
    "/Users/ahura/Nexus/TWMC/Run 06_processed_dataset.csv",
    "/Users/ahura/Nexus/TWMC/Run 07_processed_dataset.csv",
    "/Users/ahura/Nexus/TWMC/Run 08_processed_dataset.csv",
    "/Users/ahura/Nexus/TWMC/Run 09_processed_dataset.csv",
    "/Users/ahura/Nexus/TWMC/Run 10_processed_dataset.csv",
    "/Users/ahura/Nexus/TWMC/Run 11_processed_dataset.csv",
]
datasets = [pd.read_csv(path) for path in datasets]
combined_ds = pd.concat(datasets)

print(combined_ds["trained_on"].value_counts())

combined_ds.drop(
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
combined_ds.to_csv("combined_ds.csv", index=False)

# Split the dataset into training and testing datasets
train_ds, test_ds = train_test_split(combined_ds, test_size=0.2, random_state=42)
# Split the testing dataset into testing and validation datasets
test_ds, val_ds = train_test_split(test_ds, test_size=0.5, random_state=42)
# drop the index column
train_ds.drop(columns=["Unnamed: 0"], inplace=True)
test_ds.drop(columns=["Unnamed: 0"], inplace=True)
val_ds.drop(columns=["Unnamed: 0"], inplace=True)

##### Random Forest ####
x, y = train_ds.iloc[:, 1:].values, train_ds.iloc[:, -1].values
print(f"Features shape: {x.shape}")
print(f"Target shape: {y.shape}")
print(f"Features Snippet: {x[:5]}")
print(f"Target Snippet: {y[:5]}")

clf = RandomForestClassifier(n_estimators=100, max_depth=100, random_state=0)
clf.fit(x, y)
##### Feature Importance ####
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111)
ax.barh(train_ds.columns[:-1], clf.feature_importances_)
plt.savefig("feature_importance.png", dpi=300)

#### Correlation Matrix ####
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111)
sns.heatmap(train_ds.corr(), annot=True, fmt=".2f", ax=ax)
plt.savefig("correlation_matrix.png", dpi=300)

# plot the first decision tree
plt.figure(figsize=(40, 40))
tree.plot_tree(
    clf.estimators_[0],
    feature_names=train_ds.columns[:-1],
    class_names=["0", "1"],
    filled=True,
    fontsize=10,
)
plt.savefig("decision_tree.png", dpi=300)
# print the number of 1s and 0s in the dataset
print("Number of 1s and 0s in the train dataset:", train_ds["trained_on"].value_counts())
print("Number of 1s and 0s in the test dataset:", test_ds["trained_on"].value_counts())
# create a confusion matrix and print it
from sklearn.metrics import confusion_matrix

print("Confusion matrix:")
print(
    confusion_matrix(test_ds.iloc[:, -1].values, clf.predict(test_ds.iloc[:, 1:].values))
)
# print the accuracy
accuracy = clf.score(test_ds.iloc[:, 1:].values, test_ds.iloc[:, -1].values)
print("Accuracy:", accuracy)
# calcualte the precision and recall
from sklearn.metrics import precision_recall_fscore_support

precision, recall, fscore, _ = precision_recall_fscore_support(
    test_ds.iloc[:, -1].values, clf.predict(test_ds.iloc[:, 1:].values), average="weighted"
)
print("Precision:", precision)
print("Recall:", recall)
print("F-score:", fscore)