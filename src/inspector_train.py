import os
import pickle

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

syntactic_threshold = 100  # threshold for considering syntactic similarity
semantic_threshold = 40  # threshold for considering semantic similarity

combined_ds = pd.read_csv(
    os.path.join(
        os.getcwd(),
        "rf_data",
        f"syn{syntactic_threshold}_sem{semantic_threshold}",
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


##### Random Forest ####
x, y = train_ds.iloc[:, :-1].values, train_ds.iloc[:, -1].values
print(f"Features shape: {x.shape}")
print(f"Target shape: {y.shape}")
print(f"Features Snippet: {x[:1]}")
print(f"Target Snippet: {y[:1]}")

clf = RandomForestClassifier(n_estimators=50, max_depth=100, random_state=0)
clf.fit(x, y)
##### Feature Importance ####
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111)
ax.barh(train_ds.columns[:-1], clf.feature_importances_)
plt.savefig(
    f"feature_importance__syn{syntactic_threshold}_sem{semantic_threshold}.png", dpi=300
)

#### Correlation Matrix ####
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111)
sns.heatmap(train_ds.corr(), annot=True, fmt=".2f", ax=ax)
plt.savefig(
    f"correlation_matrix__syn{syntactic_threshold}_sem{semantic_threshold}.png", dpi=300
)

# plot the first decision tree
plt.figure(figsize=(40, 40))
tree.plot_tree(
    clf.estimators_[0],
    feature_names=train_ds.columns[:-1],
    class_names=["0", "1"],
    filled=True,
    fontsize=10,
)
plt.savefig(
    f"decision_tree__syn{syntactic_threshold}_sem{semantic_threshold}.png", dpi=300
)

# print the number of 1s and 0s in the dataset
print("Number of 1s and 0s in the train dataset:", train_ds["trained_on"].value_counts())
print("Number of 1s and 0s in the test dataset:", test_ds["trained_on"].value_counts())

# create a confusion matrix and print it
print("Confusion matrix:")
print(
    confusion_matrix(
        test_ds.iloc[:, -1].values,
        clf.predict(test_ds.iloc[:, 1:].values),
    )
)
# print the accuracy
accuracy = clf.score(test_ds.iloc[:, :-1].values, test_ds.iloc[:, -1].values)
print("Accuracy:", accuracy)

# calcualte the precision and recall
precision, recall, fscore, _ = precision_recall_fscore_support(
    test_ds.iloc[:, -1].values,
    clf.predict(test_ds.iloc[:, :-1].values),
    average="weighted",
)
print("Precision:", precision)
print("Recall:", recall)
print("F-score:", fscore)

# save the model
pickle.dump(
    clf, open(f"rf_model__syn{syntactic_threshold}_sem{semantic_threshold}.sav", "wb")
)
