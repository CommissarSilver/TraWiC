import os
import pickle

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.model_selection import GridSearchCV, train_test_split

syntactic_threshold = 100  # threshold for considering syntactic similarity
semantic_threshold = 60  # threshold for considering semantic similarity

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

##### Grid Search #####
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_features": ["sqrt", "log2"],
    "max_depth": [10, 20, 30, None],
    "criterion": ["gini", "entropy"],
}

##### Random Forest ####
x, y = train_ds.iloc[:, :-1].values, train_ds.iloc[:, -1].values
print(f"Features shape: {x.shape}")
print(f"Target shape: {y.shape}")
print(f"Features Snippet: {x[:1]}")
print(f"Target Snippet: {y[:1]}")

clf = RandomForestClassifier()
grid_search = GridSearchCV(
    estimator=clf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2, scoring="f1"
)
grid_search.fit(x, y)
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print(f"Best Parameters: {best_params}")
print(f"Best Score: {best_score}")
clf = grid_search.best_estimator_

import seaborn as sns

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
    f"feature_importance__syn{syntactic_threshold}_sem{semantic_threshold}.png", dpi=300
)

#### Correlation Matrix ####
# Increase the figure size if you have many features.
fig, ax = plt.subplots(figsize=(12, 12))

# Create the heatmap, ensuring square cells and other configurations
sns.heatmap(train_ds.corr(), annot=True, fmt=".2f", ax=ax, square=True)

# Adjusting the Y-axis limit
ax.set_ylim(len(train_ds.columns), 0)

# Adjust the tick parameters if necessary
ax.tick_params(labelsize=12)

# Apply a tight layout with padding (if needed adjust the pad size)
plt.tight_layout(pad=2)


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
tn, fp, fn, tp = confusion_matrix(
    test_ds.iloc[:, -1].values,
    clf.predict(test_ds.iloc[:, 1:].values),
).ravel()
print(
    f"True Negatives: {tn/(tn+fp+fn+tp)}, False Positives: {fp/(tn+fp+fn+tp)}, False Negatives: {fn/(tn+fp+fn+tp)}, True Positives: {tp/(tn+fp+fn+tp)}"
)
# print the accuracy
accuracy = accuracy_score(
    test_ds.iloc[:, -1].values,
    clf.predict(test_ds.iloc[:, :-1].values),
)
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
