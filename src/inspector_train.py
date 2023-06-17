from models import InspectorModel, InspectorModelRF
import torch
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

ds = pd.read_csv("/Users/ahura/Nexus/TWMC/lm_dataset.csv")
ds.drop(
    columns=[
        "file_name",
        "class_nums_total",
        "function_nums_total",
        "variable_nums_total",
        "string_nums_total",
        "comment_nums_total",
        "docstring_nums_total",
    ],
    inplace=True,
)
#### Linear Model ####
#! Unstable. may have to opt out to using ensemble models
# convert the dataframe to a tensor, the last column is the label
x = torch.tensor(ds.iloc[:, 1:].values, dtype=torch.float32)
y = torch.tensor(ds.iloc[:, -1].values, dtype=torch.float32)
# break these down into batches of 32

lr = 0.001
epochs = 100
early_stopping_patience = 15
best_loss = float("inf")
patience_counter = 0

model = InspectorModel(6)
# categorical cross entropy loss
loss = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

for epoch in range(epochs):
    # forward pass
    y_pred = model(x)
    # compute loss
    l = loss(y_pred, y.unsqueeze(1))
    # backward pass
    l.backward()
    # update parameters
    optimizer.step()
    # zero the gradients
    optimizer.zero_grad()

    if l.item() < best_loss:
        best_loss = l.item()
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # Print accuracy every ten epochs in green and loss in red
    if (epoch + 1) % 10 == 0:
        print(
            f"Epoch {epoch+1}:",
            "\033[32mAccuracy:",
            (y_pred.round() == y.unsqueeze(1)).sum().item() / len(y),
            "\033[0m",
            "\033[34mLoss:",
            l.item(),
            "\033[0m",
        )
#### Linear Model ####

#### Random Forest ####
x, y = ds.iloc[:, 1:].values, ds.iloc[:, -1].values
clf = RandomForestClassifier(n_estimators=100, max_depth=100, random_state=0)
clf.fit(x, y)
from sklearn.tree import export_graphviz

export_graphviz(clf.estimators_[0], out_file="tree.dot", feature_names=ds.columns[:-1])
#### Random Forest ####
