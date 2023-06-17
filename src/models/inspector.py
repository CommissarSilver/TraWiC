import torch
import torch.nn as nn


class InspectorModel(nn.Module):
    def __init__(self, num_features):
        super(InspectorModel, self).__init__()
        self.linear = nn.Linear(num_features, 1)  # Single output

    def forward(self, x):
        return self.linear(x)


class InspectorModelRF:
    # a random forest model
    def __init__(self, num_features):
        from sklearn.ensemble import RandomForestClassifier

        self.rf = RandomForestClassifier(n_estimators=100, max_depth=100, random_state=0)

    def fit(self, X, y):
        self.rf.fit(X, y)

    def predict(self, X):
        return self.rf.predict(X)

    def score(self, X, y):
        return self.rf.score(X, y)
