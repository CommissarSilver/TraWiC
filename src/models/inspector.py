import torch
import torch.nn as nn


class InspectorModel(nn.Module):
    def __init__(self, num_features):
        super(InspectorModel, self).__init__()
        self.linear = nn.Linear(num_features, 1)  # Single output

    def forward(self, x):
        return self.linear(x)
