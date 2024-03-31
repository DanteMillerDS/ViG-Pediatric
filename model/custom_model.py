# Selected models from https://pytorch.org/vision/0.9/models.html
# Model list includes resnet50, alexnet, vgg19, squeezenet, densenet, inception, googlenet, mobilenet_v3_large, wide_resnet50_2, mnasnet
# Many more can be considered here

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    A custom model for the AI Severity project.
    param base_model: The base model to use for the custom model.
    param num_classes: The number of classes for the model.
    param dropout: The dropout rate for the model.
    return: The custom model.
    """
    def __init__(self, base_model, num_classes=1, dropout=0.2):
        super(Model, self).__init__()
        self.features = nn.Sequential(*list(base_model.children())[:-1])
        in_features = base_model.fc.in_features
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.dropout1 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(in_features=in_features, out_features=128)
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(in_features=128, out_features=num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.features(x)
        x = self.global_avg_pool(x)
        x = self.flatten(x)
        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


