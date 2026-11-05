"""Shared CNN backbone used by every method, so comparisons differ only in objective."""
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    """Three convolutional blocks, global average pooling, dropout and three logits."""

    def __init__(self, in_channels):
        """Build layers for greyscale (1) or colour (3) input channels."""
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(.3)
        self.fc = nn.Linear(128, 3)

    def forward(self, x):
        """Return class logits for a batch of NCHW images."""
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.fc(self.dropout(x.mean(dim=(2, 3))))
