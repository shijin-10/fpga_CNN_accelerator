import torch
import torch.nn as nn


class CNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(
            1,
            8,
            kernel_size=3,
            padding=1
        )

        self.conv2 = nn.Conv2d(
            8,
            16,
            kernel_size=3,
            padding=1
        )

        self.pool = nn.MaxPool2d(
            2,
            2
        )

        self.fc = nn.Linear(
            16 * 7 * 7,
            10
        )

    def forward(self, x):

        x = self.conv1(x)
        x = torch.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = torch.relu(x)
        x = self.pool(x)

        x = x.view(
            x.size(0),
            -1
        )

        x = self.fc(x)

        return x
