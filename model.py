import torch
import torch.nn as nn
import numpy as np


class DuelingCNN(nn.Module):
    """
    CNN feature extractor + Dueling DQN heads.

    Learns directly from raw pixel frames (84x84 x 4 stacked).
    Separate value V(s) and advantage A(s,a) streams (Dueling architecture).

    Input : (B, C, 84, 84)  pixel values 0-255
    Output: (B, n_actions)  Q-values
    """

    def __init__(self, in_channels: int, n_actions: int):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        conv_out = self._conv_out_size(in_channels)

        self.fc = nn.Sequential(
            nn.Linear(conv_out, 512),
            nn.ReLU(),
        )

        # Dueling streams
        self.value_stream     = nn.Linear(512, 1)
        self.advantage_stream = nn.Linear(512, n_actions)

    def _conv_out_size(self, in_channels: int) -> int:
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 84, 84)
            return int(np.prod(self.conv(dummy).shape[1:]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x / 255.0                          # normalise to [0, 1]
        features = self.conv(x).flatten(1)
        features = self.fc(features)
        v = self.value_stream(features)        # (B, 1)
        a = self.advantage_stream(features)    # (B, n_actions)
        # Dueling combination: Q = V + (A - mean(A))
        return v + a - a.mean(dim=1, keepdim=True)
