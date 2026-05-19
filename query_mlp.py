"""
QueryMLP v3: learns the mapping from IMU + chart inputs -> (cx, bottom_cy)
in normalized image coordinates [0, 1].

Inputs (6D):
    [dist_norm, inv_dist, bearing_norm, pitch_norm, roll_norm, heading_norm]

Improvements over v2:
- BatchNorm1d after each Linear, before ReLU
- Dropout(0.2) after each activation
- 3 hidden layers instead of 2
- Width stays at 128 (dataset too small for 256)
"""

import torch
import torch.nn as nn


class QueryMLP(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=128, output_dim=2, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()   # output in [0, 1] -- normalized image coordinates
        )

    def forward(self, x):
        """
        x: [..., input_dim] tensor
        returns: [..., 2] tensor of [cx, bottom_cy] in [0, 1]
        """
        return self.net(x)