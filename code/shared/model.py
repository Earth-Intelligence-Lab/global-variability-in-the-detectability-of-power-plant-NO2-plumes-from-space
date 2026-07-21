"""Shared MLP model definition for TROPOMI plume classification.

This is the single source of truth for the MLP architecture used across
all training (Stage 6) and analysis (Stage 7) scripts.

Architecture: input -> 256 -> 128 -> 64 -> 32 -> 1
Activation: ReLU with 0.3 dropout between hidden layers.
Output: raw logit (use BCEWithLogitsLoss for training, sigmoid for inference).

Verified identical across all 30+ training/analysis scripts in the codebase.
"""

import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)
