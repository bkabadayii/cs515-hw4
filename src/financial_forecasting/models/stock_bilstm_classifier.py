"""Bidirectional LSTM classifier for turning-point detection (Phase 5).

The model takes a sequence of shape (batch, T, input_size) and outputs
a single raw logit of shape (batch, 1).  Apply sigmoid to convert to
probability; use BCEWithLogitsLoss during training for numerical stability.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class StockBiLSTMClassifier(nn.Module):
    """Bidirectional LSTM for binary stock turning-point classification.

    Architecture:
        BiLSTM layers  ->  dropout  ->  FC(hidden_size*2, output_size)

    The bidirectional LSTM concatenates forward and backward hidden states,
    so the effective feature size fed to the FC head is hidden_size * 2.
    """

    def __init__(
        self,
        input_size: int = 4,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_size: int = 1,
    ) -> None:
        """Initialise the bidirectional LSTM classifier.

        Parameters
        ----------
        input_size:
            Number of features per timestep.
        hidden_size:
            Per-direction hidden-state dimension.  The FC layer receives
            hidden_size * 2 because of the bidirectional concatenation.
        num_layers:
            Number of stacked BiLSTM layers.
        dropout:
            Dropout probability applied between stacked layers.
            Ignored (and not passed to LSTM) when num_layers == 1.
        output_size:
            Logit output dimension (default = 1 for binary classification).
        """
        super().__init__()

        lstm_dropout = dropout if num_layers > 1 else 0.0

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=lstm_dropout,
            bidirectional=True,
            batch_first=True,
        )
        # Apply dropout before the classification head
        self.dropout = nn.Dropout(p=dropout)
        # Bidirectional doubles the hidden dimension
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x:
            Input tensor of shape (batch, sequence_length, input_size).

        Returns
        -------
        torch.Tensor:
            Raw logits of shape (batch, 1).  Apply sigmoid to get
            probabilities; use BCEWithLogitsLoss during training.
        """
        # lstm_out: (batch, seq_len, hidden_size * 2)
        lstm_out, _ = self.lstm(x)
        # Use the representation at the final timestep
        final_out = lstm_out[:, -1, :]          # (batch, hidden_size * 2)
        final_out = self.dropout(final_out)
        out: torch.Tensor = self.fc(final_out)  # (batch, 1)
        return out
