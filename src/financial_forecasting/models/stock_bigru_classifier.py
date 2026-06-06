"""Bidirectional GRU classifier for turning-point detection (Phase 5).

Mirrors StockBiLSTMClassifier but uses GRU cells.  GRU has fewer
parameters than LSTM (no separate cell state) which may train faster
on small datasets while achieving similar classification performance.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class StockBiGRUClassifier(nn.Module):
    """Bidirectional GRU for binary stock turning-point classification.

    Architecture:
        BiGRU layers  ->  dropout  ->  FC(hidden_size*2, output_size)

    The bidirectional GRU concatenates forward and backward hidden states,
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
        """Initialise the bidirectional GRU classifier.

        Parameters
        ----------
        input_size:
            Number of features per timestep.
        hidden_size:
            Per-direction hidden-state dimension.
        num_layers:
            Number of stacked BiGRU layers.
        dropout:
            Dropout probability (ignored when num_layers == 1).
        output_size:
            Logit output dimension (default = 1 for binary classification).
        """
        super().__init__()

        gru_dropout = dropout if num_layers > 1 else 0.0

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=gru_dropout,
            bidirectional=True,
            batch_first=True,
        )
        self.dropout = nn.Dropout(p=dropout)
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
            Raw logits of shape (batch, 1).
        """
        # gru_out: (batch, seq_len, hidden_size * 2)
        gru_out, _ = self.gru(x)
        final_out = gru_out[:, -1, :]           # (batch, hidden_size * 2)
        final_out = self.dropout(final_out)
        out: torch.Tensor = self.fc(final_out)  # (batch, 1)
        return out
