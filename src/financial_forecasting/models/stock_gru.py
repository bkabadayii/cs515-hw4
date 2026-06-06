"""StockGRU model definition.

Mirrors the StockLSTM architecture but uses GRU cells instead of LSTM cells.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class StockGRU(nn.Module):
    """GRU-based architecture for multi-horizon stock return forecasting.

    The model consists of a configurable stack of PyTorch GRU layers
    followed by a dropout layer and a final fully connected linear layer.
    GRU cells have fewer parameters than LSTM cells (no cell state),
    which may allow faster convergence on small datasets.
    """

    def __init__(
        self,
        input_size: int = 4,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_size: int = 5,
    ) -> None:
        """Initialize the GRU model architecture.

        Parameters
        ----------
        input_size:
            Number of features per timestep (default = 4).
        hidden_size:
            Number of units in each GRU layer's hidden state (default = 64).
        num_layers:
            Number of stacked GRU layers (default = 2).
        dropout:
            Dropout probability applied between GRU layers if num_layers > 1
            (default = 0.2).
        output_size:
            Number of prediction horizons (default = 5).
        """
        super().__init__()
        # If num_layers == 1, PyTorch's GRU ignores dropout (and warns if dropout > 0).
        # We handle this explicitly to prevent warnings.
        gru_dropout = dropout if num_layers > 1 else 0.0

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=gru_dropout,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the GRU model.

        Parameters
        ----------
        x:
            Input sequence tensor of shape ``(batch, sequence_length, input_size)``.

        Returns
        -------
        torch.Tensor:
            Forecasted return values of shape ``(batch, output_size)``.
        """
        # gru_out shape: (batch, sequence_length, hidden_size)
        gru_out, _ = self.gru(x)

        # Retrieve the hidden state from the final time step
        # final_out shape: (batch, hidden_size)
        final_out = gru_out[:, -1, :]

        # Output projection: (batch, output_size)
        out: torch.Tensor = self.fc(final_out)
        return out
