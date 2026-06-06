"""StockLSTM model definition."""

from __future__ import annotations

import torch
import torch.nn as nn


class StockLSTM(nn.Module):
    """LSTM-based architecture for multi-horizon stock return forecasting.

    The model consists of a configurable stack of PyTorch LSTM layers
    followed by a dropout layer and a final fully connected linear layer.
    """

    def __init__(
        self,
        input_size: int = 4,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_size: int = 5,
    ) -> None:
        """Initialize the model architecture.

        Parameters
        ----------
        input_size:
            Number of features per timestep (default = 4).
        hidden_size:
            Number of units in each LSTM layer's hidden state (default = 64).
        num_layers:
            Number of stacked LSTM layers (default = 2).
        dropout:
            Dropout probability applied between LSTM layers if num_layers > 1
            (default = 0.2).
        output_size:
            Number of prediction horizons (default = 5).
        """
        super().__init__()
        # If num_layers == 1, PyTorch's LSTM ignores dropout (and warns if dropout > 0).
        # We handle this explicitly to prevent warnings.
        lstm_dropout = dropout if num_layers > 1 else 0.0

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=lstm_dropout,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Parameters
        ----------
        x:
            Input sequence tensor of shape ``(batch, sequence_length, input_size)``.

        Returns
        -------
        torch.Tensor:
            Forecasted return values of shape ``(batch, output_size)``.
        """
        # lstm_out shape: (batch, sequence_length, hidden_size)
        lstm_out, _ = self.lstm(x)

        # Retrieve the hidden state from the final time step
        # final_out shape: (batch, hidden_size)
        final_out = lstm_out[:, -1, :]

        # Output projection: (batch, output_size)
        out: torch.Tensor = self.fc(final_out)
        return out
