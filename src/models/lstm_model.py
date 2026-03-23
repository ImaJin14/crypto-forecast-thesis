"""Vanilla LSTM forecasting model."""
import torch
import torch.nn as nn
from .base_model import BaseForecaster


class LSTMForecaster(BaseForecaster):
    """
    Stacked LSTM for price sequence forecasting.

    Args:
        input_size:   Number of input features per timestep
        hidden_size:  LSTM hidden state dimension
        num_layers:   Number of stacked LSTM layers
        dropout:      Dropout rate between layers
        output_size:  Forecast horizon (default 1)
    """

    def __init__(self, input_size: int, hidden_size: int = 256,
                 num_layers: int = 2, dropout: float = 0.2,
                 output_size: int = 1, learning_rate: float = 1e-3):
        super().__init__(input_size, output_size, learning_rate)
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        out, _ = self.lstm(x)         # (batch, seq_len, hidden)
        out    = self.dropout(out)
        out    = self.fc(out[:, -1])  # last timestep → (batch, output)
        return out
