"""GRU forecasting model — lightweight LSTM variant."""
import torch
import torch.nn as nn
from .base_model import BaseForecaster


class GRUForecaster(BaseForecaster):
    def __init__(self, input_size: int, hidden_size: int = 256,
                 num_layers: int = 2, dropout: float = 0.2,
                 output_size: int = 1, learning_rate: float = 1e-3):
        super().__init__(input_size, output_size, learning_rate)
        self.gru = nn.GRU(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        out    = self.dropout(out[:, -1])
        return self.fc(out)
