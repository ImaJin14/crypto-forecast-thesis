"""Bidirectional LSTM forecasting model."""
import torch
import torch.nn as nn
from .base_model import BaseForecaster


class BiLSTMForecaster(BaseForecaster):
    def __init__(self, input_size: int, hidden_size: int = 128,
                 num_layers: int = 2, dropout: float = 0.2,
                 output_size: int = 1, learning_rate: float = 1e-3):
        super().__init__(input_size, output_size, learning_rate)
        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=num_layers, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size * 2, output_size)  # *2 for bidirectional

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        out    = self.dropout(out[:, -1])
        return self.fc(out)
