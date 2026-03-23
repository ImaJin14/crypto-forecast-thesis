"""CNN + LSTM hybrid: CNN extracts local features, LSTM models temporal dynamics."""
import torch
import torch.nn as nn
from .base_model import BaseForecaster


class CNNLSTMForecaster(BaseForecaster):
    def __init__(self, input_size: int, num_filters: int = 64,
                 kernel_size: int = 3, hidden_size: int = 128,
                 num_layers: int = 2, dropout: float = 0.2,
                 output_size: int = 1, learning_rate: float = 1e-3):
        super().__init__(input_size, output_size, learning_rate)
        self.conv = nn.Sequential(
            nn.Conv1d(input_size, num_filters, kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.Conv1d(num_filters, num_filters, kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
        )
        self.lstm    = nn.LSTM(num_filters, hidden_size, num_layers,
                               batch_first=True,
                               dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, features) → conv needs (batch, features, seq_len)
        x   = x.permute(0, 2, 1)
        x   = self.conv(x)
        x   = x.permute(0, 2, 1)   # back to (batch, seq_len, filters)
        out, _ = self.lstm(x)
        out    = self.dropout(out[:, -1])
        return self.fc(out)
