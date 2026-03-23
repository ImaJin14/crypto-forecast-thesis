"""LSTM + Bahdanau-style attention mechanism."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseForecaster


class AttentionLSTMForecaster(BaseForecaster):
    def __init__(self, input_size: int, hidden_size: int = 256,
                 num_layers: int = 2, dropout: float = 0.2,
                 output_size: int = 1, learning_rate: float = 1e-3):
        super().__init__(input_size, output_size, learning_rate)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.attention = nn.Linear(hidden_size, 1)
        self.dropout   = nn.Dropout(dropout)
        self.fc        = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)                        # (B, T, H)
        scores = self.attention(out).squeeze(-1)      # (B, T)
        weights = F.softmax(scores, dim=1).unsqueeze(2)  # (B, T, 1)
        context = (out * weights).sum(dim=1)          # (B, H)
        context = self.dropout(context)
        return self.fc(context)
