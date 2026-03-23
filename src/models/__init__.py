from .lstm_model          import LSTMForecaster
from .gru_model           import GRUForecaster
from .bilstm_model        import BiLSTMForecaster
from .cnn_lstm_model      import CNNLSTMForecaster
from .attention_lstm_model import AttentionLSTMForecaster
from .transformer_model   import TransformerForecaster

MODEL_REGISTRY = {
    "lstm":           LSTMForecaster,
    "gru":            GRUForecaster,
    "bilstm":         BiLSTMForecaster,
    "cnn_lstm":       CNNLSTMForecaster,
    "attention_lstm": AttentionLSTMForecaster,
    "transformer":    TransformerForecaster,
}

def get_model(name: str, **kwargs):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](**kwargs)
