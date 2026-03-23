"""Forward pass shape validation for all models."""
import pytest
import torch
from src.models import MODEL_REGISTRY


@pytest.mark.parametrize("model_name", list(MODEL_REGISTRY.keys()))
def test_forward_pass_shape(model_name, sample_batch):
    X, y = sample_batch
    model = MODEL_REGISTRY[model_name](input_size=20)
    model.eval()
    with torch.no_grad():
        out = model(X)
    assert out.shape == (32, 1), f"{model_name}: expected (32,1), got {out.shape}"


@pytest.mark.parametrize("model_name", list(MODEL_REGISTRY.keys()))
def test_model_has_parameters(model_name):
    model = MODEL_REGISTRY[model_name](input_size=20)
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params > 0, f"{model_name} has no parameters"
