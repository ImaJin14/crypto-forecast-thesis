"""Pytest fixtures — sample tensors and mock data."""
import pytest
import torch
import numpy as np


@pytest.fixture
def sample_batch():
    """A batch of (X, y) tensors with shape (32, 60, 20)."""
    X = torch.randn(32, 60, 20)   # batch=32, seq=60, features=20
    y = torch.randn(32, 1)
    return X, y


@pytest.fixture
def sample_prices():
    """1000 synthetic daily price points for metric testing."""
    np.random.seed(42)
    prices = np.cumsum(np.random.randn(1000)) + 100
    return prices.astype(np.float32)
