import torch
import pytest

from nwtk.attention import ChannelAttention

@pytest.fixture
def make_tensor():
    """x-values for testing."""
    return torch.randn(16, 8 ,32)

def test_attention(make_tensor):
    chattn = ChannelAttention(dim = make_tensor.shape[2])
    assert chattn(make_tensor).shape == make_tensor.shape