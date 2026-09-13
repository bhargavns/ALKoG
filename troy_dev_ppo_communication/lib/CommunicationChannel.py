"""DIAL's noisy training channel and four-symbol binary execution channel."""

import math

import torch
from torch import nn


class CommunicationChannel(nn.Module):
    def __init__(self, sigma=2.0):
        super().__init__()
        if not math.isfinite(sigma) or sigma < 0:
            raise ValueError("sigma must be finite and nonnegative")
        self.sigma = float(sigma)

    def forward(self, logits, noise=None, *, hard=False):
        if logits.shape[-1] != 2:
            raise ValueError("The four-symbol channel requires two logits")
        if hard:
            return (logits > 0).to(logits.dtype)
        if noise is None:
            noise = torch.randn_like(logits)
        if noise.shape != logits.shape:
            raise ValueError("Channel noise must have the same shape as logits")
        return torch.sigmoid(logits + self.sigma * noise)


def bits_to_symbols(bits):
    """Threshold only for execution/logging, never on the training gradient path."""
    hard = (bits > 0.5).long()
    return 2 * hard[..., 0] + hard[..., 1]


def symbols_to_bits(symbols, dtype=torch.float32):
    return torch.stack((symbols // 2, symbols % 2), dim=-1).to(dtype)


def message_weights(bits):
    """Continuous interpolation of the a=00, b=01, c=10, d=11 embeddings."""
    b0, b1 = bits.unbind(-1)
    return torch.stack(((1-b0)*(1-b1), (1-b0)*b1, b0*(1-b1), b0*b1), -1)
