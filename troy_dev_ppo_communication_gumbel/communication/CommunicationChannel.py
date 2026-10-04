"""Straight-through categorical Gumbel communication channel."""
import math
import torch
from torch import nn


class StraightThroughGumbelChannel(nn.Module):
    """Discrete categories forward, temperature-smoothed gradients backward.

    Store/replay the parameter-independent Gumbel noise for PPO. ``hard=True``
    means deterministic argmax evaluation; training is also discrete, but sampled.
    """
    # Parameters:
    #   temperature: Positive Gumbel surrogate temperature; changes gradients, not discrete sampling probabilities.
    #   n_symbols: Number of categories in the communication vocabulary (V), at least two.
    def __init__(self, temperature=1.2, n_symbols=4):
        super().__init__()
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError('temperature must be finite and positive')
        if not isinstance(n_symbols, int) or n_symbols < 2:
            raise ValueError("n_symbols must be an integer >= 2")
        self.n_symbols = n_symbols
        self.temperature = float(temperature)

    # Parameters:
    #   shape: Requested Gumbel noise tensor shape; usually [V] or [B,V].
    #   device: PyTorch device for policy execution/tensor storage, e.g. cpu or cuda:0.
    @staticmethod
    def sample_noise(shape, device):
        uniform_samples = torch.rand(shape, device=device).clamp(1e-7, 1-1e-7)
        return -torch.log(-torch.log(uniform_samples))

    # Parameters (B = batch size, K = scene rows, V = message categories):
    #   symbol_logits: Unnormalized sender category scores [...,V]; entropy averages over the leading axes.
    #   gumbel_noise: Parameter-independent Gumbel draws matching symbol_logits; forward() samples them if None.
    def soft(self, symbol_logits, gumbel_noise):
        return ((symbol_logits + gumbel_noise) / self.temperature).softmax(-1)

    # Parameters (B = batch size, K = scene rows, V = message categories):
    #   symbol_logits: Unnormalized sender category scores [...,V]; entropy averages over the leading axes.
    #   gumbel_noise: Parameter-independent Gumbel draws matching symbol_logits; forward() samples them if None.
    #   hard: True selects deterministic argmax; False uses sampled discrete messages with surrogate gradients.
    def forward(self, symbol_logits, gumbel_noise=None, *, hard=False):
        if symbol_logits.shape[-1] != self.n_symbols:
            raise ValueError(f'Gumbel channel requires {self.n_symbols} logits')
        if hard:
            return torch.nn.functional.one_hot(symbol_logits.argmax(-1), self.n_symbols).to(symbol_logits.dtype)
        if gumbel_noise is None:
            gumbel_noise = self.sample_noise(symbol_logits.shape, symbol_logits.device)
        if gumbel_noise.shape != symbol_logits.shape:
            raise ValueError('Channel noise must have the same shape as logits')
        soft_message = self.soft(symbol_logits, gumbel_noise)
        discrete_message = torch.nn.functional.one_hot((symbol_logits + gumbel_noise).argmax(-1), self.n_symbols).to(symbol_logits.dtype)
        return discrete_message + (soft_message - soft_message.detach())
