"""Shared device selection for the whole project.

Resolves to the first CUDA device when one is available, otherwise falls back
to CPU. Every tensor/device transfer in the codebase should target DEVICE
rather than hardcoding ``.cpu()`` or ``.cuda()``.
"""
import torch

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
