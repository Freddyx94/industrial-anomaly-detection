"""Choose the compute device and fix random seeds.

Why: the same code must run on a Mac (Apple GPU via MPS), in CI (CPU only) and on an NVIDIA
machine (CUDA) without edits, so the device is never hard-coded. Seeds make "random" steps
(sample choice, weight initialisation, shuffling) repeat exactly, so two runs can be compared.
"""

import random

import numpy as np
import torch


def get_device() -> torch.device:
    """Return the fastest available device: cuda, then mps, then cpu."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch.

    torch.manual_seed also seeds CUDA and MPS. Some GPU operations are still not bit-exact
    between runs, so expect tiny differences on GPU, not different conclusions.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
