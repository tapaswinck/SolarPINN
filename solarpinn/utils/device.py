"""
Device selection utilities
"""

from __future__ import annotations

import torch

def get_device(preference: str = "auto")-> torch.device:
    """
    Select the best available compute device.

    Parameters
    ----------
    preference: str
        "auto", "cpu", "cuda" or "mps"

    Returns
    -------
    torch.device
    """

    preference = preference.lower()

    if preference == "cpu":
        return torch.device("cpu")
    
    if preference == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available.")
        return torch.device("cuda")
    
    if preference == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("Apple MPS is not available.")
        return torch.device("mps")
    
    #Automatic selection
    if torch.cuda.is_available():
        return torch.device("cuda")
    
    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")

