"""
Weiht initialization functions.
"""

from __future__ import annotations

import math

import torch

def _check_tensor(tensor: torch.Tensor)-> None:
    """
    Validate that a tensor can be initialized.
    """

    assert isinstance(tensor, torch.Tensor), \
        "Expected a torch.Tensor."
    
    assert tensor.ndim >= 2, \
        "Tensor must have atleast two dimensions."
    

def zeros(
        tensor: torch.Tensor
)-> torch.Tensor:
    """
    Fill a tensor with zeros.
    """
    tensor.zero_()
    
    return tensor

def ones(
        tensor: torch.Tensor
)-> torch.Tensor:
    """
    Fill a tensor with ones.
    """

    tensor.fill_(1.0)

    return tensor

def uniform(
        tensor: torch.Tensor,
        low: float = -1.0,
        high: float = 1.0
)-> torch.Tensor:
    """
    Uniform initialization.
    """
    tensor.uniform_(low, high)

    return tensor

def normal(
        tensor: torch.Tensor,
        mean: float = 0.0,
        std: float  = 1.0
)-> torch.Tensor:
    """
    Normal initialization.
    """
    tensor.normal_(mean, std)

    return tensor

def calculate_fan_in_and_fan_out(
        tensor: torch.Tensor

)-> tuple[int, int]:
    """
    Compute fan-in and fan-out
    """
    _check_tensor(tensor)

    fan_out = tensor.shape[0]
    fan_in = tensor.shape[1]

    if tensor.ndim > 2:
        receptive_field = math.prod(tensor.shape[2:])

        fan_in *= receptive_field
        fan_out *= receptive_field

    return fan_in, fan_out

def xavier_uniform(
        tensor: torch.Tensor
)-> torch.Tensor:
    """
    Xavier (Glorot) uniform initialization.

    References
    ----------
    Glorot & Bengio (2010)
    """
    fan_in, fan_out = calculate_fan_in_and_fan_out(tensor)

    bound = math.sqrt(6.0 / (fan_in + fan_out))

    tensor.uniform_(-bound, bound)

    return tensor

def xavier_normal(
    tensor: torch.Tensor,
) -> torch.Tensor:
    """
    Xavier normal initialization.
    """

    fan_in, fan_out = calculate_fan_in_and_fan_out(tensor)

    std = math.sqrt(2.0 / (fan_in + fan_out))

    tensor.normal_(0.0, std)

    return tensor

def kaiming_uniform(
    tensor: torch.Tensor,
) -> torch.Tensor:
    """
    Kaiming (He) uniform initialization.
    """

    fan_in, _ = calculate_fan_in_and_fan_out(tensor)

    bound = math.sqrt(6.0 / fan_in)

    tensor.uniform_(-bound, bound)

    return tensor

def kaiming_normal(
    tensor: torch.Tensor,
) -> torch.Tensor:
    """
    Kaiming (He) normal initialization.
    """

    fan_in, _ = calculate_fan_in_and_fan_out(tensor)

    std = math.sqrt(2.0 / fan_in)

    tensor.normal_(0.0, std)

    return tensor

