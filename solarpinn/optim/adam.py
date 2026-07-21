"""
Adam optimizer.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch

from solarpinn.nn.parameter import Parameter

from .optimizer import Optimizer

class Adam(Optimizer):
    """
    Adaptive Moment Estimation (ADAM) optimizer.
    """

    def __init__(
            self,
            parameters: Iterable[Parameter],
            lr: float = 1e-3,
            betas: tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-8,
            weight_decay: float = 0.0
    )-> None:
        
        super().__init__(parameters, lr)

        beta1, beta2 = betas

        assert 0.0 <= beta1 < 1.0
        assert 0.0 <= beta2 < 1.0
        assert eps > 0.0
        assert weight_decay >= 0.0

        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay

        self._step = 0

        self._m: dict[int, torch.Tensor] ={}
        self._v: dict[int, torch.Tensor] = {}

    def step(self)-> None:
        """
        Perform an optimzation step.
        """

        self._step += 1
        
        with torch.no_grad():
            for parameter in self.parameters:

                if parameter.data.grad is None:
                    continue
                
                grad = parameter.data.grad

                if self.weight_decay != 0.0:
                    grad = grad + self.weight_decay * parameter.data
                
                key = id(parameter)
                
                if key not in self._m:
                    self._m[key] = torch.zeros_like(parameter.data)
                    self._v[key] = torch.zeros_like(parameter.data)
                
                m = self._m[key]
                v = self._v[key]

                m.mul_(self.beta1)
                m.add_(grad, alpha=1.0 - self.beta1)

                v.mul_(self.beta2)
                v.addcmul_(grad, grad, value=1.0 - self.beta2)

                m_hat = m / (1.0 - self.beta1 ** self._step)
                v_hat = v / (1.0 - self.beta2 ** self._step)

                parameter.data.addcdiv_(
                    m_hat,
                    torch.sqrt(v_hat) + self.eps,
                    value=-self.lr,
                )


