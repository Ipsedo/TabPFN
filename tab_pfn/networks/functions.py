# -*- coding: utf-8 -*-
import math
from typing import Tuple

import torch as th
from torch import nn


def init_scm(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        module.weight.data.copy_(tnlu(module.weight.size(), 1e-2, 10, 0.0))


# truncated noise log uniform
def tnlu(
    sizes: Tuple[int, ...],
    mu_min: float,
    mu_max: float,
    min_value: float,
) -> th.Tensor:
    log_min = math.log(mu_min)
    log_max = math.log(mu_max)

    sigma = th.exp((log_max - log_min) * th.rand(sizes) + log_min)
    mu = th.exp((log_max - log_min) * th.rand(sizes) + log_min)

    sample = th.clamp(th.normal(mu, sigma), 0.0, th.inf) + min_value

    return sample


def tnlu_int(
    mu_min: int,
    mu_max: int,
    min_value: int,
) -> int:
    return int(
        th.round(tnlu((1,), mu_min, mu_max, min_value)).to(th.int).item()
    )


def beta(x: th.Tensor, y: th.Tensor) -> th.Tensor:
    return th.exp(th.lgamma(x) + th.lgamma(y) - th.lgamma(x + y))
