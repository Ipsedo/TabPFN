# -*- coding: utf-8 -*-
import math
from typing import Tuple

import torch as th
from torch import nn


def init_scm(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        module.weight.data.copy_(
            truncated_noise_log_uniform(
                module.weight.size(), 1e-2, 10, False, 0.0
            )
        )


def truncated_noise_log_uniform(
    sizes: Tuple[int, ...],
    mu_min: float,
    mu_max: float,
    to_int: bool,
    min_value: float,
) -> th.Tensor:
    log_min = math.log(mu_min)
    log_max = math.log(mu_max)

    sigma = th.exp((log_max - log_min) * th.rand(sizes) + log_min)
    mu = th.exp((log_max - log_min) * th.rand(sizes) + log_min)

    sample = th.clamp(th.normal(mu, sigma), 0.0, th.inf) + min_value

    return th.round(sample).to(th.int) if to_int else sample


def beta(x: th.Tensor, y: th.Tensor) -> th.Tensor:
    return th.exp(th.lgamma(x) + th.lgamma(y) - th.lgamma(x + y))
