# -*- coding: utf-8 -*-
import math
from typing import Tuple

import torch as th
from torch.distributions import Normal
from torch.nn import functional as F


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

    loi = Normal(mu, sigma)
    sample: th.Tensor = (
        loi.icdf((th.rand(*sizes) - 1) * (1 - loi.cdf(th.tensor(0))) + 1)
        + min_value
    )

    return sample


def tnlu_int(
    mu_min: int,
    mu_max: int,
    min_value: int,
) -> int:
    return int(
        th.round(tnlu((1,), mu_min, mu_max, min_value)).to(th.int).item()
    )


def tnlu_float(
    mu_min: float,
    mu_max: float,
    min_value: float,
) -> float:
    return tnlu((1,), mu_min, mu_max, min_value).item()


def pad_features(x_to_pad: th.Tensor, max_features: int) -> th.Tensor:
    actual_size = x_to_pad.size(1)

    return (
        F.pad(
            x_to_pad,
            (0, max_features - actual_size),
            mode="constant",
            value=0,
        )
        * max_features
        / actual_size
    )
