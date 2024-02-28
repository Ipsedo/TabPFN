# -*- coding: utf-8 -*-
import math
from typing import Tuple

import torch as th
from torch.distributions import Normal


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


def repeat_features(x: th.Tensor, max_features: int) -> th.Tensor:
    actual_size = x.size(1)

    missing_features = max_features - actual_size
    nb_repeat = missing_features // actual_size + 1
    out = x.repeat(1, nb_repeat)
    out = th.cat([out, x[:, : missing_features % actual_size]], dim=1)
    return out


def normalize_repeat_features(
    x_to_pad: th.Tensor, max_features: int
) -> th.Tensor:
    out_mean = x_to_pad.mean(dim=-2, keepdim=True)
    out_std = x_to_pad.std(dim=-2, keepdim=True) + 1e-8

    x_to_pad = (x_to_pad - out_mean) / out_std
    out = repeat_features(x_to_pad, max_features)

    return out
