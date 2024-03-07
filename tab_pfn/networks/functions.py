# -*- coding: utf-8 -*-
import math
from typing import Literal, Tuple

import torch as th
from torch.distributions import Normal
from torch.nn import functional as F


# truncated noise log uniform
def tnlu(
    sizes: Tuple[int, ...],
    mu_min: float,
    mu_max: float,
    min_value: float,
    device: Literal["cpu", "cuda"] = "cpu",
) -> th.Tensor:
    log_min = math.log(mu_min)
    log_max = math.log(mu_max)

    sigma = th.exp(
        (log_max - log_min) * th.rand(sizes, device=device) + log_min
    )
    mu = th.exp((log_max - log_min) * th.rand(sizes, device=device) + log_min)

    loi = Normal(mu, sigma)

    sample: th.Tensor = (
        th.clamp_min(
            loi.icdf(
                (th.rand(sizes, device=device) - 1)
                * (1 - loi.cdf(th.zeros(1, device=device)))
                + 1
            ),
            0,
        )
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
    actual_size = x.size(-1)

    missing_features = max_features - actual_size
    nb_repeat = missing_features // actual_size + 1

    x_slice = [slice(0, s) for s in x.size()]
    x_slice[-1] = slice(0, missing_features % actual_size)

    return th.cat([x.repeat_interleave(nb_repeat, dim=-1), x[x_slice]], dim=-1)


def pad_features(x: th.Tensor, max_features: int) -> th.Tensor:
    return (
        F.pad(x, (0, max(max_features - x.size(-1), 0)))
        * max_features
        / x.size(-1)
    )


def normalize_features(x: th.Tensor) -> th.Tensor:
    return (x - x.mean(dim=-2, keepdim=True)) / (
        x.std(dim=-2, keepdim=True) + 1e-8
    )


def normalize_pad_features(x: th.Tensor, max_features: int) -> th.Tensor:
    return pad_features(normalize_features(x), max_features)


def normalize_repeat_features(x: th.Tensor, max_features: int) -> th.Tensor:
    return repeat_features(normalize_features(x), max_features)
