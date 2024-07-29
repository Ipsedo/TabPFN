# -*- coding: utf-8 -*-
from typing import Literal, Tuple

import pytest
import torch as th

from tab_pfn.networks.functions import (
    pad_features,
    repeat_features,
    tnlu,
    tnlu_float,
    tnlu_int,
    truncated_normal_sample,
)


@pytest.mark.parametrize("sizes", [(8, 4), (4,), (8, 2, 4), (1024, 1024)])
@pytest.mark.parametrize("min_value", [-10.0, -5.0, 0.0])
@pytest.mark.parametrize("max_value", [1.0, 5.0, 10.0])
@pytest.mark.parametrize("sigma_scale", [10.0, 1.0, 1e-1])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_truncated_normal_sample(
    sizes: Tuple[int, ...],
    min_value: float,
    max_value: float,
    sigma_scale: float,
    device: str,
) -> None:
    mu = th.rand(*sizes, device=device) * (max_value - min_value) + min_value
    sigma = th.rand(*sizes, device=device) * sigma_scale

    out = truncated_normal_sample(mu, sigma, min_value, max_value)

    assert len(out.size()) == len(sizes)
    assert all(s_o == s_i for s_i, s_o in zip(out.size(), sizes))
    assert th.all(th.ge(out, min_value))
    assert th.all(th.le(out, max_value))
    assert out.device.type == device


@pytest.mark.parametrize("sizes", [(8, 4), (4,), (8, 2, 4), (1024, 1024)])
@pytest.mark.parametrize("mu_min", [0.1, 1.0])
@pytest.mark.parametrize("mu_max", [2.0, 10.0])
@pytest.mark.parametrize("min_value", [1e-2, 1.0])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_tnlu(
    sizes: Tuple[int, ...],
    mu_min: float,
    mu_max: float,
    min_value: float,
    device: Literal["cpu", "cuda"],
) -> None:
    out = tnlu(sizes, mu_min, mu_max, min_value, device)

    assert len(out.size()) == len(sizes)
    assert all(s_o == s_i for s_i, s_o in zip(out.size(), sizes))
    assert th.all(th.ge(out, min_value))
    assert out.device.type == device


@pytest.mark.parametrize("mu_min", [2, 3])
@pytest.mark.parametrize("mu_max", [4, 5])
@pytest.mark.parametrize("min_value", [0, 1])
def test_tnlu_int(mu_min: int, mu_max: int, min_value: int) -> None:
    out = tnlu_int(mu_min, mu_max, min_value)

    assert isinstance(out, int)
    assert out >= min_value


@pytest.mark.parametrize("mu_min", [0.1, 1.0])
@pytest.mark.parametrize("mu_max", [2.0, 10.0])
@pytest.mark.parametrize("min_value", [1e-2, 1.0])
def test_tnlu_float(mu_min: float, mu_max: float, min_value: float) -> None:
    out = tnlu_float(mu_min, mu_max, min_value)

    assert isinstance(out, float)
    assert out >= min_value


@pytest.mark.parametrize("sizes", [(2,), (4, 2), (2, 8, 3)])
@pytest.mark.parametrize("max_size", [10, 20])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_pad_features(
    sizes: Tuple[int, ...], max_size: int, device: str
) -> None:
    x = th.randn(sizes, device=device)

    out = pad_features(x, max_size)

    assert len(out.size()) == len(x.size())
    assert out.size(-1) == max_size
    assert all(s_o == s_i for s_o, s_i in zip(out.size()[:-1], x.size()[:-1]))
    assert out.device.type == device


@pytest.mark.parametrize("sizes", [(2,), (4, 2), (2, 8, 3)])
@pytest.mark.parametrize("max_size", [10, 20])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_repeat_features(
    sizes: Tuple[int, ...], max_size: int, device: str
) -> None:
    x = th.randn(sizes, device=device)

    out = repeat_features(x, max_size)

    assert len(out.size()) == len(x.size())
    assert out.size(-1) == max_size
    assert all(s_o == s_i for s_o, s_i in zip(out.size()[:-1], x.size()[:-1]))
    assert out.device.type == device
