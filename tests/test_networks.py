# -*- coding: utf-8 -*-
from typing import Tuple

import pytest
import torch as th

from tab_pfn.networks import SCM, DataEncoder


@pytest.mark.parametrize("batch_size", [2, 3])
@pytest.mark.parametrize("drop_proba", [0.1, 0.2])
@pytest.mark.parametrize("n_features", [2, 3])
@pytest.mark.parametrize("layers_bound", [(4, 8), (8, 16)])
@pytest.mark.parametrize("node_bound", [(8, 16), (16, 32)])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_scm(
    batch_size: int,
    drop_proba: float,
    n_features: int,
    layers_bound: Tuple[int, int],
    node_bound: Tuple[int, int],
    device: str,
) -> None:
    scm = SCM(drop_proba, n_features, layers_bound, node_bound)
    scm.to(device)

    x, y = scm(batch_size)

    assert len(x.size()) == 2
    assert x.size(0) == batch_size
    assert x.size(1) == n_features

    assert len(y.size()) == 1
    assert y.size(0) == batch_size


@pytest.mark.parametrize("batch_size", [2, 3])
@pytest.mark.parametrize("x_dim", [64, 128])
@pytest.mark.parametrize("nb_class", [3, 4])
@pytest.mark.parametrize("y_emb_dim", [64, 128])
@pytest.mark.parametrize("hidden_dim", [64, 128])
@pytest.mark.parametrize("output_dim", [64, 128])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_data_encoder(
    batch_size: int,
    x_dim: int,
    nb_class: int,
    y_emb_dim: int,
    hidden_dim: int,
    output_dim: int,
    device: str,
) -> None:
    encoder = DataEncoder(x_dim, nb_class, y_emb_dim, hidden_dim, output_dim)
    encoder.to(device)

    x = th.randn(batch_size, x_dim, device=device)
    y = th.randint(0, nb_class, (batch_size,), device=device)

    o = encoder(x, y)

    assert len(o.size()) == 2
    assert o.size(0) == batch_size
    assert o.size(1) == output_dim
