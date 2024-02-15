# -*- coding: utf-8 -*-
from typing import Tuple

import pytest
import torch as th

from tab_pfn.networks import PPD, SCM, DataAndLabelEncoder, DataEncoder, TabPFN


@pytest.mark.parametrize("batch_size", [2, 3])
@pytest.mark.parametrize("n_features", [2, 64])
@pytest.mark.parametrize("class_bound", [(2, 4), (4, 8)])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_scm(
    batch_size: int,
    n_features: int,
    class_bound: Tuple[int, int],
    device: str,
) -> None:
    scm = SCM(n_features, class_bound)
    scm.to(device)

    x, y = scm(batch_size)

    assert len(x.size()) == 2
    assert x.size(0) == batch_size
    assert x.size(1) == n_features

    assert len(y.size()) == 1
    assert y.size(0) == batch_size
    assert th.all(y >= 0)
    assert th.all(y <= class_bound[1])


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
    data_lbl_enc = DataAndLabelEncoder(
        x_dim, nb_class, y_emb_dim, hidden_dim, output_dim
    )
    data_lbl_enc.to(device)

    date_enc = DataEncoder(x_dim, hidden_dim, output_dim)
    date_enc.to(device)

    x = th.randn(batch_size, x_dim, device=device)
    y = th.randint(0, nb_class, (batch_size,), device=device)

    o = data_lbl_enc(x, y)

    assert len(o.size()) == 2
    assert o.size(0) == batch_size
    assert o.size(1) == output_dim

    o = date_enc(x)

    assert len(o.size()) == 2
    assert o.size(0) == batch_size
    assert o.size(1) == output_dim


@pytest.mark.parametrize("nb_train", [8, 16])
@pytest.mark.parametrize("nb_test", [8, 16])
@pytest.mark.parametrize("model_dim", [8, 16])
@pytest.mark.parametrize("hidden_dim", [16, 32])
@pytest.mark.parametrize("nheads", [1, 2])
@pytest.mark.parametrize("num_layers", [1, 2])
@pytest.mark.parametrize("nb_class", [2, 3])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_ppd(
    nb_train: int,
    nb_test: int,
    model_dim: int,
    hidden_dim: int,
    nheads: int,
    num_layers: int,
    nb_class: int,
    device: str,
) -> None:
    pfn = PPD(model_dim, hidden_dim, nheads, num_layers, nb_class)
    pfn.to(device)

    x_train = th.randn(nb_train, model_dim, device=device)
    x_test = th.randn(nb_test, model_dim, device=device)

    out = pfn(x_train, x_test)

    assert len(out.size()) == 2
    assert out.size(0) == nb_test
    assert out.size(1) == nb_class


@pytest.mark.parametrize("nb_train", [8, 16])
@pytest.mark.parametrize("nb_test", [8, 16])
@pytest.mark.parametrize("max_features", [4, 8])
@pytest.mark.parametrize("nb_class", [2, 3])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_tab_pfn(
    nb_train: int,
    nb_test: int,
    max_features: int,
    nb_class: int,
    device: str,
) -> None:
    encoder_dim = 2
    model_dim = 2
    hidden_dim = 2
    nheads = 1
    num_layers = 1

    tab_pfn = TabPFN(
        max_features,
        nb_class,
        encoder_dim,
        model_dim,
        hidden_dim,
        nheads,
        num_layers,
    )
    tab_pfn.to(device)

    x_train = th.randn(nb_train, max_features, device=device)
    y_train = th.randint(0, nb_class, (nb_train,), device=device)

    x_test = th.randn(nb_test, max_features, device=device)

    out = tab_pfn(x_train, y_train, x_test)

    assert len(out.size()) == 2
    assert out.size(0) == nb_test
    assert out.size(1) == nb_class
