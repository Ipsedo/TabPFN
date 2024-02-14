# -*- coding: utf-8 -*-
from typing import Tuple

import pytest

from tab_pfn.networks import SCM


@pytest.mark.parametrize("batch_size", [2, 3])
@pytest.mark.parametrize("drop_proba", [0.1, 0.2])
@pytest.mark.parametrize("n_features", [2, 3])
@pytest.mark.parametrize("layers_bound", [(4, 8), (8, 16)])
@pytest.mark.parametrize("node_bound", [(8, 16), (16, 32)])
def test_scm(
    batch_size: int,
    drop_proba: float,
    n_features: int,
    layers_bound: Tuple[int, int],
    node_bound: Tuple[int, int],
) -> None:
    scm = SCM(drop_proba, n_features, layers_bound, node_bound)

    out = scm(batch_size)

    assert len(out.size()) == 2
    assert out.size(0) == batch_size
    assert out.size(1) == n_features
