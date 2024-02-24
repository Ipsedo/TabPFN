# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import pytest
import torch as th

from tab_pfn.networks import SklearnClassifier, TabPFN


@pytest.mark.parametrize("n_data_train", [128, 256])
@pytest.mark.parametrize("n_data_test", [8, 16])
@pytest.mark.parametrize("n_features", [4, 8])
@pytest.mark.parametrize("n_classes", [2, 10])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_torch(
    n_data_train: int,
    n_data_test: int,
    n_features: int,
    n_classes: int,
    device: str,
) -> None:
    tab_pfn = TabPFN(10, 10, 4, 4, 4, 1, 1)
    tab_pfn.to(th.device(device))

    tab_pfn_clf = SklearnClassifier.from_torch(tab_pfn)

    x = th.randn(n_data_train, n_features)
    y = th.randint(0, n_classes, (n_data_train,))

    x_test = th.randn(n_data_test, n_features)

    tab_pfn_clf.fit(x, y)

    out = tab_pfn_clf.predict(x_test)
    out_prob = tab_pfn_clf.predict_proba(x_test)

    assert isinstance(out, th.Tensor)
    assert len(out.size()) == 1
    assert out.size(0) == n_data_test
    assert all(0 <= c <= n_classes for c in out)

    assert isinstance(out_prob, th.Tensor)
    assert len(out_prob.size()) == 2
    assert out_prob.size(0) == n_data_test
    assert out_prob.size(1) == n_classes


@pytest.mark.parametrize("n_data_train", [128, 256])
@pytest.mark.parametrize("n_data_test", [8, 16])
@pytest.mark.parametrize("n_features", [4, 8])
@pytest.mark.parametrize("n_classes", [2, 10])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_numpy(
    n_data_train: int,
    n_data_test: int,
    n_features: int,
    n_classes: int,
    device: str,
) -> None:
    tab_pfn = TabPFN(10, 10, 4, 4, 4, 1, 1)
    tab_pfn.to(th.device(device))

    tab_pfn_clf = SklearnClassifier.from_torch(tab_pfn)

    x = np.random.randn(n_data_train, n_features)
    y = np.random.randint(0, n_classes, (n_data_train,))

    x_test = np.random.randn(n_data_test, n_features)

    tab_pfn_clf.fit(x, y)

    out = tab_pfn_clf.predict(x_test)
    out_prob = tab_pfn_clf.predict_proba(x_test)

    assert isinstance(out, np.ndarray)
    assert len(out.shape) == 1
    assert out.shape[0] == n_data_test
    assert all(0 <= c <= n_classes for c in out)

    assert isinstance(out_prob, np.ndarray)
    assert len(out_prob.shape) == 2
    assert out_prob.shape[0] == n_data_test
    assert out_prob.shape[1] == n_classes


@pytest.mark.parametrize("n_data_train", [128, 256])
@pytest.mark.parametrize("n_data_test", [8, 16])
@pytest.mark.parametrize("n_features", [4, 8])
@pytest.mark.parametrize("n_classes", [2, 10])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_pandas(
    n_data_train: int,
    n_data_test: int,
    n_features: int,
    n_classes: int,
    device: str,
) -> None:
    tab_pfn = TabPFN(10, 10, 4, 4, 4, 1, 1)
    tab_pfn.to(th.device(device))

    tab_pfn_clf = SklearnClassifier.from_torch(tab_pfn)

    x = pd.DataFrame(np.random.randn(n_data_train, n_features))
    y = pd.Series(np.random.randint(0, n_classes, (n_data_train,)))

    x_test = pd.DataFrame(np.random.randn(n_data_test, n_features))

    tab_pfn_clf.fit(x, y)

    out = tab_pfn_clf.predict(x_test)
    out_prob = tab_pfn_clf.predict_proba(x_test)

    class_set = set(c for c in y.unique())

    assert isinstance(out, pd.DataFrame)
    assert len(out.columns) == 1
    assert len(out) == n_data_test
    assert all(c in class_set for c in out.iloc[:, 0])

    assert isinstance(out_prob, pd.DataFrame)
    assert len(out_prob.columns) == n_classes
    assert len(out_prob) == n_data_test
    assert all(c in class_set for c in out_prob.columns)
