# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from typing import Dict, Optional, TypeVar

import numpy as np
import pandas as pd
import torch as th
from pandas.api.types import is_numeric_dtype
from sklearn.base import BaseEstimator, ClassifierMixin

from .functions import normalize_repeat_features
from .pfn import TabPFN

T = TypeVar("T", pd.DataFrame, np.ndarray, th.Tensor)
Ty = TypeVar("Ty", pd.Series, np.ndarray, th.Tensor)


class SklearnClassifier(ABC, ClassifierMixin, BaseEstimator):
    @abstractmethod
    def fit(self, x: T, y: Ty) -> None:
        pass

    @abstractmethod
    def predict(self, x: T) -> T:
        pass

    @abstractmethod
    def predict_proba(self, x: T) -> T:
        pass

    @staticmethod
    def from_torch(tab_pfn: TabPFN) -> "SklearnClassifier":
        return _SklearnTabPFN(tab_pfn)


class _SklearnTabPFN(SklearnClassifier):
    def __init__(self, torch_model: TabPFN) -> None:
        super().__init__()

        self.__model = torch_model

        self.__x_train: Optional[th.Tensor] = None
        self.__y_train: Optional[th.Tensor] = None

        self.__class_to_idx: Dict[str, int] = {}

    def fit(self, x: T, y: Ty) -> None:
        x_tensor = self.__convert_x_to_tensor(x)
        y_tensor = self.__convert_y_to_tensor(y)

        if len(self.__class_to_idx) == 0:
            # case of numpy and torch input
            self.__class_to_idx = {
                c: i for i, c in enumerate(sorted(th.unique(y_tensor)))
            }

        assert len(x_tensor.size()) == 2
        assert x_tensor.size(1) == self.__model.nb_features

        assert len(y_tensor.size()) == 1
        assert x_tensor.size(0) == y_tensor.size(0)

        assert th.all(th.ge(y_tensor, 0))
        assert th.all(th.lt(y_tensor, len(self.__class_to_idx)))

        device = th.device(
            "cuda" if next(self.__model.parameters()).is_cuda else "cpu"
        )

        self.__x_train = x_tensor.to(device=device)[None]
        self.__y_train = y_tensor.to(device=device)[None]

    def predict(self, x: T) -> T:
        out = self.__predict_proba_tensor(x).argmax(dim=-1)
        return self.__tensor_to_input_type(out, x)

    def predict_proba(self, x: T) -> T:
        out = th.softmax(self.__predict_proba_tensor(x), dim=-1)
        return self.__tensor_to_input_type(out, x)

    def __predict_proba_tensor(self, x: T) -> th.Tensor:
        device = th.device(
            "cuda" if next(self.__model.parameters()).is_cuda else "cpu"
        )

        x_test = self.__convert_x_to_tensor(x).to(device)

        assert len(x_test.size()) == 2
        assert x_test.size(1) == self.__model.nb_features

        assert (
            self.__x_train is not None
        ), "Call fit() before predict (x_train is None)"
        assert (
            self.__y_train is not None
        ), "Call fit() before predict (y_train is None)"

        with th.no_grad():
            out: th.Tensor = self.__model(
                self.__x_train, self.__y_train, x_test[None]
            )[0]

        return th.nan_to_num(out[:, : len(self.__class_to_idx)], nan=0)

    def __tensor_to_input_type(self, out: th.Tensor, x_test: T) -> T:
        if isinstance(x_test, th.Tensor):
            return out
        if isinstance(x_test, np.ndarray):
            array: np.ndarray = out.cpu().numpy()
            return array
        if isinstance(x_test, pd.DataFrame):
            array = out.cpu().numpy()
            if len(out.size()) > 1:
                df = pd.DataFrame(
                    array, columns=list(self.__class_to_idx.keys())
                )
                return df[sorted(df.columns)]
            idx_to_class = {idx: c for c, idx in self.__class_to_idx.items()}
            pred_class = [idx_to_class[c] for c in array]
            return pd.DataFrame(pred_class, columns=["prediction"])

        raise TypeError(f"Unsupported input type: {type(x_test)}")

    def __convert_x_to_tensor(self, x: T) -> th.Tensor:
        if isinstance(x, pd.DataFrame):
            out = self.__x_df_to_tensor(x)
        elif isinstance(x, np.ndarray):
            out = self.__x_np_to_tensor(x)
        elif isinstance(x, th.Tensor):
            out = x
        else:
            raise TypeError(f"Unrecognized input type {type(x)}")

        return normalize_repeat_features(
            out.to(th.float), self.__model.nb_features
        )

    def __convert_y_to_tensor(self, y: T) -> th.Tensor:
        if isinstance(y, pd.Series):
            out = self.__y_df_to_tensor(y)
        elif isinstance(y, np.ndarray):
            out = self.__y_np_to_tensor(y)
        elif isinstance(y, th.Tensor):
            out = y
        else:
            raise TypeError(f"Unrecognized input type {type(y)}")

        return out.to(th.long)

    def __x_df_to_tensor(self, x: pd.DataFrame) -> th.Tensor:
        for c in x.columns:
            if not is_numeric_dtype(x[c]):
                ohe = pd.get_dummies(
                    x[c], prefix=c, prefix_sep="_", dtype=float
                )
                x = x.drop(c, axis=1).join(ohe)
                if len(x.columns) > self.__model.nb_features:
                    break

        if len(x.columns) > self.__model.nb_features:
            raise ValueError(
                f"pandas.DataFrame contains "
                f"more than {self.__model.nb_features} columns, "
                f"input {len(x.columns)}"
            )

        return th.tensor(x.to_numpy())

    def __y_df_to_tensor(self, y: pd.Series) -> th.Tensor:
        self.__class_to_idx = {c: i for i, c in enumerate(y.unique())}

        y = y.apply(lambda c: self.__class_to_idx[c])

        return th.tensor(y.to_numpy())

    @staticmethod
    def __x_np_to_tensor(x: np.ndarray) -> th.Tensor:
        assert len(x.shape) == 2
        return th.tensor(x)

    @staticmethod
    def __y_np_to_tensor(y: np.ndarray) -> th.Tensor:
        assert len(y.shape) == 1
        return th.tensor(y)
