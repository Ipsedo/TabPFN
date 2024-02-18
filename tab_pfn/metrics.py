# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from os.path import join
from statistics import mean
from typing import Generic, List, Optional, Tuple, TypeVar, Union

import matplotlib.pyplot as plt
import torch as th

T = TypeVar("T")
I = TypeVar("I")


class Meter(Generic[I, T], ABC):
    def __init__(self, window_size: Optional[int]) -> None:
        self.__window_size = window_size

        self.__results: List[T] = []

    @abstractmethod
    def _process_value(self, *args: I) -> T:
        pass

    @property
    def _results(self) -> List[T]:
        return self.__results

    def add(self, *args: I) -> None:
        if (
            self.__window_size is not None
            and len(self.__results) >= self.__window_size
        ):
            self.__results.pop(0)

        self.__results.append(self._process_value(*args))

    def set_window_size(self, new_window_size: Union[int, None]) -> None:
        if new_window_size is not None:
            assert (
                new_window_size > 0
            ), f"window size must be > 0 : {new_window_size}"

        self.__window_size = new_window_size


class ConfusionMeter(Meter[th.Tensor, Tuple[th.Tensor, th.Tensor]]):
    def __init__(
        self,
        nb_class: int,
        window_size: Optional[int] = None,
    ):
        super().__init__(window_size)
        self.__nb_class = nb_class

    def _process_value(self, *args: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        y_proba = args[0]
        y_true = args[1]
        return y_proba.argmax(dim=-1), y_true

    def conf_mat(self) -> th.Tensor:
        y_pred = th.cat([y_p for y_p, _ in self._results], dim=0)
        y_true = th.cat([y_t for _, y_t in self._results], dim=0)

        conf_matrix_indices = th.multiply(y_true, self.__nb_class) + y_pred
        conf_matrix = th.bincount(
            conf_matrix_indices, minlength=self.__nb_class**2
        ).reshape(self.__nb_class, self.__nb_class)

        return conf_matrix

    def precision(self) -> th.Tensor:
        conf_mat = self.conf_mat()

        precs_sum = conf_mat.sum(dim=0)
        diag = th.diagonal(conf_mat, 0)

        precs = th.zeros(self.__nb_class, device=conf_mat.device)

        mask = precs_sum != 0

        precs[mask] = diag[mask] / precs_sum[mask]

        return precs

    def recall(self) -> th.Tensor:
        conf_mat = self.conf_mat()

        recs_sum = conf_mat.sum(dim=1)
        diag = th.diagonal(conf_mat, 0)

        recs = th.zeros(self.__nb_class, device=conf_mat.device)

        mask = recs_sum != 0

        recs[mask] = diag[mask] / recs_sum[mask]

        return recs

    def save_conf_matrix(self, epoch: int, output_dir: str) -> None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

        conf_mat = self.conf_mat()
        conf_mat_normalized = conf_mat / th.sum(conf_mat, dim=1, keepdim=True)
        cax = ax.matshow(conf_mat_normalized.tolist(), cmap="plasma")
        fig.colorbar(cax)

        ax.set_title(f"confusion matrix epoch {epoch}")
        ax.set_ylabel("True Label")
        ax.set_xlabel("Predicated Label")

        fig.savefig(join(output_dir, f"confusion_matrix_epoch_{epoch}.png"))

        plt.close()


class LossMeter(Meter[float, float]):
    def _process_value(self, *args: float) -> float:
        return args[0]

    def loss(self) -> float:
        return mean(self._results)


class AccuracyMeter(Meter[th.Tensor, Tuple[th.Tensor, th.Tensor]]):
    def _process_value(self, *args: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        y_proba = args[0]
        y_true = args[1]
        return y_proba.argmax(dim=-1), y_true

    def accuracy(self) -> float:
        y_pred = th.cat([y_p for y_p, _ in self._results], dim=0)
        y_true = th.cat([y_t for _, y_t in self._results], dim=0)

        return float((y_pred == y_true).sum().item() / y_true.view(-1).size(0))
