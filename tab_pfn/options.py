# -*- coding: utf-8 -*-
from typing import Any, Dict, NamedTuple, Tuple

from .networks import SCM, TabPFN


class ModelOptions(NamedTuple):
    max_features: int
    max_class: int
    ppd_dim: int
    ppd_hidden_dim: int
    nheads: int
    num_layers: int
    cuda: bool

    def get_tab_pfn(self) -> TabPFN:
        return TabPFN(
            self.max_features,
            self.max_class,
            self.ppd_dim,
            self.ppd_hidden_dim,
            self.nheads,
            self.num_layers,
        )

    def get_scm(self) -> SCM:
        return SCM(
            self.max_features,
            (2, self.max_class),
            True,
        )

    def to_dict(self) -> Dict[str, Any]:
        return dict(self._asdict())


class TrainOptions(NamedTuple):
    run_name: str
    learning_rate: float
    steps: int
    batch_size: int
    n_data: int
    data_ratios: Tuple[float, float]
    save_every: int
    metric_window_size: int
    output_folder: str

    def to_dict(self) -> Dict[str, Any]:
        return dict(self._asdict())


class InferOptions(NamedTuple):
    csv_path: str
    class_col: str
    csv_sep: str
    train_ratio: float
    state_dict: str
    output_folder: str

    def to_dict(self) -> Dict[str, Any]:
        return dict(self._asdict())
