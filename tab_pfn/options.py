# -*- coding: utf-8 -*-
from typing import NamedTuple

from .networks import SCM, TabPFN


class ModelOptions(NamedTuple):
    max_features: int
    max_class: int
    model_dim: int
    hidden_dim: int
    cuda: bool

    def get_tab_pfn(self) -> TabPFN:
        return TabPFN(
            self.max_features,
            self.max_class,
            self.model_dim,
            self.hidden_dim,
        )

    def get_scm(self) -> SCM:
        return SCM(
            0.4, self.max_features, (4, 8), (64, 128), (2, self.max_class)
        )


class TrainOptions(NamedTuple):
    run_name: str
    learning_rate: float
    n_datasets: int
    n_data: int
    data_ratio: float
    batch_size: int
    output_folder: str
