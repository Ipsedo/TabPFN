# -*- coding: utf-8 -*-
from typing import NamedTuple

from .networks import SCM, TabPFN


class ModelOptions(NamedTuple):
    max_features: int
    max_class: int
    encoder_dim: int
    ppd_dim: int
    ppd_hidden_dim: int
    nheads: int
    num_layers: int
    cuda: bool

    def get_tab_pfn(self) -> TabPFN:
        return TabPFN(
            self.max_features,
            self.max_class,
            self.encoder_dim,
            self.ppd_dim,
            self.ppd_hidden_dim,
            self.nheads,
            self.num_layers,
        )

    def get_scm(self) -> SCM:
        return SCM(
            0.4,
            self.max_features,
            (2, self.max_class),
        )


class TrainOptions(NamedTuple):
    run_name: str
    learning_rate: float
    n_datasets: int
    n_data: int
    data_ratio: float
    batch_size: int
    output_folder: str
