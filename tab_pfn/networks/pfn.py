# -*- coding: utf-8 -*-
from statistics import mean

import numpy as np
import torch as th
from torch import nn
from torch.nn import functional as F

from .encoder import DataAndLabelEncoder, DataEncoder


class PPD(nn.Module):
    def __init__(
        self,
        model_dim: int,
        hidden_dim: int,
        nheads: int,
        num_layers: int,
        nb_class: int,
    ) -> None:
        super().__init__()

        self.__trf = nn.Transformer(
            model_dim,
            nheads,
            num_layers,
            num_layers,
            hidden_dim,
            dropout=0.1,
            activation=F.gelu,
            batch_first=True,
        )

        self.__nheads = nheads

        self.__to_class = nn.Sequential(
            nn.Linear(model_dim, nb_class),
            nn.Softmax(dim=-1),
        )

    def forward(self, x_train: th.Tensor, x_test: th.Tensor) -> th.Tensor:
        device = "cuda" if next(self.parameters()).is_cuda else "cpu"

        src_mask = th.ones(
            (
                x_train.size(0) * self.__nheads,
                x_train.size(1),
                x_train.size(1),
            ),
            device=device,
        )
        tgt_mask = th.eye(x_test.size(1), device=device)[None].repeat(
            x_test.size(0) * self.__nheads, 1, 1
        )

        out: th.Tensor = self.__trf(
            x_train, x_test, src_mask=src_mask, tgt_mask=tgt_mask
        )

        out = self.__to_class(out)

        return out


class TabPFN(nn.Module):
    def __init__(
        self,
        max_features: int,
        max_nb_class: int,
        encoder_dim: int,
        ppd_dim: int,
        ppd_hidden_dim: int,
        nheads: int,
        num_layers: int,
    ) -> None:
        super().__init__()

        self.__max_features = max_features

        self.__data_lbl_enc = DataAndLabelEncoder(
            max_features,
            max_nb_class,
            max_features,
            encoder_dim,
            ppd_dim,
        )

        self.__data_enc = DataEncoder(
            max_features,
            encoder_dim,
            ppd_dim,
        )

        self.__trf = PPD(
            ppd_dim, ppd_hidden_dim, nheads, num_layers, max_nb_class
        )

    def forward(
        self, x_train: th.Tensor, y_train: th.Tensor, x_test: th.Tensor
    ) -> th.Tensor:
        x_mean = x_train.mean(dim=1, keepdim=True)
        x_std = x_train.std(dim=1, keepdim=True) + 1e-5

        x_train = (x_train - x_mean) / x_std
        x_test = (x_test - x_mean) / x_std

        train_enc = self.__data_lbl_enc(x_train, y_train)
        test_enc = self.__data_enc(x_test)

        out: th.Tensor = self.__trf(train_enc, test_enc)

        return out

    def count_parameters(self) -> int:
        return sum(int(np.prod(p.size())) for p in self.parameters())

    def grad_norm(self) -> float:
        return mean(
            float(p.grad.norm().item())
            for p in self.parameters()
            if p.grad is not None
        )

    @property
    def nb_features(self) -> int:
        return self.__max_features
