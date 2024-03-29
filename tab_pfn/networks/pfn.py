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

        dropout = 0.0
        layer_norm_eps = 1e-5

        self.__trf_enc = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                model_dim,
                nheads,
                hidden_dim,
                dropout=dropout,
                activation=F.mish,
                layer_norm_eps=layer_norm_eps,
                batch_first=True,
            ),
            num_layers,
            enable_nested_tensor=False,
        )

        self.__trf_dec = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                model_dim,
                nheads,
                hidden_dim,
                dropout=dropout,
                activation=F.mish,
                layer_norm_eps=layer_norm_eps,
                batch_first=True,
            ),
            num_layers,
        )

        self.__nheads = nheads

        self.__to_class = nn.Linear(model_dim, nb_class)

    def __get_device(self) -> str:
        return "cuda" if next(self.parameters()).is_cuda else "cpu"

    def __get_src_mask(self, x_train: th.Tensor) -> th.Tensor:
        return th.zeros(
            (
                x_train.size(0) * self.__nheads,
                x_train.size(1),
                x_train.size(1),
            ),
            device=self.__get_device(),
        )

    def __get_tgt_mask(self, x_test: th.Tensor) -> th.Tensor:
        device = self.__get_device()

        tgt_mask = (
            th.zeros(
                x_test.size(0) * self.__nheads,
                x_test.size(1),
                x_test.size(1),
                device=device,
            )
            - th.inf
        )

        diag_indices = th.arange(x_test.size(1), device=device)
        tgt_mask[:, diag_indices, diag_indices] = th.zeros(
            x_test.size(1), device=device
        )

        return tgt_mask

    def forward(self, x_train: th.Tensor, x_test: th.Tensor) -> th.Tensor:
        out_enc = self.__trf_enc(x_train, mask=self.__get_src_mask(x_train))
        out_dec = self.__trf_dec(
            x_test, out_enc, tgt_mask=self.__get_tgt_mask(x_test)
        )

        out: th.Tensor = self.__to_class(out_dec)

        return out


class TabPFN(nn.Module):
    def __init__(
        self,
        max_features: int,
        max_nb_class: int,
        ppd_dim: int,
        ppd_hidden_dim: int,
        nheads: int,
        num_layers: int,
    ) -> None:
        super().__init__()

        self.__max_features = max_features

        self.__data_lbl_enc = nn.Sequential(
            DataAndLabelEncoder(
                max_features,
                max_nb_class,
                ppd_dim,
            ),
            nn.Mish(),
            nn.LayerNorm(ppd_dim),
        )

        self.__data_enc = nn.Sequential(
            DataEncoder(
                max_features,
                ppd_dim,
            ),
            nn.Mish(),
            nn.LayerNorm(ppd_dim),
        )

        self.__trf = PPD(
            ppd_dim, ppd_hidden_dim, nheads, num_layers, max_nb_class
        )

    def forward(
        self, x_train: th.Tensor, y_train: th.Tensor, x_test: th.Tensor
    ) -> th.Tensor:
        assert len(x_train.size()) == 3
        assert x_train.size(2) == self.__max_features

        assert len(y_train.size()) == 2
        assert x_train.size(0) == y_train.size(0)
        assert x_train.size(1) == y_train.size(1)

        assert len(x_test.size()) == 3
        assert x_test.size(2) == self.__max_features
        assert x_test.size(0) == x_train.size(0)

        train_enc = self.__data_lbl_enc((x_train, y_train))
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
