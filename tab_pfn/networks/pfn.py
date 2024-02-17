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

        self.__trf = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                model_dim,
                nheads,
                hidden_dim,
                activation=F.mish,
                batch_first=True,
                dropout=0.1,
            ),
            num_layers,
            enable_nested_tensor=False,
        )

        self.__to_class = nn.Linear(model_dim, nb_class)

    @staticmethod
    def __get_mask(
        x_train: th.Tensor, x_test: th.Tensor, device: str
    ) -> th.Tensor:
        mask = th.eye(x_train.size(1) + x_test.size(1), device=device)

        mask[:, : x_train.size(1)] = 1

        return mask

    def forward(self, x_train: th.Tensor, x_test: th.Tensor) -> th.Tensor:
        device = "cuda" if next(self.parameters()).is_cuda else "cpu"

        src_mask = self.__get_mask(x_train, x_test, device)

        enc_input = th.cat([x_train, x_test], dim=1)

        out: th.Tensor = self.__trf(enc_input, mask=src_mask)[
            :, x_train.size(1) :, :
        ]
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
