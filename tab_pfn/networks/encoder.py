# -*- coding: utf-8 -*-
from typing import Callable

import torch as th
from torch import nn


def _init_encoder(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.xavier_normal_(module.weight)

        if module.bias is not None:
            nn.init.normal_(module.bias)

    elif isinstance(module, nn.BatchNorm1d):
        if module.affine:
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0.0)


class _SeqWrapper(nn.Module):
    def __init__(self, module: Callable[[th.Tensor], th.Tensor]) -> None:
        super().__init__()
        self.__module = module

    def forward(self, x: th.Tensor) -> th.Tensor:
        b, l, c = x.size()
        return self.__module(x.view(-1, c)).view(b, l, -1)


class DataEncoder(nn.Sequential):
    def __init__(
        self, x_max_dim: int, hidden_dim: int, output_dim: int
    ) -> None:
        super().__init__(
            nn.Linear(x_max_dim, hidden_dim),
            nn.Mish(),
            _SeqWrapper(nn.BatchNorm1d(hidden_dim)),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            _SeqWrapper(nn.BatchNorm1d(hidden_dim)),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
            nn.Mish(),
            _SeqWrapper(nn.BatchNorm1d(output_dim)),
            nn.Dropout(0.1),
        )

        self.apply(_init_encoder)


class DataAndLabelEncoder(nn.Module):
    def __init__(
        self,
        x_max_dim: int,
        nb_class_max: int,
        y_emb_dim: int,
        hidden_dim: int,
        output_dim: int,
    ) -> None:
        super().__init__()

        self.__y_emb = nn.Embedding(nb_class_max, y_emb_dim, max_norm=1.0)
        self.__encoder = DataEncoder(
            x_max_dim + y_emb_dim, hidden_dim, output_dim
        )

    def forward(self, x: th.Tensor, y: th.Tensor) -> th.Tensor:
        y_emb = self.__y_emb(y)

        out = th.cat([x, y_emb], dim=2)
        out = self.__encoder(out)

        return out
