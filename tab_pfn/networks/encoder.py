# -*- coding: utf-8 -*-
from typing import Tuple

import torch as th
from torch import nn


def _init_encoder(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.xavier_normal_(module.weight)

        if module.bias is not None:
            nn.init.normal_(module.bias)


class DataEncoder(nn.Linear):
    def __init__(self, x_max_dim: int, output_dim: int) -> None:
        super().__init__(x_max_dim, output_dim)

        self.apply(_init_encoder)


class DataAndLabelEncoder(nn.Module):
    def __init__(
        self,
        x_max_dim: int,
        nb_class_max: int,
        output_dim: int,
    ) -> None:
        super().__init__()

        self.__y_emb = nn.Embedding(nb_class_max, output_dim)
        self.__encoder = DataEncoder(x_max_dim, output_dim)

    def forward(self, x_y: Tuple[th.Tensor, th.Tensor]) -> th.Tensor:
        x, y = x_y
        out: th.Tensor = self.__encoder(x) + self.__y_emb(y)
        return out
