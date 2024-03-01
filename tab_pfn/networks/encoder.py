# -*- coding: utf-8 -*-

import torch as th
from torch import nn


def _init_encoder(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.xavier_normal_(module.weight)

        if module.bias is not None:
            nn.init.normal_(module.bias)

    elif isinstance(module, nn.LayerNorm):
        if module.elementwise_affine:
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0.0)


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

    def forward(self, x: th.Tensor, y: th.Tensor) -> th.Tensor:
        y_emb = self.__y_emb(y)
        out: th.Tensor = self.__encoder(x) + y_emb

        return out
