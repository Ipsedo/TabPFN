# -*- coding: utf-8 -*-

import torch as th
from torch import nn


def _init_encoder(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.xavier_normal_(module.weight, gain=1e-1)

        if module.bias is not None:
            nn.init.normal_(module.bias, std=1e-1)

    elif isinstance(module, nn.LayerNorm):
        if module.elementwise_affine:
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0.0)


class DataEncoder(nn.Sequential):
    def __init__(
        self, x_max_dim: int, hidden_dim: int, output_dim: int
    ) -> None:
        super().__init__(
            nn.Linear(x_max_dim, hidden_dim),
            nn.Mish(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
            nn.Mish(),
            nn.LayerNorm(output_dim),
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

        self.__y_emb = nn.Embedding(nb_class_max, y_emb_dim)
        self.__encoder = DataEncoder(
            x_max_dim + y_emb_dim, hidden_dim, output_dim
        )

    def forward(self, x: th.Tensor, y: th.Tensor) -> th.Tensor:
        y_emb = self.__y_emb(y)

        out = th.cat([x, y_emb], dim=2)
        out = self.__encoder(out)

        return out
