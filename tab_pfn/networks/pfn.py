# -*- coding: utf-8 -*-
import torch as th
from torch import nn
from torch.nn import functional as F


class PFN(nn.Module):
    def __init__(self, model_dim: int, hidden_dim: int, nb_class: int) -> None:
        super().__init__()

        self.__trf = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                model_dim, 4, hidden_dim, activation=F.gelu, batch_first=True
            ),
            6,
        )

        self.__to_class = nn.Linear(model_dim, nb_class)

    @staticmethod
    def __get_mask(
        x_train: th.Tensor, x_test: th.Tensor, device: str
    ) -> th.Tensor:
        mask = th.eye(x_train.size(0) + x_test.size(0), device=device)

        mask[:, : x_train.size(0)] = 1

        return mask

    def forward(self, x_train: th.Tensor, x_test: th.Tensor) -> th.Tensor:
        device = "cuda" if next(self.parameters()).is_cuda else "cpu"

        src_mask = self.__get_mask(x_train, x_test, device)

        enc_input = th.cat([x_train, x_test], dim=0)

        out: th.Tensor = self.__trf(enc_input, mask=src_mask)[
            x_train.size(0) :, :
        ]
        out = self.__to_class(out)

        return out
