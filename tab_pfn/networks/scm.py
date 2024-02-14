# -*- coding: utf-8 -*-
from random import choice, randint
from typing import Callable, List, Tuple

import torch as th
from torch import nn
from torch.distributions import MultivariateNormal
from torch.nn import functional as F


class SCM(nn.Module):
    def __init__(
        self,
        drop_neuron_proba: float,
        n_features: int,
        layer_bounds: Tuple[int, int] = (4, 16),
        node_bounds: Tuple[int, int] = (16, 64),
    ) -> None:
        super().__init__()

        n_layer = randint(layer_bounds[0], layer_bounds[1])
        hidden_size = randint(node_bounds[0], node_bounds[1])

        self.__mlp = nn.ModuleList(
            nn.Linear(hidden_size, hidden_size, bias=False)
            for i in range(n_layer)
        )

        self.__mask = th.ge(th.rand(n_layer, hidden_size), drop_neuron_proba)

        act_fn: List[Callable[[th.Tensor], th.Tensor]] = [
            F.relu,
            F.tanh,
            F.leaky_relu,
            F.elu,
        ]

        self.__act: List[Callable[[th.Tensor], th.Tensor]] = [
            choice(act_fn) for _ in range(n_layer)
        ]

        non_masked_nodes = self.__mask.nonzero()
        non_masked_nodes = non_masked_nodes[
            th.randperm(non_masked_nodes.size(0))
        ]

        self.__x_idx = non_masked_nodes[:n_features].split(1, dim=1)

        cov_mat = th.randn(hidden_size, hidden_size)
        cov_mat = th.matmul(cov_mat.transpose(0, 1), cov_mat)

        loc = th.randn(hidden_size)

        self.__distribution = MultivariateNormal(loc, cov_mat)

        self.__y_idx = non_masked_nodes[n_features + 1]

    @th.no_grad()
    def forward(self, batch_size: int) -> Tuple[th.Tensor, th.Tensor]:
        epsilon_size = th.Size([batch_size])

        out = self.__distribution.sample(epsilon_size)
        outs = []

        for layer, act, mask in zip(self.__mlp, self.__act, self.__mask):
            out = layer(out) + self.__distribution.sample(epsilon_size)
            out = act(out)
            out = out * mask[None, :]
            outs.append(out)

        # stack layers output
        # (batch, layer, hidden_features)
        outs_stacked = th.stack(outs, dim=1)

        # select features
        x = outs_stacked[:, *self.__x_idx].squeeze(-1)
        # select label
        y = outs_stacked[:, *self.__y_idx].squeeze(-1)

        return x, y
