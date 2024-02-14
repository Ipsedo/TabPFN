# -*- coding: utf-8 -*-
from random import choice, randint, uniform
from typing import Callable, List, Tuple

import torch as th
from torch import nn
from torch.distributions import MultivariateNormal, Wishart
from torch.nn import functional as F


class SCM(nn.Module):
    def __init__(
        self,
        drop_neuron_proba: float,
        n_features: int,
        layer_bounds: Tuple[int, int] = (4, 8),
        node_bounds: Tuple[int, int] = (64, 128),
        class_bounds: Tuple[int, int] = (2, 16),
    ) -> None:
        super().__init__()

        n_layer = randint(layer_bounds[0], layer_bounds[1])
        hidden_size = randint(node_bounds[0], node_bounds[1])

        self.__mlp = nn.ModuleList(
            nn.Linear(hidden_size, hidden_size, bias=False)
            for i in range(n_layer)
        )

        mask = th.ge(th.rand(n_layer, hidden_size), drop_neuron_proba)
        self.register_buffer("_mask", mask)

        act_fn: List[Callable[[th.Tensor], th.Tensor]] = [
            F.relu,
            F.tanh,
            F.leaky_relu,
            F.elu,
        ]

        self.__act: List[Callable[[th.Tensor], th.Tensor]] = [
            choice(act_fn) for _ in range(n_layer)
        ]

        non_masked_nodes = self._mask.nonzero()
        non_masked_nodes = non_masked_nodes[
            th.randperm(non_masked_nodes.size(0))
        ]

        self.__x_idx = non_masked_nodes[:n_features].split(1, dim=1)

        # cov_mat = th.randn(hidden_size, hidden_size)
        # diag = th.exp(th.abs(th.randn(hidden_size)))
        # cov_mat = th.mm(th.diag(diag), cov_mat)
        # cov_mat = cov_mat @ cov_mat.t()
        cov_mat = Wishart(
            th.tensor(randint(hidden_size, hidden_size * 2)),
            th.eye(hidden_size),
        ).sample()

        loc = th.randn(hidden_size)

        self.register_buffer("_cov_mat", cov_mat)
        self.register_buffer("_loc", loc)

        self.__y_idx = non_masked_nodes[n_features + 1]

        self.__nb_class = randint(class_bounds[0], class_bounds[1])

        self.__y_class_intervals: List[float] = []

    @th.no_grad()
    def forward(self, batch_size: int) -> Tuple[th.Tensor, th.Tensor]:
        distribution = MultivariateNormal(self._loc, self._cov_mat)

        epsilon_size = th.Size([batch_size])

        out = distribution.sample(epsilon_size)
        outs = []

        for layer, act, mask in zip(self.__mlp, self.__act, self._mask):
            out = layer(out) + distribution.sample(epsilon_size)
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

        if len(self.__y_class_intervals) == 0:
            self.__y_class_intervals = sorted(
                [
                    uniform(y.min().item(), y.max().item())
                    for _ in range(self.__nb_class)
                ]
            )

        y_class = self.__split_tensor_based_on_intervals(
            y, self.__y_class_intervals
        )

        return x, y_class

    @property
    def nb_class(self) -> int:
        return self.__nb_class

    @staticmethod
    def __split_tensor_based_on_intervals(
        y_scalar: th.Tensor, intervals: List[float]
    ) -> th.Tensor:
        indices = th.zeros_like(
            y_scalar, dtype=th.long, device=y_scalar.device
        )
        for interval in intervals:
            indices += y_scalar > interval
        return indices
