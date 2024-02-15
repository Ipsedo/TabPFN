# -*- coding: utf-8 -*-
from random import choice, randint, uniform
from typing import Callable, List, Tuple

import torch as th
from torch import nn
from torch.distributions import Normal
from torch.nn import functional as F

from .functions import beta, pad_features, tnlu, tnlu_int


def _init_scm(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        module.weight.data.copy_(tnlu(module.weight.size(), 1e-2, 10, 0.0))


class SCM(nn.Module):
    def __init__(
        self,
        n_features: int,
        class_bounds: Tuple[int, int],
    ) -> None:
        super().__init__()

        # setup MLP
        n_layer: int = tnlu_int(2, 6, 2)
        hidden_size: int = tnlu_int(5, 130, 4)

        self.__mlp = nn.ModuleList(
            nn.Linear(hidden_size, hidden_size, bias=False)
            for _ in range(n_layer)
        )

        # create node indexes list
        node_list = th.tensor(
            [[i, j] for i in range(n_layer) for j in range(hidden_size)]
        )
        node_list = node_list[th.randperm(node_list.size(0))]

        # adapt features number
        self.__wanted_n_features = n_features
        self.__n_features = (
            n_features
            if n_features + 1 <= node_list.size(0)
            else node_list.size(0) - 1
        )

        # Z : used for features and label
        self.__zx_nodes_idx = node_list[: self.__n_features].split(1, dim=1)

        self.__zy_node_idx = node_list[self.__n_features]
        self.__y_class_intervals: List[float] = []

        self.__nb_class = randint(class_bounds[0], class_bounds[1])

        # E : set from which we drop neurons
        e_node_list = node_list[self.__n_features + 1 :]

        # drop neurons
        a = th.tensor(uniform(0.1, 5))
        b = th.tensor(uniform(0.1, 5))
        drop_neuron_proba = 0.9 * beta(a, b)

        mask_index_i, mask_index_j = th.split(e_node_list, 1, dim=1)

        mask = th.ones(n_layer, hidden_size)
        mask[mask_index_i.squeeze(-1), mask_index_j.squeeze(-1)] = (
            drop_neuron_proba < th.rand(e_node_list.size(0))
        ).to(th.float)

        self.register_buffer("_mask", mask)

        # activation functions
        act_fn: List[Callable[[th.Tensor], th.Tensor]] = [
            F.relu,
            F.tanh,
            F.leaky_relu,
            F.elu,
            lambda t: t,  # identity
        ]

        self.__act: List[Callable[[th.Tensor], th.Tensor]] = [
            choice(act_fn) for _ in range(n_layer)
        ]

        # noise params for SCM (epsilon)

        # cov_mat = th.rand(hidden_size, hidden_size)
        # cov_mat = 0.5 * (cov_mat + cov_mat.t())
        # cov_mat = cov_mat + hidden_size * th.eye(hidden_size)
        # cov_mat = cov_mat @ cov_mat.t()

        sigma = tnlu((hidden_size,), 1e-4, 0.3, 1e-8)
        loc = th.randn(hidden_size) * uniform(1e-4, 1e-1)

        self.register_buffer("_sigma", sigma)
        self.register_buffer("_loc", loc)

        self.apply(_init_scm)

    @th.no_grad()
    def forward(self, batch_size: int) -> Tuple[th.Tensor, th.Tensor]:
        distribution = Normal(self._loc, self._sigma)

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

        # select features and label
        x = outs_stacked[:, *self.__zx_nodes_idx].squeeze(-1)
        y = outs_stacked[:, *self.__zy_node_idx].squeeze(-1)

        # create class interval if first time
        if len(self.__y_class_intervals) == 0:
            self.__y_class_intervals = sorted(
                [
                    uniform(y.min().item(), y.max().item())
                    for _ in range(self.__nb_class - 1)
                ]
            )

        y_class = self.__split_tensor_based_on_intervals(
            y, self.__y_class_intervals
        )

        return self.__pad_features(x), y_class

    def __pad_features(self, x_to_pad: th.Tensor) -> th.Tensor:
        return pad_features(x_to_pad, self.__wanted_n_features)

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

        permutation = th.arange(0, len(intervals) + 1, device=y_scalar.device)
        permutation = permutation[th.randperm(permutation.size(0))]

        indices = permutation[indices]

        return indices
