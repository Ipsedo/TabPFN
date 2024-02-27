# -*- coding: utf-8 -*-
from random import choice, randint, sample, uniform
from typing import Callable, List, Tuple

import torch as th
from torch import nn
from torch.distributions import Beta, Normal
from torch.nn import functional as F

from .functions import pad_features, tnlu, tnlu_float, tnlu_int


def _init_scm(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, std=tnlu_float(1e-2, 10, 0.0))


class RandomActivation(nn.Module):
    def __init__(self, size: int) -> None:
        super().__init__()

        act_fn: List[Callable[[th.Tensor], th.Tensor]] = [
            F.relu,
            F.tanh,
            F.leaky_relu,
            F.elu,
            lambda t: t,  # identity
        ]

        self.__act: List[Callable[[th.Tensor], th.Tensor]] = [
            choice(act_fn) for _ in range(size)
        ]

    def forward(self, x: th.Tensor) -> th.Tensor:
        return th.stack(
            [act(d) for d, act in zip(x.unbind(dim=1), self.__act)], dim=1
        )


class SCM(nn.Module):
    def __init__(
        self,
        n_features: int,
        class_bounds: Tuple[int, int],
        shuffle_class: bool = True,
    ) -> None:
        super().__init__()

        # setup MLP
        n_layer: int = tnlu_int(1, 6, 2)
        hidden_sizes: List[int] = [tnlu_int(5, 130, 4) for _ in range(n_layer)]
        self.__max_hidden_size = max(hidden_sizes)

        self.__mlp = nn.ModuleList(
            nn.Linear(
                hidden_sizes[i - 1] if i > 0 else hidden_sizes[0],
                hidden_sizes[i],
                bias=False,
            )
            for i in range(len(hidden_sizes))
        )

        # create node indexes list
        node_list = th.tensor(
            [[i, j] for i, h in enumerate(hidden_sizes) for j in range(h)]
        )
        node_list = node_list[th.randperm(node_list.size(0))]

        # adapt features number
        self.__wanted_n_features = n_features
        self.__n_features = min(
            int(uniform(2, self.__wanted_n_features)), node_list.size(0) - 1
        )

        # Z : used for features and label
        self.__zx_nodes_idx = node_list[: self.__n_features].split(1, dim=1)
        self.__zx_rand_perm = th.randperm(self.__wanted_n_features)

        self.__zy_node_idx = node_list[self.__n_features]
        self.__y_class_intervals: List[float] = []

        self.__max_nb_class = class_bounds[1]
        self.__nb_class = randint(class_bounds[0], class_bounds[1])
        self.__zy_rand_perm = th.randperm(self.__max_nb_class)
        self.__shuffle_class = shuffle_class

        # drop neuron connexions
        a = th.tensor(uniform(0.1, 5))
        b = th.tensor(uniform(0.1, 5))
        drop_neuron_proba = 0.9 * Beta(a, b).sample(th.Size((1,))).item()

        for lin in self.__mlp:
            lin.weight.data.mul_(
                th.rand(*lin.weight.size()) > drop_neuron_proba
            )

        # activation functions
        self.__act = nn.ModuleList(RandomActivation(h) for h in hidden_sizes)

        # noise params for SCM (epsilon)

        # cov_mat = th.rand(hidden_size, hidden_size)
        # cov_mat = 0.5 * (cov_mat + cov_mat.t())
        # cov_mat = cov_mat + hidden_size * th.eye(hidden_size)
        # cov_mat = cov_mat @ cov_mat.t()

        for i, h in enumerate(hidden_sizes):
            self.register_buffer(f"_noise_mean_{i}", th.zeros(h))
            self.register_buffer(
                f"_noise_sigma_{i}",
                tnlu((h,), 1e-4, 0.3, 1e-8),
            )

        self.register_buffer("_cause_mean", th.zeros(hidden_sizes[0]))
        self.register_buffer("_cause_sigma", th.ones(hidden_sizes[0]))

        self.apply(_init_scm)

    @th.no_grad()
    def forward(self, batch_size: int) -> Tuple[th.Tensor, th.Tensor]:
        epsilon_size = th.Size([batch_size])

        out = Normal(self._cause_mean, self._cause_sigma).sample(epsilon_size)
        outs = []

        for i, (layer, act) in enumerate(zip(self.__mlp, self.__act)):
            loc = self.get_buffer(f"_noise_mean_{i}")
            sig = self.get_buffer(f"_noise_sigma_{i}")

            dist = Normal(loc, sig)

            out = act(layer(out) + dist.sample(epsilon_size))
            outs.append(F.pad(out, (0, self.__max_hidden_size - out.size(1))))

        # stack layers output
        # (batch, layer, hidden_features)
        outs_stacked = th.stack(outs, dim=1)

        # select features and label
        x = outs_stacked[:, *self.__zx_nodes_idx].squeeze(-1)
        y = outs_stacked[:, *self.__zy_node_idx].squeeze(-1)

        # create class interval if first time
        if len(self.__y_class_intervals) == 0:
            self.__y_class_intervals = sorted(
                sample(y.tolist(), self.__nb_class - 1)
            )

        return (
            self.__pad_shuffle_features(x),
            self.__scalar_to_shuffled_class(y),
        )

    def __pad_shuffle_features(self, x_to_pad: th.Tensor) -> th.Tensor:
        return pad_features(x_to_pad, self.__wanted_n_features)[
            :, self.__zx_rand_perm
        ]

    @property
    def nb_class(self) -> int:
        return self.__nb_class

    def get_class_index(self) -> th.Tensor:
        return self.__zy_rand_perm[
            th.arange(0, self.__max_nb_class)[: self.__nb_class]
        ]

    def get_class_mask(self) -> th.Tensor:
        mask = th.zeros(
            self.__max_nb_class,
            device="cuda" if next(self.buffers()).is_cuda else "cpu",
        )
        mask[self.get_class_index()] = 1.0
        return mask

    def inverse_class_permutations(self) -> th.Tensor:
        permutations = th.arange(
            0,
            self.__max_nb_class,
            device="cuda" if next(self.buffers()).is_cuda else "cpu",
        )[self.__zy_rand_perm]
        return th.argsort(permutations)

    def __scalar_to_shuffled_class(self, y_scalar: th.Tensor) -> th.Tensor:
        indices = th.zeros_like(
            y_scalar, dtype=th.long, device=y_scalar.device
        )
        for interval in self.__y_class_intervals:
            indices += y_scalar > interval

        if self.__shuffle_class:
            permutations = th.arange(
                0, self.__max_nb_class, device=y_scalar.device
            )[self.__zy_rand_perm]

            indices = permutations[indices]

        return indices
