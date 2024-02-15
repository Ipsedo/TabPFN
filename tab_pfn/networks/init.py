# -*- coding: utf-8 -*-
from random import uniform

from torch import nn


def init_scm(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, std=uniform(1e-3, 1e-1))
