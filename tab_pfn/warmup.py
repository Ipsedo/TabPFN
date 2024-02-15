# -*- coding: utf-8 -*-
import math

import torch as th
from torch.optim.lr_scheduler import LambdaLR


def _linear_warmup(
    step: int, total_steps: int, warmup_proportion: float
) -> float:
    return min(1.0, step / (total_steps * warmup_proportion))


def _cosine_annealing(
    step: int, total_steps: int, warmup_proportion: float
) -> float:
    if step < total_steps * warmup_proportion:
        return 1.0

    progress = (step - total_steps * warmup_proportion) / (
        total_steps * (1 - warmup_proportion)
    )

    return 0.5 * (1 + math.cos(math.pi * progress))


def warmup_cosine_scheduler(
    optimizer: th.optim.Optimizer, warmup_proportion: float, total_steps: int
) -> LambdaLR:
    def lr_lambda(step: int) -> float:
        return _linear_warmup(
            step, total_steps, warmup_proportion
        ) * _cosine_annealing(step, total_steps, warmup_proportion)

    return LambdaLR(optimizer, lr_lambda)
