# -*- coding: utf-8 -*-
import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


class CosineScheduleWarmup(LambdaLR):
    def __init__(
        self,
        optimizer: Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        min_lr: float,
    ) -> None:
        def lr_lambda(current_step: int) -> float:
            if current_step < num_warmup_steps:
                return max(
                    float(current_step) / float(max(1, num_warmup_steps)),
                    min_lr,
                )

            progress = float(current_step - num_warmup_steps) / float(
                max(1, num_training_steps - num_warmup_steps)
            )

            return max(min_lr, 0.5 * (1.0 + math.cos(math.pi * progress)))

        super().__init__(optimizer, lr_lambda)
