# -*- coding: utf-8 -*-
from .options import InferOptions, ModelOptions


def infer(model_options: ModelOptions, infer_options: InferOptions) -> None:
    print(model_options)
    print(infer_options)
