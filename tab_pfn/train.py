# -*- coding: utf-8 -*-
from os import mkdir
from os.path import exists, isdir, join

import torch as th
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .options import ModelOptions, TrainOptions


def train(model_options: ModelOptions, train_options: TrainOptions) -> None:

    if not exists(train_options.output_folder):
        mkdir(train_options.output_folder)
    elif not isdir(train_options.output_folder):
        raise NotADirectoryError(train_options.output_folder)

    tab_pfn = model_options.get_tab_pfn()

    if model_options.cuda:
        device = th.device("cuda")
        tab_pfn.to(device)
    else:
        device = th.device("cpu")
        tab_pfn.to(device)

    optim = th.optim.Adam(tab_pfn.parameters(), lr=train_options.learning_rate)

    for k in range(train_options.n_datasets):

        scm = model_options.get_scm()
        scm.to(device)

        x, y = scm(train_options.n_data)

        train_index = int(train_options.data_ratio * train_options.n_data)

        x_train, y_train = x[:train_index], y[:train_index]
        x_test, y_test = x[train_index:], y[train_index:]

        data_loader = DataLoader(
            TensorDataset(x_test, y_test), batch_size=train_options.batch_size
        )
        tqdm_bar = tqdm(data_loader)

        for x_t, y_t in tqdm_bar:

            out = tab_pfn(x_train, y_train, x_t)
            loss = F.cross_entropy(out, y_t, reduction="mean")

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()

            tqdm_bar.set_description(
                f"dataset : {k}, loss = {loss.item():.4f}"
            )

        th.save(
            tab_pfn.state_dict(),
            join(train_options.output_folder, f"model_{k}.pt"),
        )
        th.save(
            optim.state_dict(),
            join(train_options.output_folder, f"optim_{k}.pt"),
        )
