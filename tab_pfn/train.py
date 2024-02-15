# -*- coding: utf-8 -*-
from os import mkdir
from os.path import exists, isdir, join

import mlflow
import torch as th
from torch.nn import functional as F
from tqdm import tqdm

from .metrics import ConfusionMeter, LossMeter
from .options import ModelOptions, TrainOptions


def train(model_options: ModelOptions, train_options: TrainOptions) -> None:
    with mlflow.start_run(run_name=train_options.run_name):
        if not exists(train_options.output_folder):
            mkdir(train_options.output_folder)
        elif not isdir(train_options.output_folder):
            raise NotADirectoryError(train_options.output_folder)

        tab_pfn = model_options.get_tab_pfn()

        print(f"parameters : {tab_pfn.count_parameters()}")

        if model_options.cuda:
            device = th.device("cuda")
            tab_pfn.to(device)
        else:
            device = th.device("cpu")
            tab_pfn.to(device)

        optim = th.optim.Adam(
            tab_pfn.parameters(), lr=train_options.learning_rate
        )

        # total_steps = int(
        #     train_options.n_datasets
        #     * (train_options.n_data * (1.0 - train_options.data_ratio))
        # )
        # warmup_proportion = 0.1

        # scheduler = warmup_cosine_scheduler(
        #     optim, warmup_proportion, total_steps
        # )

        mlflow.log_params(
            {
                "model_options": model_options,
                "train_options": train_options,
            }
        )

        save_every = 256

        window_size = 64
        loss_meter = LossMeter(window_size)
        confusion_meter = ConfusionMeter(model_options.max_class, window_size)

        tqdm_bar = tqdm(range(train_options.n_datasets))
        for k in tqdm_bar:

            scm = model_options.get_scm()
            scm.to(device)

            x, y = scm(train_options.n_data)

            train_index = int(train_options.data_ratio * train_options.n_data)

            x_train, y_train = x[:train_index], y[:train_index]
            x_test, y_test = x[train_index:], y[train_index:]

            out = tab_pfn(x_train, y_train, x_test)
            loss = F.cross_entropy(out, y_test, reduction="mean")

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            # scheduler.step()

            loss_meter.add(loss.item())
            confusion_meter.add(out, y_test)

            precision = confusion_meter.precision().mean().item()
            recall = confusion_meter.recall().mean().item()

            tqdm_bar.set_description(
                f"dataset [{k} / {train_options.n_datasets}] - "
                f"(n_class={scm.nb_class}) : "
                f"loss = {loss_meter.loss():.4f}, "
                f"precision = {precision:.4f}, "
                f"recall = {recall:.4f}, "
                f"grad_norm = {tab_pfn.grad_norm():.4f}, "
                f"lr = {optim.param_groups[0]['lr']:.8f}"
            )

            mlflow.log_metrics(
                {
                    "loss": loss.item(),
                    "recall": recall,
                    "precision": precision,
                },
                step=k,
            )

            if k % save_every == save_every - 1:
                th.save(
                    tab_pfn.state_dict(),
                    join(train_options.output_folder, f"model_{k}.pt"),
                )
                th.save(
                    optim.state_dict(),
                    join(train_options.output_folder, f"optim_{k}.pt"),
                )
