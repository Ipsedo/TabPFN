# -*- coding: utf-8 -*-
from os import mkdir
from os.path import exists, isdir, join
from random import uniform

import mlflow
import torch as th
from torch.nn import functional as F
from tqdm import tqdm

from .metrics import AccuracyMeter, ConfusionMeter, LossMeter
from .networks import get_cosine_schedule_with_warmup
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

        scheduler = get_cosine_schedule_with_warmup(
            optim,
            train_options.warmup_steps,
            train_options.steps,
            train_options.cosine_min_lr,
        )

        mlflow.log_params(
            {
                "model_options": model_options.to_dict(),
                "train_options": train_options.to_dict(),
            }
        )

        loss_meter = LossMeter(train_options.metric_window_size)
        confusion_meter = ConfusionMeter(
            model_options.max_class, train_options.metric_window_size
        )
        accuracy_meter = AccuracyMeter(train_options.metric_window_size)

        tqdm_bar = tqdm(range(train_options.steps))

        for s in tqdm_bar:

            x_batch, y_batch = zip(
                *[
                    model_options.get_scm()(train_options.n_data)
                    for _ in range(train_options.batch_size)
                ]
            )

            x = th.stack(x_batch, dim=0).to(device)
            y = th.stack(y_batch, dim=0).to(device)

            # train_index = int(train_options.data_ratio * train_options.n_data)
            train_index = int(
                uniform(
                    train_options.data_ratios[0], train_options.data_ratios[1]
                )
                * train_options.n_data
            )

            x_train, y_train = x[:, :train_index], y[:, :train_index]
            x_test, y_test = x[:, train_index:], y[:, train_index:]

            out = tab_pfn(x_train, y_train, x_test)
            loss = F.cross_entropy(
                out.permute(0, 2, 1), y_test, reduction="mean"
            )

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            scheduler.step()

            loss_meter.add(loss.item())
            confusion_meter.add(out.flatten(0, 1), y_test.flatten(0, 1))
            accuracy_meter.add(out.flatten(0, 1), y_test.flatten(0, 1))

            precision = confusion_meter.precision().mean().item()
            recall = confusion_meter.recall().mean().item()
            accuracy = accuracy_meter.accuracy()

            grad_norm = tab_pfn.grad_norm()

            tqdm_bar.set_description(
                f"loss = {loss_meter.loss():.4f}, "
                f"precision = {precision:.4f}, "
                f"recall = {recall:.4f}, "
                f"grad_norm = {grad_norm:.4f}, "
                f"accuracy = {accuracy:.4f}, "
                f"lr = {optim.param_groups[0]['lr']:.10f}"
            )

            mlflow.log_metrics(
                {
                    "loss": loss.item(),
                    "recall": recall,
                    "precision": precision,
                    "grad_norm": grad_norm,
                    "accuracy": accuracy,
                    "lr": optim.param_groups[0]["lr"],
                },
                step=s,
            )

            if s % train_options.save_every == train_options.save_every - 1:
                th.save(
                    tab_pfn.state_dict(),
                    join(train_options.output_folder, f"model_{s}.pt"),
                )
                th.save(
                    optim.state_dict(),
                    join(train_options.output_folder, f"optim_{s}.pt"),
                )

                confusion_meter.save_conf_matrix(
                    s, train_options.output_folder
                )
