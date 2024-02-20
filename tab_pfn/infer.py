# -*- coding: utf-8 -*-
from os import mkdir
from os.path import exists, isdir

import torch as th
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from .data import CsvDataset
from .metrics import AccuracyMeter, ConfusionMeter
from .networks import pad_features
from .options import InferOptions, ModelOptions


def infer(model_options: ModelOptions, infer_options: InferOptions) -> None:
    if not exists(infer_options.output_folder):
        mkdir(infer_options.output_folder)
    elif not isdir(infer_options.output_folder):
        raise NotADirectoryError(infer_options.output_folder)

    print(f'Will infer from "{infer_options.csv_path}"')

    dataset = CsvDataset(
        infer_options.csv_path,
        infer_options.csv_sep,
        target_column=infer_options.class_col,
    )

    tab_pfn = model_options.get_tab_pfn()
    tab_pfn.load_state_dict(th.load(infer_options.state_dict))

    device = th.device("cuda") if model_options.cuda else th.device("cpu")

    tab_pfn.to(device)
    tab_pfn.eval()

    train_dataset, test_dataset = random_split(
        dataset, [infer_options.train_ratio, 1.0 - infer_options.train_ratio]
    )

    features_randperm = th.randperm(model_options.max_features)

    x_tmp, y_tmp = zip(*[train_dataset[i] for i in range(len(train_dataset))])

    x_train = pad_features(
        th.stack(x_tmp, dim=0).to(device), model_options.max_features
    )[None, :, features_randperm]
    y_train = th.stack(y_tmp, dim=0).to(device)[None]

    data_loader = DataLoader(test_dataset, batch_size=128)

    acc_meter = AccuracyMeter(None)
    conf_meter = ConfusionMeter(dataset.nb_classes, None)

    for x, y in tqdm(data_loader):

        x = pad_features(x.to(device), model_options.max_features)[
            None, :, features_randperm
        ]
        y = y.to(device)

        with th.no_grad():
            out = tab_pfn(x_train, y_train, x)[0]
            acc_meter.add(out, y)
            conf_meter.add(out, y)

    print(f"accuracy : {acc_meter.accuracy()}")
    print(f"precisions = {conf_meter.precision().numpy().tolist()}")
    print(f"recalls = {conf_meter.recall().numpy().tolist()}")
    print(f"confusion_matrix :\n{conf_meter.conf_mat().numpy()}")

    conf_meter.save_conf_matrix(-1, infer_options.output_folder)
