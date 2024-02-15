# -*- coding: utf-8 -*-
import argparse

from .options import ModelOptions, TrainOptions
from .train import train


def main() -> None:
    parser = argparse.ArgumentParser("tab_pfn main")

    parser.add_argument("--max-features", type=int, default=100)
    parser.add_argument("--max-class", type=int, default=10)
    parser.add_argument("--encoder-dim", type=int, default=256)
    parser.add_argument("--ppd-dim", type=int, default=512)
    parser.add_argument("--ppd-hidden-dim", type=int, default=1024)
    parser.add_argument("--nheads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=12)
    parser.add_argument("--cuda", action="store_true")

    sub_parser = parser.add_subparsers(dest="mode")

    train_parser = sub_parser.add_parser("train")
    train_parser.add_argument("run_name", type=str)
    train_parser.add_argument("output_folder", type=str)
    train_parser.add_argument("--learning-rate", type=float, default=1e-4)
    train_parser.add_argument("--datasets", type=int, default=2**20)
    train_parser.add_argument("--data", type=int, default=2**11)
    train_parser.add_argument("--data-ratio", type=float, default=0.75)

    args = parser.parse_args()

    model_options = ModelOptions(
        args.max_features,
        args.max_class,
        args.encoder_dim,
        args.ppd_dim,
        args.ppd_hidden_dim,
        args.nheads,
        args.num_layers,
        args.cuda,
    )

    if args.mode == "train":

        train_options = TrainOptions(
            args.run_name,
            args.learning_rate,
            args.datasets,
            args.data,
            args.data_ratio,
            args.output_folder,
        )

        train(model_options, train_options)
    else:
        parser.error(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
