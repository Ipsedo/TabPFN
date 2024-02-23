# -*- coding: utf-8 -*-
import argparse

from .infer import infer
from .options import InferOptions, ModelOptions, TrainOptions
from .train import train


def main() -> None:
    parser = argparse.ArgumentParser("tab_pfn main")

    parser.add_argument("--max-features", type=int, default=100)
    parser.add_argument("--max-class", type=int, default=10)
    parser.add_argument("--encoder-dim", type=int, default=128)
    parser.add_argument("--ppd-dim", type=int, default=256)
    parser.add_argument("--ppd-hidden-dim", type=int, default=512)
    parser.add_argument("--nheads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--cuda", action="store_true")

    sub_parser = parser.add_subparsers(dest="mode")

    train_parser = sub_parser.add_parser("train")
    train_parser.add_argument("run_name", type=str)
    train_parser.add_argument("output_folder", type=str)
    train_parser.add_argument("--learning-rate", type=float, default=1e-4)
    train_parser.add_argument("--steps", type=int, default=400000)
    train_parser.add_argument("--batch-size", type=int, default=8)
    train_parser.add_argument("--data", type=int, default=2048)
    train_parser.add_argument(
        "--data-ratios", type=float, nargs=2, default=(0.5, 0.75)
    )
    train_parser.add_argument("--warmup-steps", type=int, default=50000)
    train_parser.add_argument("--cosine-min-lr", type=float, default=1e-7)
    train_parser.add_argument("--save-every", type=int, default=1024)
    train_parser.add_argument("--metric-window-size", type=int, default=64)

    infer_parser = sub_parser.add_parser("infer")
    infer_parser.add_argument("csv_path", type=str)
    infer_parser.add_argument("state_dict", type=str)
    infer_parser.add_argument("output_folder", type=str)
    infer_parser.add_argument("--class-col", type=str, required=True)
    infer_parser.add_argument("--csv-sep", type=str, default=",")
    infer_parser.add_argument("--train-ratio", type=float, default=0.5)

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
            args.steps,
            args.batch_size,
            args.data,
            args.data_ratios,
            args.save_every,
            args.metric_window_size,
            args.warmup_steps,
            args.cosine_min_lr,
            args.output_folder,
        )

        train(model_options, train_options)
    elif args.mode == "infer":
        infer_options = InferOptions(
            args.csv_path,
            args.class_col,
            args.csv_sep,
            args.train_ratio,
            args.state_dict,
            args.output_folder,
        )

        infer(model_options, infer_options)
    else:
        parser.error(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
