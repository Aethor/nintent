import argparse

import torch

from tree import IntentTree
from config import Config


if __name__ == "__main__":
    config = Config("./datas/default-config.json")
    arg_parser = argparse.ArgumentParser(argparse.ArgumentDefaultsHelpFormatter)
    arg_parser.add_argument(
        "-en",
        "--epochs-nb",
        type=int,
        default=config["epochs_nb"],
        help="Number of epochs",
    )
    args = arg_parser.parse_args()
    if args.config_file:
        config.load_from_file_(args.config_file)
    else:
        config.update_(vars(args))

    trees = []
    with open("./datas/train.tsv") as train_file:
        for line in train_file:
            trees.append(IntentTree.from_str(line.split("\t")[-1]))
