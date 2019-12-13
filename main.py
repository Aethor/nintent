import argparse

import torch

from tree import IntentTree


if __name__ == "__main__":
    trees = []

    with open("./datas/train.tsv") as train_file:
        for line in train_file:
            trees.append(IntentTree.from_str(line.split("\t")[-1]))
