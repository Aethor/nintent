import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


from typing import List
import argparse
import random

import torch
from tqdm import tqdm

from tree import IntentTree, Intent
from datas import Dataset
from config import Config
from model import TreeMaker


def score(
    model: TreeMaker,
    trees: List[IntentTree],
    device: torch.device,
    verbose: bool = False,
):
    model.eval()
    model.to(device)

    with torch.no_grad():
        pred_trees = [
            model.make_tree(tree.tokens, device, Intent, tree.span_coords)
            for tree in tqdm(trees)
        ]
    if verbose:
        for pred_tree in random.choices(pred_trees, k=10):
            tqdm.write(str(pred_tree))

    exact_accuracy = IntentTree.exact_accuracy_metric(pred_trees, trees)
    labeled_precision, labeled_recall, labeled_f1 = IntentTree.labeled_bracketed_metric(
        pred_trees, trees
    )

    return exact_accuracy, labeled_precision, labeled_recall, labeled_f1


if __name__ == "__main__":

    config = Config("./configs/default-score-config.json")
    arg_parser = argparse.ArgumentParser(argparse.ArgumentDefaultsHelpFormatter)
    arg_parser.add_argument(
        "-tdur",
        "--test-datas-usage-ratio",
        type=float,
        default=config["test_datas_usage_ratio"],
        help="test datas usage ratio (between 0 and 1)",
    )
    arg_parser.add_argument(
        "-v",
        "--verbose",
        type=bool,
        default=config["verbose"],
        help="wether the score function should be verbose or not",
    )
    arg_parser.add_argument(
        "-mp",
        "--model-path",
        type=str,
        default=config["model_path"],
        help="path from where the model will be loaded",
    )
    arg_parser.add_argument(
        "-cf",
        "--config-file",
        type=str,
        default=None,
        help="Config file overriding default-score-config.json",
    )
    args = arg_parser.parse_args()
    if args.config_file:
        config.load_from_file_(args.config_file)
    else:
        config.update_(vars(args))

    print("[info] loading test datas")
    test_dataset = Dataset.from_file(
        "./datas/test.tsv", config["test_datas_usage_ratio"]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TreeMaker()
    model.load_state_dict(
        torch.load(config["model_path"], map_location=device), strict=False
    )

    exact_accuracy, labeled_precision, labeled_recall, labeled_f1 = score(
        model, test_dataset.trees, device, verbose=config["verbose"]
    )

    print("train exact accuracy : {:10.4f}".format(exact_accuracy))
    print("train labeled precision : {:10.4f}".format(labeled_precision))
    print("train labeled recall : {:10.4f}".format(labeled_recall))
    print("train labeled f1 : {:10.4f}".format(labeled_f1))
