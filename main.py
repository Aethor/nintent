import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


import argparse
import random
from typing import Optional, List

import torch
from torch.optim.optimizer import Optimizer
from torch.nn.modules.loss import _Loss
from torch.optim.lr_scheduler import _LRScheduler
from transformers import BertTokenizer
from tqdm import tqdm

from tree import IntentTree, Intent
from datas import Dataset
from config import Config
from model import TreeScorer


def train_(
    model: TreeScorer,
    dataset: Dataset,
    device: torch.device,
    optimizer: Optimizer,
    scheduler: Optional[_LRScheduler],
    epochs_nb: int,
    batch_size: int,
):
    model.to(device)

    for epoch in range(epochs_nb):

        mean_loss_list = []

        batches_progress = tqdm(
            dataset.batches(batch_size, device), total=dataset.batches_nb(batch_size)
        )

        def update_progress_bar(epoch: int, loss: float):
            if len(mean_loss_list) > 0:
                batches_progress.set_description(
                    "[epoch:{}][loss:{:2f}]".format(epoch + 1, loss)
                )

        # TODO: only possible batch size is 1
        for i, target_trees in enumerate(batches_progress):
            model.train()
            optimizer.zero_grad()

            # tqdm.write(f"example {i}")
            # tqdm.write(f"target tree")
            # tqdm.write(str(target_trees[0]))

            loss = model(target_trees[0], device)

            if loss.grad_fn is None:
                tqdm.write("[warning] skipped a tree")
                continue

            loss.backward()
            optimizer.step()

            mean_loss_list.append(loss.item())
            update_progress_bar(epoch, loss.item())

            mean_loss_list.append(loss.item())

        if not scheduler is None:
            scheduler.step()

        model.eval()
        pred_trees = list()
        for valid_tree in dataset.valid_trees:
            with torch.no_grad():
                pred_tree = model.make_tree(valid_tree.tokens, device, Intent)
                pred_trees.append(pred_tree)
        exact_accuracy = IntentTree.exact_accuracy_metric(
            pred_trees, dataset.valid_trees
        )
        labeled_precision, labeled_recall, labeled_f1 = IntentTree.labeled_bracketed_metric(
            pred_trees, dataset.valid_trees
        )
        for pred_tree in pred_trees[:10]:
            tqdm.write(str(pred_tree))
        tqdm.write("validation exact accuracy : {:4f}".format(exact_accuracy))
        tqdm.write("validation labeled precision : {:4f}".format(labeled_precision))
        tqdm.write("validation labeled recall : {:4f}".format(labeled_recall))
        tqdm.write("validation labeled f1 : {:4f}".format(labeled_f1))

        tqdm.write(f"mean loss : {sum(mean_loss_list) / len(mean_loss_list)}")

    return model


if __name__ == "__main__":
    config = Config("./configs/default-config.json")
    arg_parser = argparse.ArgumentParser(argparse.ArgumentDefaultsHelpFormatter)
    arg_parser.add_argument(
        "-en",
        "--epochs-nb",
        type=int,
        default=config["epochs_nb"],
        help="Number of epochs",
    )
    arg_parser.add_argument(
        "-bz",
        "--batch-size",
        type=int,
        default=config["batch_size"],
        help="Size of batches",
    )
    arg_parser.add_argument(
        "-lr",
        "--learning-rate",
        type=float,
        default=config["learning_rate"],
        help="learning rate",
    )
    arg_parser.add_argument(
        "-cf",
        "--config-file",
        type=str,
        default=None,
        help="Config file overriding the default-config.json default config",
    )
    args = arg_parser.parse_args()
    if args.config_file:
        config.load_from_file_(args.config_file)
    else:
        config.update_(vars(args))

    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    print("[info] loading datas...")
    dataset = Dataset.from_file("./datas/train.tsv", 0.9)
    model = TreeScorer(tokenizer)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    train_(
        model,
        dataset,
        device,
        optimizer,
        None,  # scheduler,
        config["epochs_nb"],
        config["batch_size"],
    )
