import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


import argparse
from typing import Optional, List

import torch
from torch.optim.optimizer import Optimizer
from torch.nn.modules.loss import _Loss
from torch.optim.lr_scheduler import _LRScheduler
from transformers import BertTokenizer
from tqdm import tqdm

from tree import IntentTree
from datas import Dataset
from config import Config
from model import TreeScorer


def hinge_loss(
    pred_score: torch.Tensor, target_score: torch.Tensor
) -> Optional[torch.Tensor]:
    if (1 - target_score + pred_score).item() <= 0:
        return None
    return 1 - target_score + pred_score


def train_(
    model: TreeScorer,
    dataset: Dataset,
    device: torch.device,
    optimizer: Optimizer,
    scheduler: Optional[_LRScheduler],
    epochs_nb: int,
    batch_size: int,
):
    model.train()
    model.to(device)

    for epoch in range(epochs_nb):

        mean_loss_list = []

        batches_progress = tqdm(
            dataset.batches(batch_size), total=dataset.batches_nb(batch_size)
        )

        def update_progress_bar(epoch: int, mean_loss_list: List[float]):
            if len(mean_loss_list) > 0:
                batches_progress.set_description(
                    "[epoch:{}][mean loss:{:2f}]".format(
                        epoch + 1, sum(mean_loss_list) / len(mean_loss_list)
                    )
                )

        # TODO: only possible batch size is 1
        for sequences, target_trees in batches_progress:
            optimizer.zero_grad()
            sequences = sequences.to(device)

            with torch.no_grad():
                model.eval()
                model.clear_tree_cache_()
                pred_tree = model.make_tree(sequences, None)
            model.train()
            print("[debug]")
            print(pred_tree)

            if pred_tree == target_trees[0]:
                mean_loss_list.append(0)
                update_progress_bar(epoch, mean_loss_list)
                continue

            target_score = model(target_trees[0], device)
            pred_score = model(pred_tree, device)

            loss = hinge_loss(pred_score, target_score)
            if loss is None:
                mean_loss_list.append(0)
                update_progress_bar(epoch, mean_loss_list)
                continue

            loss.backward()
            optimizer.step()

            mean_loss_list.append(loss.item())
            update_progress_bar(epoch, mean_loss_list)

            mean_loss_list.append(loss.item())

        if not scheduler is None:
            scheduler.step()

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
    dataset = Dataset.from_file("./datas/train.tsv", tokenizer)
    model = TreeScorer(tokenizer)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.Adam(model.parameters())

    train_(
        model,
        dataset,
        device,
        optimizer,
        None,  # scheduler,
        config["epochs_nb"],
        config["batch_size"],
    )
