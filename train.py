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
from tqdm.auto import tqdm

from tree import IntentTree, Intent
from datas import Dataset
from config import Config
from model import TreeMaker
from score import score


def train_(
    model: TreeMaker,
    train_dataset: Dataset,
    valid_dataset: Dataset,
    device: torch.device,
    optimizer: Optimizer,
    scheduler: Optional[_LRScheduler],
    epochs_nb: int,
    batch_size: int,
    verbose: bool = False,
):
    model.to(device)

    for epoch in range(epochs_nb):

        mean_loss_list = []

        batches_progress = tqdm(
            train_dataset.batches(batch_size, device),
            total=train_dataset.batches_nb(batch_size),
        )

        # TODO: only possible batch size is 1
        for i, target_trees in enumerate(batches_progress):
            model.train()
            optimizer.zero_grad()

            loss = model(
                target_trees[0],
                device,
                target_trees[0].tokens,
                target_trees[0].span_coords,
            )

            loss.backward()
            optimizer.step()

            mean_loss_list.append(loss.item())
            batches_progress.set_description(
                "[e:{}][l:{:10.4f}]".format(epoch + 1, loss.item())
            )

        if not scheduler is None:
            scheduler.step()

        tqdm.write("scoring train trees...")
        train_metrics = score(
            model,
            train_dataset.trees[: int(0.10 * len(train_dataset.trees))],
            device,
            verbose,
        )
        tqdm.write("scoring validation trees...")
        valid_metrics = score(model, valid_dataset.trees, device, verbose)

        tqdm.write("train exact accuracy : {:10.4f}".format(train_metrics[0]))
        tqdm.write("train labeled precision : {:10.4f}".format(train_metrics[1]))
        tqdm.write("train labeled recall : {:10.4f}".format(train_metrics[2]))
        tqdm.write("train labeled f1 : {:10.4f}".format(train_metrics[3]))

        tqdm.write("validation exact accuracy : {:10.4f}".format(valid_metrics[0]))
        tqdm.write("validation labeled precision : {:10.4f}".format(valid_metrics[1]))
        tqdm.write("validation labeled recall : {:10.4f}".format(valid_metrics[2]))
        tqdm.write("validation labeled f1 : {:10.4f}".format(valid_metrics[3]))

        tqdm.write(
            "mean loss : {:10.4f}".format(sum(mean_loss_list) / len(mean_loss_list))
        )

    return model


if __name__ == "__main__":

    config = Config("./configs/default-train-config.json")
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
        "-tdur",
        "--train-datas-usage-ratio",
        type=float,
        default=config["train_datas_usage_ratio"],
        help="train datas usage ratio (between 0 and 1)",
    )
    arg_parser.add_argument(
        "-vdur",
        "--validation-datas-usage-ratio",
        type=float,
        default=config["validation_datas_usage_ratio"],
        help="validation datas usage ratio (between 0 and 1)",
    )
    arg_parser.add_argument(
        "-mp",
        "--model-path",
        type=str,
        default=config["model_path"],
        help="path where the model will be saved",
    )
    arg_parser.add_argument(
        "-cf",
        "--config-file",
        type=str,
        default=None,
        help="Config file overriding default-train-config.json",
    )
    args = arg_parser.parse_args()
    if args.config_file:
        config.load_from_file_(args.config_file)
    else:
        config.update_(vars(args))

    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    print("[info] loading datas...")
    train_dataset, valid_dataset = Dataset.from_files(
        ["./datas/train.tsv", "./datas/eval.tsv"],
        [config["train_datas_usage_ratio"], config["validation_datas_usage_ratio"]],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    intent_weights, slot_weights = train_dataset.class_weights()
    intent_weights = torch.tensor(intent_weights).to(device)
    slot_weights = torch.tensor(slot_weights).to(device)

    model = TreeMaker()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    train_(
        model,
        train_dataset,
        valid_dataset,
        device,
        optimizer,
        None,  # scheduler,
        config["epochs_nb"],
        config["batch_size"],
        True,
    )

    torch.save(model.state_dict(), config["model_path"])
