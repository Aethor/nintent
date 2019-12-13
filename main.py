import argparse
from typing import Optional

import torch
from torch.optim.optimizer import Optimizer
from torch.nn.modules.loss import _Loss
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm

from tree import IntentTree
from config import Config
from model import TreeMaker


# TODO: batches
def train_(
    model: TreeMaker,
    batches,
    device: torch.device,
    optimizer: Optimizer,
    scheduler: Optional[_LRScheduler],
    loss_fn: _Loss,
    epochs_nb: int,
):
    model.train()
    model.to(device)

    batches_progress = tqdm(batches)

    for epoch in range(epochs_nb):

        mean_loss_list = []

        for batch in batches_progress:
            optimizer.zero_grad()

            pred = model(batch.X)  # TODO: batch mockup

            loss = loss_fn(pred, batch.y)  # TODO: batch mockup
            loss.backward()
            optimizer.step()

            batches_progress.set_description(f"[epoch:{epoch + 1}][loss:{loss.item()}]")
            mean_loss_list.append(loss.item())

        if len(mean_loss_list) > 0:
            mean_loss = sum(mean_loss_list) / len(mean_loss_list)
            print(f"mean loss : {mean_loss}")
        else:
            print("[warning] mean loss cannot be reported")

        if not scheduler is None:
            scheduler.step()

    return model


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
